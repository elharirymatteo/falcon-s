import csv
import inspect
import json
import shutil
from pathlib import Path

import pytest

from falcons.benchmark.reproduce import compare, environment

GOLD = Path(__file__).parent / "golden"
GATED = ["altitude.csv", "ge_trim.csv", "ge_energy.csv", "maneuvers.csv", "robustness.csv"]


def test_environment_records_versions():
    e = environment()
    assert e["torch"].startswith("2.7.0") and e["warp"] == "1.8.1" and e["python"].startswith("3.10")


def test_environment_fingerprints_the_machine():
    """Every field the claim needs is present; the two that depend on the surroundings (a GPU, a
    checkout) are allowed to be absent but not to be missing."""
    e = environment()
    assert set(e) == {"date", "python", "torch", "cuda", "warp", "gpu", "driver", "commit"}
    assert e["commit"] is None or len(e["commit"]) == 40


def _shipped(d, names=GATED):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        shutil.copy(GOLD / n, d / n)
    return d


def test_compare_passes_when_every_table_is_identical(tmp_path):
    lines, code = compare(_shipped(tmp_path / "out"), _shipped(tmp_path / "results"))
    tables = [ln for ln in lines if ln.startswith("- ")]
    assert code == 0
    assert tables and all("PASS" in ln for ln in tables), lines
    assert any("excursions" in ln and "0 of" in ln for ln in lines), lines


def test_compare_fails_on_a_drifted_row(tmp_path):
    """PPO rows are gated exactly, so a metre of altitude RMSE moving is a FAIL, not a rounding."""
    out, results = _shipped(tmp_path / "out"), _shipped(tmp_path / "results")
    with open(out / "altitude.csv") as f:
        rows = list(csv.DictReader(f)); cols = list(rows[0])
    ppo = next(r for r in rows if r["method"] == "PPO")
    ppo["rmse"] = f"{float(ppo['rmse']) + 1.0}"
    with open(out / "altitude.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    lines, code = compare(out, results)
    assert code == 1 and any("altitude.csv: FAIL" in ln for ln in lines)


def test_compare_refuses_to_pass_on_an_empty_results_dir(tmp_path):
    """A reproduction against a results/ that ships no table compared nothing; that is code 2, not
    a pass, so `falcons reproduce` before Task 11 populates results/ cannot look green."""
    lines, code = compare(_shipped(tmp_path / "out"), tmp_path / "results")
    assert code == 2 and "NOTHING COMPARED" in lines[-1]


def test_compare_reports_a_table_the_shipped_dir_lacks(tmp_path):
    out = _shipped(tmp_path / "out")
    results = _shipped(tmp_path / "results", ["altitude.csv"])
    lines, code = compare(out, results)
    assert code == 0                                              # the one table it can gate passes
    assert sum("no shipped file" in ln for ln in lines) == len(GATED) - 1


def _stub(monkeypatch, module, name, ret=None):
    """Replace a runner with a recorder that first checks the call would have fitted the real
    signature, so the wiring test fails on a renamed argument instead of two hours later."""
    real = getattr(module, name)
    calls = []

    def stub(*a, **kw):
        inspect.signature(real).bind(*a, **kw)
        calls.append((a, kw))
        return ret

    monkeypatch.setattr(module, name, stub)
    return calls


def test_run_calls_every_protocol_and_writes_its_fingerprint(monkeypatch, tmp_path):
    """End to end, `run` is only exercised by the slow gate, and that one skips until results/ is
    populated; this drives the same orchestration with recorders in place of the runners."""
    from falcons.benchmark import altitude, figures, ground_effect, maneuvers, robustness, tables
    from falcons.benchmark.reproduce import run
    alt = _stub(monkeypatch, altitude, "run")
    trim = _stub(monkeypatch, ground_effect, "run_trim")
    energy = _stub(monkeypatch, ground_effect, "run_energy")
    man = _stub(monkeypatch, maneuvers, "run")
    rob = _stub(monkeypatch, robustness, "run")
    emit = _stub(monkeypatch, tables, "emit")
    render = _stub(monkeypatch, figures, "render", ret=[])
    out = tmp_path / "out"

    assert run("ckpts", tmp_path / "results", out) == 2      # nothing shipped to compare against

    planes = ["Airship_V7", "Volantex_Ranger"]
    assert alt[0] == ((planes, "ckpts", out), {"mppi_seeds": (0, 1, 2)})
    assert man[0] == ((planes, "ckpts", out), {"refresh": True, "mppi_draws": 5, "mppi_seed": 0})
    assert rob[0] == ((planes, "ckpts", out), {}) and emit[0] == ((out,), {})
    assert render[0] == ((None, "ckpts", out), {}) and len(trim) == len(energy) == 1
    assert set(json.loads((out / "environment.json").read_text())) == \
        {"date", "python", "torch", "cuda", "warp", "gpu", "driver", "commit"}
    assert "NOTHING COMPARED" in (out / "REPORT.md").read_text()


def test_cli_reproduce_propagates_the_exit_code(monkeypatch, tmp_path):
    """`falcons reproduce` is a gate a script reads: main() hands back the runner's code, the 2 of
    an empty comparison included, rather than collapsing it to success."""
    import falcons.benchmark.reproduce as reproduce
    from falcons.cli import main
    seen = []
    monkeypatch.setattr(reproduce, "run", lambda *a: (seen.append(a), 2)[1])
    assert main(["reproduce", "--out", str(tmp_path)]) == 2
    assert seen[0][2] == tmp_path


@pytest.mark.slow
@pytest.mark.cuda
def test_full_reproduce_passes(tmp_path):
    from falcons.benchmark.reproduce import run
    from falcons.paths import CKPT_DIR, RESULTS_DIR
    if not (RESULTS_DIR / "altitude.csv").exists():
        pytest.skip("results/ not populated yet (Task 11)")
    assert run(CKPT_DIR, RESULTS_DIR, tmp_path) == 0


def _one_table(d, name, text):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text)
    return d


def test_drift_on_a_deterministic_row_fails_but_a_draw_excursion_does_not(tmp_path, monkeypatch):
    """The split R24 asks for: a mismatch on a row the rule calls deterministic breaks the claim; a
    mismatch on a sampling row is an excursion, reported in full and tolerated under the cap."""
    import falcons.benchmark.reproduce as rp
    rule = {"key": ["method"], "exact_if": {"method": ["PPO"]}, "sigma_if": {"method": ["MPPI"]},
            "sigma_k": 1.0, "abs": 0.0}
    monkeypatch.setattr(rp, "TOLERANCE", {"altitude.csv": rule})
    head = "method,rmse,rmse_std"
    gold = _one_table(tmp_path / "results", "altitude.csv", f"{head}\nPPO,0.288,0.0\nMPPI,30.0,1.0\n")

    # MPPI 5 sigma out, PPO untouched -> an excursion, not a failure
    out = _one_table(tmp_path / "a", "altitude.csv", f"{head}\nPPO,0.288,0.0\nMPPI,35.0,1.0\n")
    lines, code = rp.compare(out, gold)
    assert code == 0, lines
    assert any("altitude.csv: PASS" in ln and "1 sampled-row excursion" in ln for ln in lines)
    assert any("Sampled-row excursions" in ln and "1 of at most" in ln for ln in lines)
    assert any("MPPI" in ln and "35.0" in ln for ln in lines)      # listed in full

    # PPO moved -> drift, always a failure, however few
    out = _one_table(tmp_path / "b", "altitude.csv", f"{head}\nPPO,1.288,0.0\nMPPI,30.0,1.0\n")
    lines, code = rp.compare(out, gold)
    assert code == 1 and any("altitude.csv: FAIL" in ln for ln in lines)


def test_excursions_past_the_cap_fail(tmp_path, monkeypatch):
    """The backstop: draws are allowed to wander, but not all of them at once."""
    import falcons.benchmark.reproduce as rp
    rule = {"key": ["method", "cell"], "sigma_if": {"method": ["MPPI"]}, "sigma_k": 1.0, "abs": 0.0}
    monkeypatch.setattr(rp, "TOLERANCE", {"altitude.csv": rule})
    monkeypatch.setattr(rp, "MAX_SIGMA_EXCURSIONS", 2)
    head = "method,cell,rmse,rmse_std"
    rows = lambda v: head + "".join(f"\nMPPI,c{i},{v},1.0" for i in range(3))
    gold = _one_table(tmp_path / "results", "altitude.csv", rows(30.0))
    out = _one_table(tmp_path / "a", "altitude.csv", rows(35.0))
    lines, code = rp.compare(out, gold)
    assert code == 1 and any("TOO MANY: 3 > 2" in ln for ln in lines)
