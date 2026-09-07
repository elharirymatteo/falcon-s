"""Regenerate every table and figure from the checkpoints into a fresh directory, diff each CSV
against the shipped results/ under TOLERANCE, and write a report.

Exit code: 0 every shipped table reproduced within tolerance, 1 at least one did not, 2 there was
nothing to compare against -- a run over a results/ that ships none of the tables regenerates them
happily and proves nothing, so it is not allowed to look like a pass.

A mismatch on a row the rule treats as deterministic is DRIFT and always fails; one on a sampling
row is an EXCURSION, tolerated up to MAX_SIGMA_EXCURSIONS. Why that budget is what it is, and why
every tolerance constant is the number it is: docs/reproduction-tolerances.md.
"""
import csv
import json
import platform
import subprocess
from datetime import date
from pathlib import Path

from falcons.benchmark import TOLERANCE
from falcons.benchmark.diff import compare_csv

MAX_SIGMA_EXCURSIONS = 8        # one MPPI cell's worth; chance alone expects 0.16
                                # (docs/reproduction-tolerances.md)


def environment():
    """What a re-run has to match: the fingerprint of the machine and the pinned stack. The commit
    is whatever checkout the package is imported from, or None when it is an installed copy with no
    git around it."""
    import torch
    import warp
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):        # no nvidia-smi, or a wedged driver
        drv = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=10, cwd=Path(__file__).parent).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):        # no git, no checkout, or a wedged git
        commit = None
    return {"date": date.today().isoformat(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "warp": warp.__version__,
            "gpu": gpu, "driver": drv, "commit": commit}


def sampled_keys(gold, rule):
    """The row keys `rule` treats as draws rather than as deterministic rows -- the rows whose
    mismatches are excursions instead of drift."""
    sig = rule.get("sigma_if", {})
    if not sig:
        return set()
    with open(gold) as f:
        return {tuple(r[k] for k in rule["key"]) for r in csv.DictReader(f)
                if any(r.get(col) in vals for col, vals in sig.items())}


def compare(out_dir, results_dir):
    """Diff every regenerated table in `out_dir` against the shipped one in `results_dir` under its
    TOLERANCE rule. Returns (report lines, exit code).

    Mismatches are split (see docs/reproduction-tolerances.md): on a deterministic row a mismatch
    is DRIFT and always fails; on a sampling row it is an EXCURSION, reported in full under its
    own heading and tolerated up to MAX_SIGMA_EXCURSIONS.

    throughput.csv is skipped: it is wall-clock, `run` does not regenerate it, and its rule exists
    for a human reading the numbers rather than for a machine deciding whether the tables reproduced.

    A table the shipped directory does not carry cannot be compared, and a run in which NOTHING was
    compared is code 2 rather than 0: an empty comparison is vacuous, not a success."""
    lines, fails, compared, excursions = [], 0, 0, []
    for name, rule in TOLERANCE.items():
        if name == "throughput.csv":
            continue
        gold = Path(results_dir) / name
        if not gold.exists():
            lines.append(f"- {name}: no shipped file to compare")
            continue
        compared += 1
        bad = compare_csv(Path(out_dir) / name, gold, rule)
        sampled = sampled_keys(gold, rule)
        drift, drawn = [], []
        for b in bad:
            # a "row sets differ" line carries no row key and is never an excursion
            (drawn if any(b.startswith(f"{k} ") for k in sampled) else drift).append(b)
        fails += bool(drift)
        excursions += [f"{name}: {b}" for b in drawn]
        note = f"  ({len(drawn)} sampled-row excursion{'s' if len(drawn) != 1 else ''})" if drawn else ""
        lines.append(f"- {name}: {'PASS' if not drift else 'FAIL'}{note}"
                     + "".join(f"\n    - {b}" for b in drift[:20]))
    if not compared:
        lines.append(f"\nNOTHING COMPARED: {Path(results_dir)} ships none of the tables, so this "
                     f"run regenerated them against nothing.")
        return lines, 2
    over = len(excursions) > MAX_SIGMA_EXCURSIONS
    lines += ["", f"Sampled-row excursions (MPPI draws; not a reproduction failure): "
                  f"{len(excursions)} of at most {MAX_SIGMA_EXCURSIONS} tolerated."]
    lines += [f"  - {e}" for e in excursions]
    if over:
        lines.append(f"  TOO MANY: {len(excursions)} > {MAX_SIGMA_EXCURSIONS}; the sampling rows "
                     f"are no longer landing where the shipped spread says they should.")
    return lines, 1 if (fails or over) else 0


def run(ckpt_dir, results_dir, out_dir):
    """Re-fly the whole benchmark into `out_dir` and report how it compares with `results_dir`.

    The two learned-vs-classical protocols run on the two WIG airframes; the ground-effect
    sweep runs on every airframe, its trim arm needing no policy. Maneuvers refresh their rollout
    cache (refresh=True) rather than replaying the shipped one -- a reproduction that reads the
    cache proves only that the table emitter is deterministic."""
    from falcons.aircraft.config import PLANES
    from falcons.benchmark import altitude, ground_effect, maneuvers, robustness
    from falcons.benchmark.figures import render
    from falcons.benchmark.tables import emit
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    planes = ["Airship_V7", "Volantex_Ranger"]
    altitude.run(planes, ckpt_dir, out, mppi_seeds=(0, 1, 2))
    ground_effect.run_trim(PLANES, out)
    ground_effect.run_energy(PLANES, ckpt_dir, out)
    maneuvers.run(planes, ckpt_dir, out, refresh=True, mppi_draws=5, mppi_seed=0)
    robustness.run(planes, ckpt_dir, out)
    emit(out)

    env = environment()
    with open(out / "environment.json", "w") as f:
        json.dump(env, f, indent=1)
    lines, code = compare(out, results_dir)
    report = [f"# reproduce {env['date']} on {env['gpu']}", ""] + lines
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    render(None, ckpt_dir, out)                         # after the report: a figure that fails to
                                                        # draw must not discard hours of comparison
    return code
