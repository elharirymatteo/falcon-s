import csv
from pathlib import Path

import pytest

from falcons.benchmark import TOLERANCE
from falcons.benchmark.diff import compare_csv
from falcons.benchmark.robustness import steps_from_ms, obs_noise_sigma, SEVERITIES

GOLD = Path(__file__).parent / "golden"


def test_delay_steps_and_noise_monotone():
    assert steps_from_ms(20) == 2 and steps_from_ms(100) == 10
    s = [obs_noise_sigma(sev, 1.0, 1.0) for sev in SEVERITIES]
    assert all(a <= b for a, b in zip(s[0], s[1])) and all(a <= b for a, b in zip(s[1], s[2]))


@pytest.mark.cuda
def test_obs_noise_draw_is_a_property_of_the_cell():
    """Seeded per (aircraft, severity): the same cell redraws the same noise in a new process, a
    different severity draws differently. The archive's unseeded global generator did neither."""
    import torch
    from falcons.benchmark.robustness import ObsNoise, noise_seed
    assert noise_seed("Airship_V7", "severe") == noise_seed("Airship_V7", "severe")
    assert noise_seed("Airship_V7", "severe") != noise_seed("Volantex_Ranger", "severe")
    obs = torch.zeros(1, 15, device="cuda")
    sig = obs_noise_sigma("severe", 14.0, 14.0)
    draw = lambda sev: ObsNoise(lambda o: o, sig, noise_seed("Airship_V7", sev))(obs)
    assert torch.equal(draw("severe"), draw("severe"))
    assert not torch.equal(draw("severe"), draw("mild"))


def _rows_of_method(gold, method, out):
    """The shipped golden is the archive's whole sweep, which also carried SAC and SAC+CAPS rows.
    The CAPS weights are not in the shipped checkpoint set and the paper reports PPO alone
    (emit_tex's robustness table and plot_robustness_paper both filter method == "PPO"), so the
    gate compares the PPO rows -- all of them, at full tolerance."""
    with open(gold) as f, open(out, "w", newline="") as g:
        r = csv.DictReader(f)
        w = csv.DictWriter(g, fieldnames=r.fieldnames)
        w.writeheader()
        for row in r:
            if row["method"] == method:
                w.writerow(row)
    return out


@pytest.mark.slow
@pytest.mark.cuda
def test_table_matches_golden(tmp_path, gate_copy):
    from falcons.benchmark.robustness import run
    gold = _rows_of_method(GOLD / "robustness.csv", "PPO", tmp_path / "golden_ppo.csv")
    out = run(["Airship_V7", "Volantex_Ranger"], results_dir=tmp_path)
    assert compare_csv(gate_copy(out), gold, TOLERANCE["robustness.csv"]) == []
