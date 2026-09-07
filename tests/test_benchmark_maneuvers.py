import json
import os
import shutil
from pathlib import Path

import pytest

from falcons.benchmark import TOLERANCE
from falcons.benchmark.diff import compare_csv
from falcons.benchmark.streams import maneuver_stream, MANEUVERS
from falcons.paths import CKPT_DIR

GOLD = Path(__file__).parent / "golden"
# The 39 MB rollout cache lives outside the repo; point FALCONS_TRACE_FIXTURE at it to run the
# replay gate, which is the only way to reproduce the MPPI rows exactly (nothing re-flies).
FIX = Path(os.environ.get("FALCONS_TRACE_FIXTURE", "/home/matteo/Projects/falcon-s-goldens/traces"))


def test_streams_have_four_names_and_equal_length():
    s = [maneuver_stream(m, 0.6, 1.5, 28.0) for m in MANEUVERS]
    assert len(MANEUVERS) == 4 and all(len(x[0]) == len(x[1]) == len(x[2]) for x in s)


@pytest.mark.cuda
def test_table_replays_from_cached_traces(tmp_path):
    """The gate: the archive's own rollout cache, replayed through this module, must reproduce
    every cell of the golden table byte-for-byte -- MPPI included, since no rollout is flown."""
    if not FIX.exists():
        pytest.skip("trace fixture not present")
    from falcons.benchmark.maneuvers import run
    shutil.copytree(FIX, tmp_path / "traces")
    out = run(["Airship_V7", "Volantex_Ranger"], results_dir=tmp_path)      # nothing re-flies
    rule = {**TOLERANCE["maneuvers.csv"], "sigma_if": {}}                   # exact, MPPI included
    assert compare_csv(out, GOLD / "maneuvers.csv", rule) == []
    # the spread the manuscript quotes for the MPPI cells travels beside the table, so it is gated
    # with it rather than left to be regenerated unchecked
    with open(tmp_path / "mppi_variance.json") as f, open(GOLD / "mppi_variance.json") as g:
        assert json.load(f) == json.load(g)


@pytest.mark.slow
@pytest.mark.cuda
def test_one_learned_cell_reflies_identically(tmp_path):
    from falcons.benchmark.maneuvers import cell
    c = cell("Volantex_Ranger", "helix", "PPO", ckpt_dir=CKPT_DIR, cache_dir=tmp_path,
             refresh=True, tol_deg=8.6)
    assert abs(c["phi"]["rmse"] - 0.42) < 0.005      # golden maneuvers.csv: Volantex helix PPO 0.42
