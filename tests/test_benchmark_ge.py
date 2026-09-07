from pathlib import Path

import pytest

from falcons.aircraft.config import PLANES
from falcons.benchmark import TOLERANCE
from falcons.benchmark.diff import compare_csv
from falcons.benchmark.ground_effect import RATIOS, geometry, run_energy, run_trim, theory_pct

GOLD = Path(__file__).parent / "golden"


def test_theory_is_lifting_line_shape():
    """theory_pct is a thrust CHANGE, so the lifting-line shape is a large NEGATIVE saving deep in
    ground effect rising monotonically to zero out of it -- not a positive quantity decaying to
    zero. `geometry` returns the tuple (span, TR, AR, cg_z); `theory_pct` takes metres plus the
    dict `build` carries, so the keys are supplied here."""
    span, TR, AR, cg_z = geometry("Airship_V7")
    P = dict(span=span, TR=TR, AR=AR, cg_z=cg_z)
    pct = [theory_pct(r * span, P) for r in RATIOS]
    assert pct[0] < -10.0                                  # h/b = 0.25: a large saving
    assert all(a < b for a, b in zip(pct, pct[1:]))         # monotone toward zero with h/b
    assert abs(pct[-1]) < 0.01                             # h/b = 6: out of ground effect


@pytest.mark.cuda
def test_trim_table_matches_golden(tmp_path):
    out = run_trim(PLANES, results_dir=tmp_path)
    assert compare_csv(out, GOLD / "ge_trim.csv", TOLERANCE["ge_trim.csv"]) == []


@pytest.mark.slow
@pytest.mark.cuda
def test_energy_table_matches_golden(tmp_path, gate_copy):
    out = run_energy(PLANES, results_dir=tmp_path)
    assert compare_csv(gate_copy(out), GOLD / "ge_energy.csv", TOLERANCE["ge_energy.csv"]) == []
