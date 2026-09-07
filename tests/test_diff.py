from pathlib import Path

from falcons.benchmark import TOLERANCE
from falcons.benchmark.diff import compare_csv

GOLD = Path(__file__).parent / "golden"


def test_golden_equals_itself():
    """Every gated table compares clean against itself under its own rule. Asserting WHICH tables
    were compared is what keeps this honest: the loop skips a missing file, so without the tail
    assertion an absent tests/golden/ passes it without comparing anything at all."""
    compared = []
    for name, rule in TOLERANCE.items():
        p = GOLD / name
        if not p.exists():
            continue
        assert compare_csv(p, p, rule) == []
        compared.append(name)
    # throughput is gated as wall-clock and ships as throughput.json, not a table; every other
    # entry in TOLERANCE has a golden CSV beside it and must have been one of the files compared
    assert compared == [n for n in TOLERANCE if n != "throughput.csv"], compared


def test_sigma_floor_is_the_band_when_the_golden_sigma_is_smaller(tmp_path):
    """MPPI rows: tolerance is max(golden <col>_std, floor[col]); a column with no _std in the
    golden (settled_frac) is judged on the floor alone."""
    rule = {"key": ["method"], "sigma_if": {"method": ["MPPI"]}, "sigma_suffix": "_std",
            "sigma_floor": {"rmse": 0.5, "settled_frac": 0.05}, "abs": 0.0}
    gold = tmp_path / "gold.csv"
    gold.write_text("method,rmse,rmse_std,settled_frac\nMPPI,30.0,0.1,0.50\n")
    inside = tmp_path / "inside.csv"
    inside.write_text("method,rmse,rmse_std,settled_frac\nMPPI,30.4,0.1,0.53\n")
    outside = tmp_path / "outside.csv"
    outside.write_text("method,rmse,rmse_std,settled_frac\nMPPI,30.6,0.1,0.57\n")
    assert compare_csv(inside, gold, rule) == []
    assert len(compare_csv(outside, gold, rule)) == 2      # rmse and settled_frac both past the floor


def test_skip_cols_if_sigma_drops_a_column_the_table_does_not_report(tmp_path):
    """MPPI settling is not a reported quantity (<5 % acquisition), so sigma rows skip it — but
    exact rows still compare it."""
    rule = {"key": ["method"], "sigma_if": {"method": ["MPPI"]}, "sigma_suffix": "_std",
            "skip_cols_if_sigma": ["settling", "settling_std"], "abs": 0.0}
    gold = tmp_path / "gold.csv"
    gold.write_text("method,settling,rmse\nMPPI,52.5,30.0\nLQR,10.0,0.1\n")
    new = tmp_path / "new.csv"
    new.write_text("method,settling,rmse\nMPPI,99.9,30.0\nLQR,10.0,0.1\n")
    assert compare_csv(new, gold, rule) == []
    new.write_text("method,settling,rmse\nMPPI,99.9,30.0\nLQR,99.9,0.1\n")
    out = compare_csv(new, gold, rule)
    assert len(out) == 1 and "settling" in out[0]


def test_nan_in_the_new_csv_against_a_golden_number_is_a_mismatch(tmp_path):
    """aggregate() emits NaN for a sweep with zero survivors; NaN vs a number must never pass."""
    rule = {"key": ["method"], "abs": 0.0}
    gold = tmp_path / "gold.csv"; gold.write_text("method,rmse\nPPO,0.288\n")
    new = tmp_path / "new.csv"; new.write_text("method,rmse\nPPO,nan\n")
    out = compare_csv(new, gold, rule)
    assert len(out) == 1 and "NaN mismatch" in out[0]


def test_a_number_in_the_new_csv_against_a_golden_nan_is_a_mismatch(tmp_path):
    rule = {"key": ["method"], "abs": 0.0}
    gold = tmp_path / "gold.csv"; gold.write_text("method,rmse\nPPO,nan\n")
    new = tmp_path / "new.csv"; new.write_text("method,rmse\nPPO,0.288\n")
    out = compare_csv(new, gold, rule)
    assert len(out) == 1 and "NaN mismatch" in out[0]


def test_loose_abs_by_gives_each_disturbance_family_its_own_bound(tmp_path):
    """Wind and sensor noise are both single draws, but of wildly different size: the shared
    loose_abs would let a 5 deg sensor-noise drift through."""
    rule = {"key": ["disturbance"], "loose_if": {"disturbance": ["wind", "obs_noise"]},
            "loose_abs": {"phi_rmse": 5.0},
            "loose_abs_by": {"obs_noise": {"phi_rmse": 0.10}}, "abs": 0.0}
    gold = tmp_path / "gold.csv"
    gold.write_text("disturbance,phi_rmse\nwind,20.0\nobs_noise,2.0\nnominal,1.0\n")
    new = tmp_path / "new.csv"
    new.write_text("disturbance,phi_rmse\nwind,23.0\nobs_noise,2.05\nnominal,1.0\n")
    assert compare_csv(new, gold, rule) == []                  # both inside their own bound
    new.write_text("disturbance,phi_rmse\nwind,23.0\nobs_noise,2.5\nnominal,1.0\n")
    out = compare_csv(new, gold, rule)                         # 0.5 passes wind's bound, not its own
    assert len(out) == 1 and "obs_noise" in out[0] and "tol 0.1" in out[0]


def test_each_column_of_a_sigma_row_is_judged_on_its_own_sigma(tmp_path):
    """The maneuvers rule: every averaged column of an MPPI cell carries its own `<col>_std`, so
    the band is per column and not one shared number. phi_rmse_deg keeps an explicit sigma_cols
    mapping; the rest resolve by suffix; a column whose spread is ~0 falls back to its floor."""
    rule = {"key": ["method"], "sigma_if": {"method": ["MPPI"]},
            "sigma_cols": {"phi_rmse_deg": "phi_rmse_std"},
            "sigma_floor": {"phi_theil_u": 0.005}, "abs": 0.0}
    head = "method,phi_rmse_deg,phi_rmse_std,va_rmse_ms,va_rmse_ms_std,phi_theil_u,phi_theil_u_std"
    gold = tmp_path / "gold.csv"
    gold.write_text(f"{head}\nMPPI,3.97,5.24,0.19,0.04,0.056,0.000\n")
    new = tmp_path / "new.csv"
    # 4.0 within phi's 5.24; 0.22 within va's own 0.04, not phi's; theil_u on its 0.005 floor
    new.write_text(f"{head}\nMPPI,9.00,5.24,0.22,0.04,0.060,0.000\n")
    assert compare_csv(new, gold, rule) == []
    # va moves 0.10: inside the bank spread, far outside its own
    new.write_text(f"{head}\nMPPI,4.00,5.24,0.29,0.04,0.056,0.000\n")
    out = compare_csv(new, gold, rule)
    assert len(out) == 1 and "va_rmse_ms" in out[0] and "tol 0.04" in out[0]
    # theil_u past its floor, with no spread of its own to hide behind
    new.write_text(f"{head}\nMPPI,4.00,5.24,0.19,0.04,0.070,0.000\n")
    out = compare_csv(new, gold, rule)
    assert len(out) == 1 and "phi_theil_u" in out[0] and "tol 0.005" in out[0]


def test_sigma_k_widens_the_band_to_k_standard_deviations(tmp_path):
    """A sigma row is one sample compared against another sample of the same statistic, so the band
    is k sigma, not one. A learned row in the same table stays exact."""
    from falcons.benchmark.diff import compare_csv
    gold = tmp_path / "g.csv"
    gold.write_text("m,v,v_std\nMPPI,10.0,1.0\nPPO,5.0,0.1\n")
    rule = {"key": ["m"], "sigma_if": {"m": ["MPPI"]}, "sigma_k": 3.0, "abs": 0.0}
    for delta, expect_pass in ((2.5, True), (3.5, False)):        # inside 3 sigma, then outside
        new = tmp_path / f"n{delta}.csv"
        new.write_text(f"m,v,v_std\nMPPI,{10.0 + delta},1.0\nPPO,5.0,0.1\n")
        bad = compare_csv(new, gold, rule)
        assert (not bad) is expect_pass, (delta, bad)
    drift = tmp_path / "d.csv"                                    # the learned row is not a sample
    drift.write_text("m,v,v_std\nMPPI,10.0,1.0\nPPO,5.01,0.1\n")
    assert compare_csv(drift, gold, rule), "an exact row must not inherit the sigma band"


def test_partial_survival_widens_the_sigma_only_when_draws_were_lost(tmp_path):
    """A cell that loses draws reports a survivor-conditioned mean, and which draws survive changes
    between sweeps, so its stored within-sweep sigma understates the scale. The widening must apply
    to that row and NOT to a sigma row that kept every draw."""
    rule = {"key": ["m", "cell"], "sigma_if": {"m": ["MPPI"]}, "sigma_k": 3.0,
            "sigma_partial_survival_k": 2.0, "abs": 0.0}
    head = "m,cell,survival,v,v_std"
    gold = tmp_path / "g.csv"
    gold.write_text(f"{head}\nMPPI,partial,0.80,0.217,0.115\nMPPI,whole,1.00,0.217,0.115\n")
    new = tmp_path / "n.csv"
    # 0.551 away: inside 3*2*0.115 = 0.69 on the partial cell, outside 3*0.115 = 0.345 on the whole one
    new.write_text(f"{head}\nMPPI,partial,0.80,0.768,0.115\nMPPI,whole,1.00,0.768,0.115\n")
    out = compare_csv(new, gold, rule)
    assert len(out) == 1 and "'whole'" in out[0], out
    # and the widening is opt-out-able / defaults sanely when the column is absent altogether
    rule_nosurv = {"key": ["m", "cell"], "sigma_if": {"m": ["MPPI"]}, "sigma_k": 3.0, "abs": 0.0}
    gold.write_text("m,cell,v,v_std\nMPPI,partial,0.217,0.115\n")
    new.write_text("m,cell,v,v_std\nMPPI,partial,0.768,0.115\n")
    assert len(compare_csv(new, gold, rule_nosurv)) == 1     # no survival column: no widening


def test_sigma_max_clamps_a_band_wider_than_half_the_column_domain(tmp_path):
    """R26: survival is bounded on 0-1, so the V7 MPPI cell's 3*2*0.154 = 0.93 band would admit
    every value the column can take. The cap clamps it, and clamps a floor the same way."""
    rule = {"key": ["method"], "sigma_if": {"method": ["MPPI"]}, "sigma_k": 3.0,
            "sigma_max": {"survival": 0.5}, "abs": 0.0}
    head = "method,survival,survival_std"
    gold = tmp_path / "gold.csv"
    gold.write_text(f"{head}\nMPPI,0.71,0.31\n")                  # 3 sigma = 0.93
    inside = tmp_path / "inside.csv"
    inside.write_text(f"{head}\nMPPI,0.25,0.31\n")                # 0.46 away: inside the cap
    assert compare_csv(inside, gold, rule) == []
    outside = tmp_path / "outside.csv"
    outside.write_text(f"{head}\nMPPI,0.09,0.31\n")               # 0.62 away: past it, tol is the cap
    bad = compare_csv(outside, gold, rule)
    assert len(bad) == 1 and "tol 0.5" in bad[0], bad
    # applied last, so an over-wide floor is capped too
    floored = dict(rule, sigma_floor={"survival": 0.9})
    gold.write_text(f"{head}\nMPPI,0.71,0.0\n")
    outside.write_text(f"{head}\nMPPI,0.09,0.0\n")
    bad = compare_csv(outside, gold, floored)
    assert len(bad) == 1 and "tol 0.5" in bad[0], bad
