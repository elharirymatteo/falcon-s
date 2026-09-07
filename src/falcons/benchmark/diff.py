"""Compare a regenerated CSV against a golden one under a TOLERANCE rule. Returns a list of
human-readable mismatches; an empty list is a pass.

Only the golden's columns are compared: a column present in the new CSV but not in the golden is
ignored, which is how a new statistic (settled_frac_std) ships without re-freezing the golden."""
import csv
import math


def _rows(path, key):
    with open(path) as f:
        rs = list(csv.DictReader(f))
    return {tuple(r[k] for k in key): r for r in rs}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def compare_csv(new, gold, rule):
    key = rule["key"]
    a, b = _rows(new, key), _rows(gold, key)
    out = []
    if set(a) != set(b):
        out.append(f"row sets differ: missing {sorted(set(b) - set(a))}, extra {sorted(set(a) - set(b))}")
    for k in sorted(set(a) & set(b)):
        ra, rb = a[k], b[k]
        mode = "exact"
        loose_abs = rule.get("loose_abs", {})
        for col, vals in rule.get("sigma_if", {}).items():
            if ra.get(col) in vals: mode = "sigma"
        for col, vals in rule.get("loose_if", {}).items():
            if ra.get(col) in vals:
                mode = "loose"
                # a family with a bound of its own (loose_abs_by) is judged on that, not on the
                # shared one: one disturbance's spread says nothing about another's
                loose_abs = rule.get("loose_abs_by", {}).get(ra[col], loose_abs)
        # A cell that loses draws reports a survivor-conditioned mean: the metric is averaged over
        # the draws that stayed inside the envelope, and WHICH draws those are changes between
        # sweeps. Measured on the Volantex s-turn hdot_rmse_ms, the spread between sweeps (0.241
        # over three sweeps) ran ~2x the within-sweep spread across draws (0.115), so the stored
        # sigma understates the scale of the quantity actually being compared. Widen it.
        partial_k = 1.0
        if mode == "sigma":
            surv = _num(rb.get("survival"))
            if surv is not None and surv < 1.0:
                partial_k = rule.get("sigma_partial_survival_k", 2.0)
        for col in rb:
            if col in key or col.endswith(rule.get("sigma_suffix", "_std")):
                continue
            if mode == "sigma" and col in rule.get("skip_cols_if_sigma", []):
                continue
            va, vb = _num(ra.get(col)), _num(rb.get(col))
            if va is None or vb is None:
                if (ra.get(col) or "") != (rb.get(col) or ""): out.append(f"{k} {col}: {ra.get(col)!r} != {rb.get(col)!r}")
                continue
            if math.isnan(va) != math.isnan(vb):   # one side dead, the other not: never a pass
                out.append(f"{k} {col}: {va} vs golden {vb} (NaN mismatch)")
                continue
            if math.isnan(va) and math.isnan(vb):
                continue
            if mode == "sigma":
                # A sigma row is a sample, and the row it is compared against is another sample of
                # the same statistic: two independent draws sit ~sqrt(2)*sigma apart on average, so a
                # 1-sigma band would reject an honest re-run about half the time. `sigma_k` widens the
                # band to k standard deviations; the floor is an absolute minimum, never scaled.
                sd_col = rule.get("sigma_cols", {}).get(col, col + rule.get("sigma_suffix", "_std"))
                sd = (_num(rb.get(sd_col)) or 0.0) * partial_k
                tol = max(sd * rule.get("sigma_k", 1.0),
                          rule.get("sigma_floor", {}).get(col, rule.get("abs", 0.0)))
                # R26, applied last so it caps the floor as well as the band: a tolerance wider
                # than half a bounded statistic's domain tests nothing; beyond a quarter turn of
                # bank error the executor has plainly failed, whatever its draw spread.
                cap = rule.get("sigma_max", {}).get(col)
                if cap is not None:
                    tol = min(tol, cap)
            elif mode == "loose":
                tol = loose_abs.get(col, rule.get("abs", 0.0))
            elif "rel" in rule:
                tol = abs(vb) * rule["rel"]
            else:
                tol = rule.get("abs", 0.0)
            if abs(va - vb) > tol:
                out.append(f"{k} {col}: {va} vs golden {vb} (tol {tol})")
    return out
