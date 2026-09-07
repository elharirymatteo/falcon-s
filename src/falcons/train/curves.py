"""Append-only CSV learning curves, one file per training run.

log_curve("sac_attitude_Airship_V7_s0", env_steps, curves_dir, rob=..., phi_rmse=...)
-> <curves_dir>/sac_attitude_Airship_V7_s0.csv

The first call for a name in a process truncates the file (fresh curve per run);
later calls append. Columns come from the first call's metric names.
"""
import csv
import os

from falcons.paths import RESULTS_DIR

_started = set()


def log_curve(name, step, curves_dir=RESULTS_DIR / "curves", **metrics):
    os.makedirs(curves_dir, exist_ok=True)
    path = os.path.join(curves_dir, f"{name}.csv")
    fresh = name not in _started
    _started.add(name)
    with open(path, "w" if fresh else "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["env_steps", *metrics])
        w.writerow([step, *(f"{v:.6g}" for v in metrics.values())])
