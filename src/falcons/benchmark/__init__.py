"""Benchmark protocols. TOLERANCE is the single statement of what "reproduces" means per table:
learned and LQR rows are exact (seeded rollouts are bit-stable); MPPI is a sampling planner whose
cost reduction uses non-deterministic atomics, so its rows must land within the shipped spread;
wind cells are one Dryden draw each. Why each constant below is the number it is, floor by floor
and cap by cap: docs/reproduction-tolerances.md."""
TOLERANCE = {
    "altitude.csv":   {"sigma_k": 3.0, "key": ["aircraft", "method"], "exact_if": {"method": ["PPO", "SAC", "TD3", "LQR"]},
                       "sigma_if": {"method": ["MPPI"]}, "sigma_suffix": "_std",
                       "skip_cols_if_sigma": ["settling"],
                       "sigma_floor": {"survival": 0.05, "acquired": 0.04, "settled_frac": 0.1,
                                       "rmse": 0.5, "energy": 0.02, "overshoot": 6.0},
                       "sigma_max": {"survival": 0.5, "acquired": 0.5, "settled_frac": 0.5},
                       "abs": 0.0},
    "ge_trim.csv":    {"key": ["aircraft", "h_over_b"], "abs": 1e-9},
    "ge_energy.csv":  {"key": ["aircraft", "h_over_b"], "abs": 0.0},
    "maneuvers.csv":  {"sigma_k": 3.0, "key": ["aircraft", "maneuver", "method"], "exact_if": {"method": ["PPO", "SAC", "TD3", "LQR"]},
                       "sigma_if": {"method": ["MPPI"]}, "sigma_cols": {"phi_rmse_deg": "phi_rmse_std"},
                       "sigma_floor": {"survival": 0.2, "time_in_band_pct": 5.0,
                                       "phi_theil_u": 0.005, "phi_gain": 0.01,
                                       "phi_phase_lag_s": 0.02},
                       "sigma_max": {"survival": 0.5, "time_in_band_pct": 50.0,
                                     "phi_rmse_deg": 45.0},
                       "abs": 0.0},
    "robustness.csv": {"key": ["aircraft", "method", "disturbance", "severity"],
                       "exact_if": {"disturbance": ["nominal", "sensor_delay", "action_delay"]},
                       "loose_if": {"disturbance": ["wind", "obs_noise"]},
                       "loose_abs": {"survival": 0.25, "phi_rmse": 5.0},
                       "loose_abs_by": {"obs_noise": {"survival": 0.0, "phi_rmse": 0.10,
                                                      "hdot_rmse": 0.05, "va_rmse": 0.05}},
                       "abs": 0.0},
    "throughput.csv": {"key": ["backend", "n_envs"], "rel": 0.5},   # wall-clock: hardware-dependent, order of magnitude
}


def eval_one(algo, task, plane, seed=0, maneuver="circle", n=1, plot=False,
             ckpt_dir=None, results_dir=None):
    """One cell of the benchmark, printed. `benchmark altitude/maneuvers` compose this same code.

    The altitude protocol scores a learned policy over one seed checkpoint and a classical
    controller over one sweep per sampler draw, so `n` is the number of MPPI draws (starting at
    `seed`) and the per-sweep aggregates are printed above their mean.
    """
    from pathlib import Path

    import numpy as np

    from falcons.paths import CKPT_DIR, RESULTS_DIR
    ckpt_dir = ckpt_dir or CKPT_DIR
    results_dir = Path(results_dir or RESULTS_DIR)
    if task == "altitude":
        from falcons.benchmark.altitude import run_learned, run_classical, aggregate
        if algo in ("ppo", "sac", "td3"):
            out = aggregate(run_learned(plane, algo, seed, ckpt_dir))
            if plot:
                from falcons.benchmark.figures import altitude_viz
                altitude_viz(algo, plane, ckpt_dir, results_dir)
        else:
            # run_classical returns one row-list per sampler seed; the LQR is a deterministic law
            # and gets a single sweep at draw 0 whatever `seed` and `n` say, so label the sweeps
            # with the draws it actually flew rather than with this command's --seed.
            draws = tuple(range(seed, seed + n)) if algo == "mppi" else (0,)
            per = [aggregate(rows) for rows in run_classical(plane, algo, draws, ckpt_dir)]
            for d, p in zip(draws, per):
                print(f"sweep {d}: " + "  ".join(f"{k} {v:.4g}" for k, v in p.items()))
            out = (per[0] if len(per) == 1
                   else {k: float(np.nanmean([p[k] for p in per])) for k in per[0]})
    else:
        from falcons.benchmark.maneuvers import cell
        from falcons.envs.configs import attitude_env_cfg
        tol = float(np.degrees(attitude_env_cfg(plane).get("sig_phi", 0.15)))
        out = cell(plane, maneuver, algo.upper(), ckpt_dir, results_dir / "traces",
                   refresh=False, mppi_draws=n, tol_deg=tol)
        if plot:
            from falcons.benchmark.figures import maneuver_facet
            maneuver_facet(plane, ckpt_dir, results_dir)
    for k, v in out.items():
        if isinstance(v, (int, float)):
            print(f"{k:14s} {v:.4g}")
    return out
