"""Attitude-maneuver protocol: drive ANY controller through a fixed maneuver stream in
AttitudeEnv(1), score tracking + envelope survival, and collapse the (aircraft x maneuver x
method) cells into maneuvers.csv. Uniform for RL policies and classical controllers, which is
what makes it an identical-conditions comparison on the four named maneuvers.

controller: callable(obs_torch[1,15]) -> action_torch[1,4] in [-1,1].

Run: run(planes) -> <results_dir>/maneuvers.csv + mppi_variance.json.
"""
import csv
import json
import os
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import torch
import warp as wp

from falcons.envs.attitude import AttitudeEnv
from falcons.envs.configs import attitude_env_cfg
from falcons.benchmark.streams import maneuver_stream, DT
from falcons.paths import CKPT_DIR, RESULTS_DIR

SETTLE = 50   # 0.5 s to enter the bank before scoring
G_ACCEL = 9.81


def _rms(x):
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def _nrmse(cmd, err_rms):
    """RMSE / commanded range: dimensionless, so it is comparable across maneuvers and airframes
    (Bohn et al.'s RMSE-over-difficulty, generalized). Near-constant reference (e.g. Va at trim):
    fall back to the mean
    level. Degenerate hold-at-zero (both range and mean ~0, e.g. commanded climb-rate on a level
    turn): nRMSE is undefined -> None (report absolute RMSE for that channel instead)."""
    rng = float(cmd.max() - cmd.min())
    if rng < 1e-6:
        level = abs(float(cmd.mean()))
        if level < 1e-3:
            return None
        rng = level
    return err_rms / rng


def _theil_u(cmd, ach):
    """Theil's inequality coefficient RMS(err)/(RMS(ach)+RMS(cmd)) in [0,1], 0 = perfect. The
    aerospace tracking-validation standard; below 0.25-0.30 counts as good agreement."""
    d = _rms(ach) + _rms(cmd)
    return float(_rms(ach - cmd) / d) if d > 1e-9 else None


def _gain_phase(cmd, ach, dt, max_lag_s=2.0):
    """Regression gain G and phase lag tau (s) of achieved vs commanded, at the lag that maximizes
    normalized cross-correlation (achieved lags commanded -> search non-negative lags only). G is a
    slope (cov/var), so uncorrelated chatter lands in the residual and does NOT inflate the gain.
    Returns (None, None) when the commanded signal is ~constant (gain/phase undefined for a hold)."""
    c = cmd - cmd.mean()
    a = ach - ach.mean()
    if float(np.dot(c, c)) < 1e-6 * len(c):
        return None, None
    max_lag = int(max_lag_s / dt)
    best_lag, best_corr = 0, -np.inf
    for L in range(max_lag + 1):
        aa, cc = (a[L:], c[:len(c) - L]) if L else (a, c)
        if len(cc) < 10:
            break
        den = np.sqrt(float(np.dot(aa, aa)) * float(np.dot(cc, cc)))
        if den < 1e-9:
            continue
        corr = float(np.dot(aa, cc)) / den
        if corr > best_corr:
            best_lag, best_corr = L, corr
    L = best_lag
    aa, cc = (a[L:], c[:len(c) - L]) if L else (a, c)
    gain = float(np.dot(aa, cc) / np.dot(cc, cc))
    return gain, L * dt


def _descriptors(phi_cmd_rad, base_vel):
    """Maneuver difficulty from the commanded bank amplitude: phi_max [deg], coordinated-turn
    radius R = V^2/(g tan phi) [m], load factor n = 1/cos phi. Reported as context for reading a
    cell, never scored. tan-based radius/n diverge at 90 deg -> guarded."""
    phi = float(np.max(np.abs(phi_cmd_rad)))
    tan = np.tan(phi)
    R = base_vel ** 2 / (G_ACCEL * tan) if tan > 1e-3 else float("inf")
    n = 1.0 / np.cos(phi) if np.cos(phi) > 1e-3 else float("inf")
    return dict(phi_max_deg=float(np.degrees(phi)), turn_radius_m=R, load_factor=n)


def _tracking(cmd, ach, kind, dt):
    """Full metric bundle for one channel. `kind` in {bank, climb, airspeed}; gain/phase only for
    bank (the shape channel)."""
    err_rms = _rms(ach - cmd)
    out = dict(rmse=err_rms, nrmse=_nrmse(cmd, err_rms), theil_u=_theil_u(cmd, ach))
    if kind == "bank":
        out["gain"], out["phase_lag_s"] = _gain_phase(cmd, ach, dt)
    return out


def make_env(plane, steps, turbulence=None):
    cfg = attitude_env_cfg(plane, horizon=steps + 2)
    if turbulence:
        cfg["turbulence"] = turbulence
    return AttitudeEnv(1, "cuda", cfg)


@torch.no_grad()
def fly(plane, maneuver, controller_factory, turbulence=None, turb_W20=None):
    """controller_factory is called as f(env, phi_s, hdot_s, va_s) AFTER the rollout env exists:
    the classical executors (falcons/controllers/{lqr,mppi}/attitude.py) read the plant state of
    that env directly, so they cannot be built before it.

    Survival = completed the whole maneuver inside the flight envelope (Va in [0.55,1.8]*base_vel,
    finite, position bounded, no env stall/bank termination). Tracking metrics are computed over
    t >= settle, and only for a surviving run.

    turbulence: enable a Dryden gust field ("light"/"moderate"/"severe" seeds the base W20);
    turb_W20 then overrides the wind speed at 20 ft (m/s) so the caller can hit a target gust
    intensity. Metrics are always computed from the TRUE state, never the (possibly disturbed) obs.
    """
    probe = make_env(plane, 10)
    phi_s, hdot_s, va_s = maneuver_stream(maneuver, probe.phi_max_full,
                                          probe.hdot_max_full, probe.base_vel)
    steps = len(phi_s)
    e = make_env(plane, steps, turbulence=turbulence)
    e.reset()
    if turb_W20 is not None:
        e.set_turb_W20(turb_W20)
    controller = controller_factory(e, phi_s, hdot_s, va_s)
    base = e.base_vel
    survived = True
    # per-step arrays collected every run (cheap, ~steps*12 floats); bank in deg, climb/Va in SI.
    # `pos` is the NED position (north, east, down) and `yaw` the heading in radians: the flight
    # path itself, which the figure layer draws rather than reconstructing it from the bank.
    tr = {"t": [], "roll": [], "phi_cmd": [], "climb": [], "hdot_cmd": [],
          "Va": [], "va_cmd": [], "pos": [], "yaw": []}
    for t in range(steps):
        wp.copy(e._phi_tgt, wp.array(phi_s[t:t + 1], device="cuda"))
        wp.copy(e._hdot_tgt, wp.array(hdot_s[t:t + 1], device="cuda"))
        wp.copy(e._va_tgt, wp.array(va_s[t:t + 1], device="cuda"))
        e._obs_launch()
        obs = wp.to_torch(e._obs)
        act = controller(obs).clamp(-1.0, 1.0)

        s = e.model._state
        q = s["orientation"].numpy()[0]
        vb = s["linear_vel"].numpy()[0]
        p = s["position"].numpy()[0]
        qx, qy, qz, qw = q
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
        tt = 2.0 * np.cross(q[:3], vb); vw = vb + qw * tt + np.cross(q[:3], tt)
        climb = -vw[2]
        Va = float(np.linalg.norm(vb))

        if (not np.isfinite(p).all() or np.abs(p).max() > 2000.0
                or Va < 0.55 * base or Va > 1.8 * base):
            survived = False
            break
        tr["t"].append(t * DT); tr["roll"].append(np.degrees(roll))
        tr["phi_cmd"].append(np.degrees(phi_s[t])); tr["climb"].append(climb)
        tr["hdot_cmd"].append(hdot_s[t]); tr["Va"].append(Va); tr["va_cmd"].append(va_s[t])
        tr["pos"].append(p.copy()); tr["yaw"].append(yaw)
        e.step(act)   # advance the real dynamics

    tr = {k: np.array(v, dtype=float) for k, v in tr.items()}
    trace = tr
    desc = _descriptors(phi_s, base)
    n_scored = max(0, len(tr["t"]) - SETTLE)
    if not survived or n_scored == 0:
        return dict(survival=0.0, phi_rmse=None, hdot_rmse=None, va_rmse=None,
                    phi=None, climb=None, airspeed=None, descriptors=desc,
                    steps=steps, t_fail=t * DT, trace=trace)
    sl = slice(SETTLE, None)
    phi = _tracking(tr["phi_cmd"][sl], tr["roll"][sl], "bank", DT)
    climb = _tracking(tr["hdot_cmd"][sl], tr["climb"][sl], "climb", DT)
    airspeed = _tracking(tr["va_cmd"][sl], tr["Va"][sl], "airspeed", DT)
    return dict(survival=1.0,
                # back-compat scalar keys
                phi_rmse=phi["rmse"], hdot_rmse=climb["rmse"], va_rmse=airspeed["rmse"],
                # full per-channel bundles (rmse / nrmse / theil_u [+ gain,phase_lag_s for bank])
                phi=phi, climb=climb, airspeed=airspeed, descriptors=desc,
                steps=steps, t_fail=None, trace=trace)


# ───── cells, caching, table ─────
SHAPES = ["circle", "figure8", "helix", "sturn"]
METHODS = ["LQR", "MPPI", "PPO", "SAC", "TD3"]  # table order
ALGO = {"LQR": "lqr", "MPPI": "mppi", "PPO": "ppo", "SAC": "sac", "TD3": "td3"}
# Methods flown several times per entry, ON TOP of the seed checkpoints controllers() returns.
# The learned methods are flown once per training seed and need none of these. MPPI has no seeds
# but its rollout-cost reduction is not deterministic (see falcons/controllers/mppi/attitude.py),
# so a single rollout is one draw: on the Volantex Ranger the measured bank RMSE of one cell ranged
# over 3.0-34.0 deg across repetitions. Every MPPI cell is therefore flown `mppi_draws` times, at
# sampling seeds mppi_seed + 0 .. mppi_draws - 1, so the set of draws is itself reproducible. Either
# way a cell is reported as the mean over the runs that survived, with their spread in phi_rmse_std.
DRAWS = ("MPPI",)


def _suffix(seed):
    """Cache-file suffix for one run of a cell: the first seed is unsuffixed, later seeds are
    "_s1"/"_s2". The `_s0` naming rule applies to CHECKPOINTS only, not to these caches."""
    return "" if seed == 0 else f"_s{seed}"


def controllers(plane, ckpt_dir=CKPT_DIR):
    """Method -> [(seed suffix, factory)], in table order, where a factory is
    f(env, phi_s, hdot_s, va_s, draw=0) -> callable(obs) -> action.

    All five methods drive the IDENTICAL (phi*, hdot*, Va*) stream through the same plant: the
    learned executors from their attitude checkpoints, LQR/MPPI as classical attitude executors
    built on the rollout env (falcons/controllers/{lqr,mppi}/attitude.py). Two asymmetries to keep
    in mind when reading the table: LQR has no climb-rate channel (hdot* enters as an integrated
    altitude reference) and MPPI plans over a 1 s preview of the stream, while the learned
    executors see only the current target. NB: PPO trains WITH CAPS smoothness (baked into
    falcons/train/ppo.py), so "PPO" here is the CAPS-regularized on-policy method; "SAC" is the
    no-smoothness baseline.

    The learned methods get one entry per seed checkpoint that exists, so a cell is scored over the
    same seed set the altitude protocol uses rather than over one lucky or unlucky training run.
    The classical executors have no seeds and get a single unsuffixed entry; `draw` picks MPPI's
    sampling realization and is ignored by everything else. A method with no checkpoint at all is
    skipped with a printed note rather than failing the table."""
    from falcons.controllers.policies import load_policy, seeds
    out = {}
    for m in METHODS:
        algo = ALGO[m]
        def make(algo=algo, seed=0):
            def factory(env, phi_s, hdot_s, va_s, draw=0):
                # MPPI has no checkpoint to select: its `seed` argument IS the sampling draw
                s = draw if algo == "mppi" else seed
                return load_policy(algo, "attitude", plane, s, env=env,
                                   stream=(phi_s, hdot_s, va_s), ckpt_dir=ckpt_dir).act
            return factory
        ss = seeds(algo, "attitude", plane, ckpt_dir)
        if not ss:
            print(f"  skip {m} on {plane}: no attitude checkpoint")
            continue
        out[m] = [(_suffix(s), make(seed=s)) for s in ss]
    return out


def _mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def _std(vals):
    """Population std over the repetitions that produced a value, None when none did."""
    v = [x for x in vals if x is not None]
    return float(np.std(v)) if v else None


def _std_pct(vals):
    """... for a column the table prints as a percentage."""
    s = _std(vals)
    return None if s is None else s * 100.0


def aggregate(reps, tol_deg):
    """Collapse repetitions of one cell into a single run-shaped dict.

    Survival is the mean over ALL repetitions; every error metric is a mean over the repetitions
    that survived, since a departed run has no tracking metric to average. The trace kept for the
    figure is the median-RMSE surviving repetition, so the plotted rollout is a typical one rather
    than the best of the set."""
    ok = [r for r in reps if r["survival"] == 1.0]
    out = dict(reps[0])
    out["survival"] = float(np.mean([r["survival"] for r in reps]))
    out["n_reps"] = len(reps)
    if not ok:
        return out
    order = sorted(ok, key=lambda r: r["phi"]["rmse"])
    out.update(order[len(order) // 2])                      # representative trace + descriptors
    out["survival"] = float(np.mean([r["survival"] for r in reps]))
    out["n_reps"] = len(reps)
    for ch in ("phi", "climb", "airspeed"):
        out[ch] = {k: _mean([r[ch][k] for r in ok]) for k in order[0][ch]}
    tibs = [time_in_band(r["trace"], tol_deg) for r in ok]
    out["tib"] = _mean(tibs)
    # survival is a draw too: the same MPPI cell departed 0 of 5 draws in one sweep and 1 of 5 in
    # the next. Spread over ALL repetitions, the way the mean is taken.
    out["survival_std"] = float(np.std([r["survival"] for r in reps]))
    rm = [r["phi"]["rmse"] for r in ok]
    out["phi_rmse_std"] = float(np.std(rm))
    out["phi_rmse_min"], out["phi_rmse_max"] = float(np.min(rm)), float(np.max(rm))
    # Every averaged metric of a cell is a draw when the method samples, not only the bank RMSE,
    # so each averaged column carries its own spread across the surviving repetitions -- in the
    # units the table prints it. TOLERANCE["maneuvers.csv"] gates the MPPI rows on these.
    out["stds"] = {"phi_nrmse_pct": _std_pct([r["phi"]["nrmse"] for r in ok]),
                   "phi_theil_u": _std([r["phi"]["theil_u"] for r in ok]),
                   "phi_gain": _std([r["phi"]["gain"] for r in ok]),
                   "phi_phase_lag_s": _std([r["phi"]["phase_lag_s"] for r in ok]),
                   "time_in_band_pct": _std_pct(tibs),
                   "hdot_rmse_ms": _std([r["climb"]["rmse"] for r in ok]),
                   "hdot_nrmse_pct": _std_pct([r["climb"]["nrmse"] for r in ok]),
                   "va_rmse_ms": _std([r["airspeed"]["rmse"] for r in ok])}
    return out


def cell(plane, maneuver, method, ckpt_dir=CKPT_DIR, cache_dir=RESULTS_DIR / "traces",
         refresh=False, mppi_draws=5, mppi_seed=0, tol_deg=None):
    """fly(...) over every (seed suffix, factory) entry of one cell, with an on-disk cache per
    entry, so the table can be rebuilt without re-flying, and so adding a seed re-flies only that
    seed rather than the whole cell.

    Methods listed in DRAWS are additionally flown `mppi_draws` times per entry, at sampling seeds
    mppi_seed + k, so the set of draws is reproducible even though a single draw is not."""
    n = mppi_draws if method in DRAWS else 1
    reps = []
    for suf, factory in controllers(plane, ckpt_dir)[method]:
        name = f"{plane}_{maneuver}_{ALGO[method]}{suf}{'' if n == 1 else f'_r{n}'}.pkl"
        path = Path(cache_dir) / name
        if not refresh and path.exists():
            with open(path, "rb") as f:
                got = pickle.load(f)
            if not isinstance(got, list):  # caches written before repetitions hold a bare run dict
                got = [got]
        else:
            got = [fly(plane, maneuver, partial(factory, draw=mppi_seed + k)) for k in range(n)]
            os.makedirs(cache_dir, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(got, f)
        reps += got
    return reps[0] if len(reps) == 1 else aggregate(reps, tol_deg)


def time_in_band(trace, tol_deg, settle=50):
    """Fraction of scored time the achieved bank stays within +/-tol of commanded."""
    r = trace["roll"][settle:]; c = trace["phi_cmd"][settle:]
    return float(np.mean(np.abs(r - c) <= tol_deg)) if len(r) else 0.0


# how each averaged column prints, and therefore how its spread prints
STD_FMT = {"phi_nrmse_pct": ".1f", "phi_theil_u": ".3f", "phi_gain": ".3f",
           "phi_phase_lag_s": ".3f", "time_in_band_pct": ".1f", "hdot_rmse_ms": ".3f",
           "hdot_nrmse_pct": ".1f", "va_rmse_ms": ".2f"}


def _std_cells(r):
    """The `<col>_std` cells of one row: each averaged column's spread across the repetitions of
    the cell, formatted like its mean. Empty for a cell flown once -- there is nothing to spread
    over -- exactly as phi_rmse_std already is."""
    st = r.get("stds", {})
    return {f"{c}_std": ("" if st.get(c) is None else format(st[c], f))
            for c, f in STD_FMT.items()}


def run(planes, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR, refresh=False,
        mppi_draws=5, mppi_seed=0):
    """One row per (aircraft, maneuver, method) in maneuvers.csv, plus mppi_variance.json with the
    spread across the runs of every cell that was flown more than once. Rollouts are cached under
    <results_dir>/traces, so a table can be rebuilt from a shipped cache without flying."""
    wp.init()
    os.makedirs(results_dir, exist_ok=True)
    cache_dir = Path(results_dir) / "traces"
    table, variance = [], {}

    for ac in planes:
        tol_deg = np.degrees(attitude_env_cfg(ac).get("sig_phi", 0.15))   # tracking tolerance band
        methods = list(controllers(ac, ckpt_dir))
        for sh in SHAPES:
            runs = {m: cell(ac, sh, m, ckpt_dir=ckpt_dir, cache_dir=cache_dir, refresh=refresh,
                            mppi_draws=mppi_draws, mppi_seed=mppi_seed, tol_deg=tol_deg)
                    for m in methods}
            # spread across the runs of the cell for the methods that are flown more than once, so
            # a reader can tell how much of an MPPI cell is the method and how much is the draw
            for m in methods:
                if "n_reps" in runs[m] and "phi_rmse_std" in runs[m]:
                    variance[f"{ac}/{sh}/{m}"] = dict(
                        n=runs[m]["n_reps"], survival=runs[m]["survival"],
                        phi_mean=runs[m]["phi"]["rmse"], phi_std=runs[m]["phi_rmse_std"],
                        phi_min=runs[m]["phi_rmse_min"], phi_max=runs[m]["phi_rmse_max"])
            # per-maneuver difficulty descriptors, identical for every method
            d = next((runs[m]["descriptors"] for m in methods), None)
            for m in methods:
                r = runs[m]
                if r["survival"] == 0:
                    table.append(dict(aircraft=ac, maneuver=sh, method=m,
                        n_runs=r.get("n_reps", 1), survival="0", survival_std="",
                        phi_rmse_deg="crash", phi_rmse_std="", phi_nrmse_pct="", phi_theil_u="",
                        phi_gain="",
                        phi_phase_lag_s="", time_in_band_pct="", hdot_rmse_ms="", hdot_nrmse_pct="",
                        va_rmse_ms="", phi_max_deg=f"{d['phi_max_deg']:.1f}",
                        turn_radius_m=f"{d['turn_radius_m']:.0f}", load_factor=f"{d['load_factor']:.3f}",
                        **{f"{c}_std": "" for c in STD_FMT}))
                    continue
                p = r["phi"]
                tib = r["tib"] if "tib" in r else time_in_band(r["trace"], tol_deg)
                hdn = r["climb"]["nrmse"]
                # the real fraction, not a flag: with repetitions a method can survive some
                # draws and depart others (Volantex Ranger s-turn under MPPI does exactly this)
                table.append(dict(aircraft=ac, maneuver=sh, method=m,
                    n_runs=r.get("n_reps", 1),
                    survival=f"{r['survival']:.2f}",
                    survival_std=(f"{r['survival_std']:.2f}" if "survival_std" in r else ""),
                    phi_rmse_deg=f"{p['rmse']:.2f}",
                    # spread across the runs of the cell: training seeds for the learned methods,
                    # sampling draws for MPPI, and empty for a method flown once
                    phi_rmse_std=(f"{r['phi_rmse_std']:.2f}" if "phi_rmse_std" in r else ""),
                    phi_nrmse_pct=(f"{p['nrmse']*100:.1f}" if p["nrmse"] is not None else ""),
                    phi_theil_u=f"{p['theil_u']:.3f}",
                    phi_gain=(f"{p['gain']:.3f}" if p['gain'] is not None else ""),
                    phi_phase_lag_s=(f"{p['phase_lag_s']:.3f}" if p['phase_lag_s'] is not None else ""),
                    time_in_band_pct=f"{tib*100:.1f}",
                    hdot_rmse_ms=f"{r['climb']['rmse']:.3f}",
                    hdot_nrmse_pct=(f"{hdn*100:.1f}" if hdn is not None else ""),
                    va_rmse_ms=f"{r['airspeed']['rmse']:.2f}",
                    phi_max_deg=f"{d['phi_max_deg']:.1f}", turn_radius_m=f"{d['turn_radius_m']:.0f}",
                    load_factor=f"{d['load_factor']:.3f}", **_std_cells(r)))

    cols = ["aircraft", "maneuver", "method", "n_runs", "survival", "survival_std",
            "phi_rmse_deg", "phi_rmse_std", "phi_nrmse_pct", "phi_nrmse_pct_std",
            "phi_theil_u", "phi_theil_u_std", "phi_gain", "phi_gain_std",
            "phi_phase_lag_s", "phi_phase_lag_s_std", "time_in_band_pct", "time_in_band_pct_std",
            "hdot_rmse_ms", "hdot_rmse_ms_std", "hdot_nrmse_pct", "hdot_nrmse_pct_std",
            "va_rmse_ms", "va_rmse_ms_std", "phi_max_deg", "turn_radius_m", "load_factor"]
    out = Path(results_dir) / "maneuvers.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in table:
            w.writerow(r)
    print(f"wrote {out}")
    if variance:
        with open(Path(results_dir) / "mppi_variance.json", "w") as f:
            json.dump(variance, f, indent=1)
        print("wrote mppi_variance.json")
    return out
