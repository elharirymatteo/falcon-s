"""Altitude acquisition-and-hold protocol: the five controllers (LQR, MPPI, PPO, SAC, TD3) on a
known, randomized initial condition, over a range of commanded altitudes, scored with the full
metric set (Survival, settled RMSE, settling time, overshoot, energy).

Every episode starts a known distance OFFSET m off the target, in both directions, at five
commanded altitudes: without an acquisition transient, settling time and overshoot are undefined
and the reported RMSE is a steady-state ripple rather than a control-quality measure.

Run: run(planes) -> <results_dir>/altitude.csv; one classical episode is a child process,
     python -m falcons.benchmark.altitude --child PLANE ALGO TARGET H0.
"""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from falcons.aircraft.config import AircraftConfig
from falcons.envs.configs import TRIM_VA
from falcons.paths import CKPT_DIR, RESULTS_DIR

TARGETS = [20.0, 40.0, 60.0, 80.0, 100.0]   # commanded altitudes [m]
OFF_LO, OFF_HI = 12.0, 18.0                 # spawn offset magnitude ~ U[lo,hi] m, both directions
N_RL = 20                                   # episodes per (target, direction) for the learned policies
N_CL = 5                                    # ... for the classical controllers (deterministic law,
                                            #     spread from the IC draw; MPPI also samples)
STEPS = 6000                                # 60 s at dt=0.01 — covers an 18 m ramp plus settling
DT = 0.01
SETTLE_FRAC = 0.4                           # settled window = final 40% of the episode
BAND = 1.0                                  # settling-time band [m], as in the original metric set
RL_ALGOS = ["ppo", "sac", "td3"]
CL_ALGOS = ["lqr", "mppi"]
# Independent repetitions per classical method. The LQR is a deterministic law and one sweep is
# the whole answer. MPPI is not: besides its sampling noise (pinned by MPPIAltitude's seed call),
# the cost reduction in falcons/controllers/mppi/kernels.py accumulates rollout costs with atomics,
# whose thread ordering is not deterministic, so the same seed still yields a different plan. A
# single MPPI sweep is one draw from that distribution, and on the Volantex Ranger the settled
# error of one cell ranges over 2.6--15.8 m across repetitions. Reported as mean +- std over reps,
# exactly as the learned methods are reported over training seeds (`mppi_seeds` below).


# ─── metrics (shared by every method, computed from altitude + throttle traces) ──────────────
def episode_metrics(h, target, h0, thr):
    """h: (T,) altitude with NaN after termination; thr: (T,) physical throttle in [0,1].

    settling_time: first time |h - target| <= BAND and stays inside to the end (inf if never).
    overshoot:     largest excursion PAST the target, in the direction of travel — NOT max|e|,
                   which under a 15 m acquisition step just returns the step size.
    energy:        mean squared physical throttle, the discrete form of (1/T) int ||u_motor||^2 dt.
    """
    n = int((~np.isnan(h)).sum())
    if n < len(h):
        return None                                  # terminated early = did not survive
    hh = h[:n]
    e = hh - target
    t = np.arange(n) * DT
    s0 = int((1.0 - SETTLE_FRAC) * n)

    inside = np.abs(e) <= BAND
    st = np.inf
    if inside[-1]:
        # walk back from the end over the final contiguous in-band run
        i = n - 1
        while i > 0 and inside[i - 1]:
            i -= 1
        st = float(t[i])

    sgn = np.sign(target - h0)                       # +1 climbing to the target, -1 descending
    over = float(max(0.0, np.max(sgn * e))) if n else 0.0

    return dict(rmse=float(np.sqrt(np.mean(e[s0:] ** 2))),
                settling=st, overshoot=over,
                energy=float(np.mean(thr[:n] ** 2)) if thr is not None else np.nan)


def _nanmean(vals):
    """np.nanmean of an all-NaN slice is `nan` with a RuntimeWarning; a method that never settled
    on any seed is a legitimate result here, so say `nan` outright instead of warning about it."""
    return float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")


def _nanstd(vals):
    """Companion to `_nanmean`, same reason."""
    return float(np.nanstd(vals)) if not np.all(np.isnan(vals)) else float("nan")


def aggregate(rows):
    """Mean over surviving episodes; survival over all. Settling time is reported as a mean over
    the episodes that actually settled, alongside the fraction that did."""
    surv = [r for r in rows if r is not None]
    if not surv:
        return dict(survival=0.0, rmse=np.nan, settling=np.nan, settled_frac=0.0,
                    overshoot=np.nan, energy=np.nan, n=len(rows))
    fin = [r["settling"] for r in surv if np.isfinite(r["settling"])]
    return dict(survival=len(surv) / len(rows),
                rmse=float(np.mean([r["rmse"] for r in surv])),
                settling=float(np.mean(fin)) if fin else np.nan,
                settled_frac=len(fin) / len(surv),
                overshoot=float(np.mean([r["overshoot"] for r in surv])),
                energy=float(np.nanmean([r["energy"] for r in surv])),
                n=len(rows))


# ─── learned policies: one batched warp env covers every (target, direction, episode) ────────
def run_learned(plane, algo, seed=0, ckpt_dir=CKPT_DIR):
    import torch
    import warp as wp
    from falcons.envs.configs import altitude_env_cfg
    from falcons.envs.altitude import AltitudeEnv
    from falcons.controllers.policies import load_policy

    rng = np.random.default_rng(0)
    tgt, alt0 = [], []
    for T in TARGETS:
        for sgn in (-1.0, +1.0):
            off = rng.uniform(OFF_LO, OFF_HI, N_RL)
            tgt.append(np.full(N_RL, T)); alt0.append(T + sgn * off)
    tgt = np.concatenate(tgt).astype(np.float32)
    alt0 = np.concatenate(alt0).astype(np.float32)
    n = tgt.shape[0]

    env = AltitudeEnv(n, "cuda", altitude_env_cfg(plane, 0.0, horizon=STEPS + 1))
    env.reset()
    pos = np.zeros((n, 3), dtype=np.float32); pos[:, 2] = -alt0
    vel = np.zeros((n, 3), dtype=np.float32); vel[:, 0] = env.base_vel
    wp.copy(env._true_target, wp.array(tgt, device="cuda"))
    wp.copy(env._target, wp.array(alt0, device="cuda"))          # sub-target starts at the spawn
    wp.copy(env.model._state["position"], wp.array(pos, dtype=wp.vec3f, device="cuda"))
    wp.copy(env.model._state["linear_vel"], wp.array(vel, dtype=wp.vec3f, device="cuda"))
    env._ierr.zero_(); env._refrate.zero_(); env._obs_launch()

    policy = load_policy(algo, "altitude", plane, seed, ckpt_dir=ckpt_dir).act
    obs = wp.to_torch(env._obs)
    hist = np.full((STEPS, n), np.nan)
    thr = np.full((STEPS, n), np.nan)
    live = np.ones(n, dtype=bool)
    with torch.no_grad():
        for t in range(STEPS):
            obs, _, d, _ = env.step(policy(obs).clamp(-1.0, 1.0))
            h = -env.model._state["position"].numpy()[:, 2]
            tl = env.model._actuator_states["throttle_left"].numpy()
            hist[t, live] = h[live]; thr[t, live] = tl[live]
            live &= ~d.cpu().numpy().astype(bool)
            if not live.any():
                break
    return [episode_metrics(hist[:, i], float(tgt[i]), float(alt0[i]), thr[:, i])
            for i in range(n)]


# ─── classical controllers: one subprocess per commanded altitude (warp bakes the reference) ──
def run_classical(plane, algo, mppi_seeds=(0, 1, 2), ckpt_dir=CKPT_DIR):
    """One sweep of episodes per sampler seed: the LQR is a deterministic law and gets a single
    sweep, MPPI one per entry of `mppi_seeds` (handed to the child as MPPI_DRAW).

    Fairness: all five methods track the IDENTICAL rate-limited acquisition reference -- the
    sub-target ramps from the spawn altitude to the commanded target at the airframe's `ref_rate`
    (falcons/envs/configs.py), then holds. That is the reference the RL policies are trained
    against; commanding the classical controllers a raw altitude STEP instead is a strictly harder
    task, and on the V7 drives the LQR past 1.5x stall alpha within half a second. The reference is
    implemented once as `acquire_ramp` in falcons/controllers/lqr/reference.py with a warp twin in
    falcons/controllers/mppi/reference.py, both reading TRAJ_CONST_ALT / TRAJ_ACQ_H0 /
    TRAJ_ACQ_RATE.

    Backends: PPO/SAC/TD3 and MPPI run on the warp plant; LQR runs on the CPU/SciPy plant through
    its own adapter. The two are validated to single-step parity, but they are not the same code
    path -- state that in the caption rather than claiming bit-identical conditions.

    One commanded altitude per SUBPROCESS: the MPPI reference is a warp module that bakes the
    commanded altitude at build time, and rebinding it in-process does not trigger a rebuild.

    Each entry of `mppi_seeds` is a different sampler realization, because a single sweep is one
    draw from the planner's distribution. Sweeping the same seed three times instead measures only
    the non-deterministic atomic-add reduction and not the sampler; `mppi_seeds=(0, 0, 0)`
    reproduces that mode."""
    gains = AircraftConfig(plane).acquisition_gains_path
    sweeps = []
    for draw in (mppi_seeds if algo == "mppi" else (0,)):
        rows = []
        for T in TARGETS:
            # TRAJ_ACQ_H0 is baked per episode too, so one child per (target, direction, episode)
            rng = np.random.default_rng(0)
            for sgn in (-1.0, +1.0):
                for k in range(N_CL):
                    h0 = float(T + sgn * rng.uniform(OFF_LO, OFF_HI))
                    env = dict(os.environ, TRAJ_CONST_ALT=f"{T}", TRAJ_ACQ_H0=f"{h0}",
                               TRAJ_REF_VA=f"{TRIM_VA[plane]}")
                    if algo == "mppi":
                        env["MPPI_DRAW"] = f"{draw}"
                    if algo == "lqr" and gains is not None:
                        env["LQR_GAINS_PATH"] = str(gains)
                    p = subprocess.run([sys.executable, "-m", "falcons.benchmark.altitude",
                                        "--child", plane, algo, f"{T}", f"{h0}"],
                                       capture_output=True, text=True, env=env)
                    line = [l for l in p.stdout.splitlines() if l.startswith("JSON")]
                    if not line:
                        # the child prints its JSON line with m=None when the AIRCRAFT died, so a
                        # missing line is the harness failing, not a flight failure -- scoring it
                        # as one would hide a broken child inside the survival tolerance
                        raise RuntimeError(f"child crashed: {plane} {algo} T={T} h0={h0:.1f}\n"
                                           f"{p.stderr[-2000:]}")
                    rows.append(json.loads(line[-1][4:])["m"])
        sweeps.append(rows)
    return sweeps


def child(plane, algo, target, h0):
    """Single classical episode; the reference constants are already baked from the environment."""
    from falcons.controllers.lqr.altitude import LQRAltitude
    from falcons.controllers.mppi.altitude import MPPIAltitude
    scen = {"name": "acquire", "trajectory_type": "acquire_ramp", "use_sensor_noise": False,
            "use_estimator": False, "wind": None}
    if algo == "lqr":
        d = LQRAltitude(plane, init_altitude=h0).run(scen)
    else:
        d = MPPIAltitude(plane, init_altitude=h0,
                         mppi_seed=int(os.environ.get("MPPI_DRAW", 0))).run(scen)
    h = -d["positions"][:, 2]
    m = min(len(h), STEPS)
    full = np.full(STEPS, np.nan); full[:m] = h[:m]
    thr = np.full(STEPS, np.nan); thr[:m] = (d["actions"][:m, 0] + 1.0) * 0.5
    met = episode_metrics(full, target, h0, thr) if m >= STEPS else None
    print("JSON" + json.dumps(dict(target=target, h0=h0, m=met)), flush=True)


def run(planes, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR, algos=tuple(RL_ALGOS + CL_ALGOS),
        mppi_seeds=(0, 1, 2)):
    """Learned policies are scored over EVERY seed checkpoint and reported as mean +- std across
    seeds; scoring one checkpoint conflates a method with a lucky or unlucky training run, and on
    this task the spread between seeds is larger than the spread between methods. The LQR has no
    seeds and is a single deterministic sweep; MPPI has no seeds either but is not deterministic
    (see above), so it is swept once per `mppi_seeds` entry and reported the same way."""
    from falcons.controllers.policies import seeds
    os.makedirs(results_dir, exist_ok=True)
    table = []
    for ac in planes:
        for algo in algos:
            try:
                if algo in RL_ALGOS:
                    ss = seeds(algo, "altitude", ac, ckpt_dir)
                    if not ss:
                        raise FileNotFoundError(
                            f"no checkpoints for {algo} altitude {ac} under {ckpt_dir}")
                    per = [aggregate(run_learned(ac, algo, s, ckpt_dir)) for s in ss]
                else:
                    per = [aggregate(rows) for rows in
                           run_classical(ac, algo, mppi_seeds, ckpt_dir)]
            except FileNotFoundError as e:
                print(f"  skip {algo} {ac}: {e}")
                continue
            a = {k: _nanmean([p[k] for p in per]) for k in
                 ("survival", "rmse", "settling", "settled_frac", "overshoot", "energy")}
            a.update({f"{k}_std": _nanstd([p[k] for p in per]) for k in
                      ("survival", "rmse", "settling", "settled_frac", "overshoot", "energy")})
            a.update(aircraft=ac, method=algo.upper(), n=per[0]["n"], n_seeds=len(per),
                     acquired=_nanmean([p["survival"] * p["settled_frac"] for p in per]),
                     acquired_std=_nanstd([p["survival"] * p["settled_frac"] for p in per]))
            table.append(a)
            print(f"{ac:16s} {algo.upper():5s} n_seeds {a['n_seeds']}  "
                  f"surv {a['survival']:.3f}+-{a['survival_std']:.3f}  "
                  f"acq {a['acquired']:.3f}+-{a['acquired_std']:.3f}  "
                  f"rmse {a['rmse']:.3f}+-{a['rmse_std']:.3f}  "
                  f"t_s {a['settling']:.1f}  over {a['overshoot']:.2f}+-{a['overshoot_std']:.2f}",
                  flush=True)
    cols = ["aircraft", "method", "n", "n_seeds", "survival", "survival_std",
            "acquired", "acquired_std", "rmse", "rmse_std", "settling", "settling_std",
            "settled_frac", "settled_frac_std", "overshoot", "overshoot_std",
            "energy", "energy_std"]
    out = Path(results_dir) / "altitude.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in table:
            w.writerow(r)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":          # child worker only; the full sweep is reached through run()
    i = sys.argv.index("--child")
    child(sys.argv[i + 1], sys.argv[i + 2], float(sys.argv[i + 3]), float(sys.argv[i + 4]))
