"""Two measurements of the simulator itself, neither of which is about a controller.

`parity` flies one trained policy on both plants -- the warp kernels (`falcons.envs.altitude`) and
the pure-torch twin (`falcons.sim.torch.altitude`) -- from the same initial conditions over the
same altitude targets, and reports each one's settled tracking error per target. `throughput`
measures env-steps per second on three backends of the same Airship dynamics: warp on the GPU,
torch on the GPU, and the single-aircraft Python/numpy plant (`falcons.sim.cpu`) at one env.

Run: parity(plane) -> dict, throughput(plane) -> <results_dir>/throughput.csv,
     run(plane) -> both, plus <results_dir>/parity.json
"""
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist

from falcons.paths import CKPT_DIR, RESULTS_DIR

# ─── parity ─────────────────────────────────────────────────────────────────────────────────
TARGETS = (20.0, 40.0, 60.0, 80.0, 100.0)   # commanded altitudes [m]
HORIZON = 1500                              # rollout length [steps]; dt = 0.01 s -> 15 s
SETTLE = 1000                               # settled window starts here: the last third of HORIZON
SPAWN_BELOW = 10.0                          # spawn this far under the target [m] ...
SPAWN_MIN = 2.0                             # ... but never below this altitude

# ─── throughput ─────────────────────────────────────────────────────────────────────────────
R_TRIALS = 10
GPU_SIZES = (64, 256, 1024, 4096, 16384)
GPU_STEPS = 1000
GPU_WARMUP = 200
CPU_STEPS = 200
CPU_WARMUP = 20
CPU_THROTTLE = 0.5                          # the constant motor command the CPU plant is stepped at


def stats(vals):
    """mean, SAMPLE std (ddof=1) and the half-width of the two-sided 95 % Student-t interval.

    The critical value comes from the t distribution at df = n-1 rather than being baked in
    at df = 9, so the interval stays correct if the trial count moves off R_TRIALS."""
    v = np.array(vals)
    t95 = t_dist.ppf(0.975, len(v) - 1)
    return v.mean(), v.std(ddof=1), t95 * v.std(ddof=1) / np.sqrt(len(v))


# ─── parity ─────────────────────────────────────────────────────────────────────────────────
def _roll_warp(cfg, policy, tgt, alt0, horizon):
    """Fly `policy` on the warp plant; return the altitude history, (horizon, n) in metres."""
    import torch
    import warp as wp
    from falcons.envs.altitude import AltitudeEnv

    with torch.no_grad():
        n = tgt.shape[0]
        env = AltitudeEnv(n, "cuda", cfg)
        env.reset()
        pos = np.zeros((n, 3), np.float32); pos[:, 2] = -alt0
        vel = np.zeros((n, 3), np.float32); vel[:, 0] = env.base_vel
        wp.copy(env._true_target, wp.array(tgt, device="cuda"))
        wp.copy(env._target, wp.array(alt0, device="cuda"))       # sub-target starts at the spawn
        wp.copy(env.model._state["position"], wp.array(pos, dtype=wp.vec3f, device="cuda"))
        wp.copy(env.model._state["linear_vel"], wp.array(vel, dtype=wp.vec3f, device="cuda"))
        env._ierr.zero_(); env._refrate.zero_(); env._obs_launch()
        obs = wp.to_torch(env._obs)
        hist = np.zeros((horizon, n), np.float32)
        for t in range(horizon):
            a = policy(obs).clamp(-1, 1)
            hist[t] = -env.model._state["position"].numpy()[:, 2]
            obs, r, d, i = env.step(a)
    return hist


def _roll_torch(cfg, policy, tgt, alt0, horizon):
    """The same flight on the pure-torch twin. Same initial state, set field by field because the
    twin holds its state as plain tensors rather than in a warp state dict."""
    import torch
    from falcons.sim.torch.altitude import AltitudeEnv, TRIM_THROTTLE

    with torch.no_grad():
        n = tgt.shape[0]
        env = AltitudeEnv(n, "cuda", cfg)
        env.reset()
        env.pos[:] = torch.tensor(np.stack([np.zeros(n), np.zeros(n), -alt0], 1),
                                  dtype=torch.float32, device="cuda")
        env.vel[:] = 0; env.vel[:, 0] = env.base_vel; env.omega[:] = 0
        env.quat[:] = torch.tensor([0, 0, 0, 1.], device="cuda")
        env.elev[:] = 0; env.ail[:] = 0; env.rud[:] = 0
        env.thr[:] = TRIM_THROTTLE; env.elev_d[:] = 0
        env.true_tgt[:] = torch.tensor(tgt, device="cuda")
        env.tgt[:] = torch.tensor(alt0, device="cuda"); env.ierr[:] = 0
        obs = env._obs(*env._post_reset_va_alpha())
        hist = np.zeros((horizon, n), np.float32)
        for t in range(horizon):
            a = policy(obs).clamp(-1, 1)
            hist[t] = -env.pos[:, 2].cpu().numpy()
            obs, r, d, i = env.step(a)
    return hist


def rmse_per_target(tgt, hist, settle=SETTLE):
    out = []
    for tv in TARGETS:
        m = tgt == tv
        err = hist[settle:, m] - tv
        out.append(np.sqrt(np.mean(err ** 2)))
    return np.array(out)


def parity(plane, ckpt_dir=CKPT_DIR, n_per_target=64, horizon=1500):
    """One PPO checkpoint, both plants, `TARGETS` commanded on each. Returns
    {"targets": [...], "warp": {target: settled RMSE [m]}, "torch": {...}}.

    The two plants are independent implementations of one set of equations, so agreement is
    evidence that the learned result is a property of the DYNAMICS and not of the kernel that
    integrates them. `falcons.benchmark.diff` and `tests/test_parity.py` pin the single step; this
    pins the 15 s closed-loop rollout, which is where an integrator difference would accumulate.
    One checkpoint is flown on both plants, so what is compared is two plants and not two training
    runs.

    Caveat, for the airframes it applies to: the torch twin hardcodes the integral clamp at 25.0
    while the warp env reads `i_clamp` from the config. They agree on every airframe except the
    A0S, whose config sets 12.0; a parity run on that airframe is comparing two slightly different
    controllers.

    The settled window is the last third of the rollout, `SETTLE/HORIZON` -- expressed as a
    fraction so that a shorter `horizon` still measures a settled flight and not a transient
    (at the default 1500 it is exactly step 1000)."""
    import warp as wp
    from falcons.envs.configs import altitude_env_cfg
    from falcons.controllers.policies import load_policy

    wp.init()
    tgt = np.repeat(np.array(TARGETS, dtype=np.float32), n_per_target)
    # every episode starts the same commanded climb below its target, so the per-target RMSEs
    # are comparable across targets as well as across plants
    alt0 = np.maximum(tgt - SPAWN_BELOW, SPAWN_MIN).astype(np.float32)
    # spawning_distance 0: the initial altitudes are written in directly, not drawn
    cfg = altitude_env_cfg(plane, 0.0, horizon=horizon + 1)
    policy = load_policy("ppo", "altitude", plane, 0, ckpt_dir=ckpt_dir).act
    settle = int(round(horizon * SETTLE / HORIZON))

    rw = rmse_per_target(tgt, _roll_warp(cfg, policy, tgt, alt0, horizon), settle)
    rt = rmse_per_target(tgt, _roll_torch(cfg, policy, tgt, alt0, horizon), settle)

    print(f"{plane} settled RMSE by target [m]  (n={n_per_target}/target, "
          f"steps {settle}-{horizon})")
    print("  target :", "  ".join(f"{int(t):4d}" for t in TARGETS))
    print("  warp   :", "  ".join(f"{v:4.2f}" for v in rw))
    print("  torch  :", "  ".join(f"{v:4.2f}" for v in rt))
    print(f"  overall warp {rw.mean():.3f}  torch {rt.mean():.3f}  "
          f"max |delta| {np.abs(rw - rt).max():.3f}", flush=True)
    return {"targets": list(TARGETS),
            "warp": {t: float(v) for t, v in zip(TARGETS, rw)},
            "torch": {t: float(v) for t, v in zip(TARGETS, rt)}}


# ─── throughput ─────────────────────────────────────────────────────────────────────────────
def bench_gpu(env_cls, n, device, cfg, steps, warmup=GPU_WARMUP, trials=R_TRIALS):
    """env-steps/s per trial for one batched GPU backend at batch `n`.

    `warmup` steps (JIT/kernel compile, allocator settle) are excluded, then `trials` timed trials
    of a fixed step count. Actions are uniform random -- the number is propagation cost, not policy
    cost -- and each trial is bracketed by `cuda.synchronize()` so the timer measures the device
    rather than the launch queue."""
    import torch
    env = env_cls(n, device, cfg)
    obs = env.reset()
    act = torch.empty(n, env.num_act, device=device)
    for _ in range(warmup):
        env.step(act.uniform_(-1, 1))
    torch.cuda.synchronize()
    rates = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            env.step(act.uniform_(-1, 1))
        torch.cuda.synchronize()
        rates.append(n * steps / (time.perf_counter() - t0))
    return rates


def bench_cpu(plane, steps, warmup=CPU_WARMUP, trials=R_TRIALS):
    """env-steps/s per trial for the single-aircraft Python/numpy plant (one env, by construction)."""
    from falcons.sim.cpu.aircraft import Aircraft
    env = Aircraft(plane)
    env.reset()
    aero = np.zeros(env.get_aero_action_size())
    motor = np.full(env.get_motor_action_size(), CPU_THROTTLE)
    for _ in range(warmup):
        env.step(aero, motor)
    rates = []
    for _ in range(trials):
        t0 = time.perf_counter()
        for _ in range(steps):
            env.step(aero, motor)
        rates.append(steps / (time.perf_counter() - t0))
    return rates


def throughput(plane, sizes=GPU_SIZES, trials=R_TRIALS, results_dir=RESULTS_DIR):
    """Sweep both GPU backends over `sizes` and the CPU plant at one env; write throughput.csv.

    Rows are reported as mean +- sample std with a 95 % Student-t interval. They are wall-clock and
    therefore hardware-bound; `TOLERANCE["throughput.csv"]` compares them at rel 0.5 (order of
    magnitude) for that reason, and the numbers are only meaningful on an otherwise idle GPU."""
    import torch
    import warp as wp
    wp.init()
    from falcons.envs.altitude import AltitudeEnv as WarpAltitudeEnv
    from falcons.sim.torch.altitude import AltitudeEnv as TorchAltitudeEnv
    from falcons.envs.configs import PLANE_CONFIGS, altitude_env_cfg

    os.makedirs(results_dir, exist_ok=True)
    cfg = altitude_env_cfg(plane, PLANE_CONFIGS[plane]["spawn"])

    rows = []
    print("benchmarking CPU python/numpy sim (1 env)...", flush=True)
    mu, sd, ci = stats(bench_cpu(plane, CPU_STEPS, trials=trials))
    rows.append(("cpu", 1, mu, sd, ci))
    cpu_mu = mu

    for name, cls in [("warp", WarpAltitudeEnv), ("torch", TorchAltitudeEnv)]:
        for n in sizes:
            print(f"benchmarking {name} n_envs={n}...", flush=True)
            try:
                mu, sd, ci = stats(bench_gpu(cls, n, "cuda", dict(cfg), GPU_STEPS, trials=trials))
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at n={n}, skipping")
                torch.cuda.empty_cache()
                continue
            rows.append((name, n, mu, sd, ci))

    out = Path(results_dir) / "throughput.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "n_envs", "steps_per_s_mean", "steps_per_s_std", "ci95"])
        w.writerows(rows)

    print("\n| backend | n_envs | env-steps/s (mean ± std) | 95% CI | speedup vs cpu |")
    print("|---|---|---|---|---|")
    for name, n, mu, sd, ci in rows:
        print(f"| {name} | {n} | {mu:,.0f} ± {sd:,.0f} | ±{ci:,.0f} | {mu / cpu_mu:,.0f}x |")
    print(f"wrote {out}")
    return out


def run(plane, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR):
    """Both claims: the throughput table, and the warp-vs-torch parity it rests on."""
    out = throughput(plane, results_dir=results_dir)
    d = parity(plane, ckpt_dir=ckpt_dir)
    with open(Path(results_dir) / "parity.json", "w") as f:
        json.dump(d, f, indent=2)
    print(f"wrote {Path(results_dir) / 'parity.json'}")
    return out
