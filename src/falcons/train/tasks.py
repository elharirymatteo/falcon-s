"""Per-task training entry points + final evaluations.

Each task builds its env, runs the shared PPO loop, saves the checkpoint under the one
checkpoint name (falcons.controllers.policies.ckpt_path), logs a learning curve, and prints
a final eval.
"""
from pathlib import Path

import numpy as np
import torch

from falcons.controllers.policies import ckpt_path
from falcons.envs.configs import (PLANE_CONFIGS, SPAWN_START, TASK_DEFAULTS, altitude_env_cfg,
                                  attitude_env_cfg)
from falcons.paths import CKPT_DIR, RESULTS_DIR
from falcons.train.curves import log_curve
from falcons.train.ppo import Agent, DEVICE, N_ENVS, ROLLOUT, LR, STD_START, train_loop

CURVES_DIR = RESULTS_DIR / "curves"


# ------------------------------------------------------------------ altitude

@torch.no_grad()
def evaluate_altitude(agent, aircraft, steps=2000, settle=200):
    """Robustness (no crash/stall) + RMSE/mean/overshoot/band vs the (ramping) target."""
    cfg = altitude_env_cfg(aircraft, PLANE_CONFIGS[aircraft]["spawn"], steps + 1)
    env = _make_altitude_env(cfg)
    env.obs_noise = 0.0                          # measure clean tracking (noise is a train-time tool)
    obs = env.reset()
    failed = torch.zeros(N_ENVS, dtype=torch.bool, device=DEVICE)
    sq = torch.zeros(N_ENVS, device=DEVICE); err = torch.zeros(N_ENVS, device=DEVICE)
    mx = torch.zeros(N_ENVS, device=DEVICE); band = torch.zeros(N_ENVS, device=DEVICE); cnt = 0
    for t in range(steps):
        obs, r, d, info = env.step(agent.actor_mean(obs).clamp(-1.0, 1.0))
        failed |= (info["reason"] == 1) | (info["reason"] == 2)
        if t >= settle:
            h = obs[:, 0].abs() * 25.0
            sq += h ** 2; err += h; mx = torch.maximum(mx, h); band += (h < 2.0).float(); cnt += 1
    return (1.0 - failed.float().mean().item(), torch.sqrt(sq / cnt).mean().item(),
            (err / cnt).mean().item(), mx.mean().item(), (band / cnt).mean().item())


def _make_altitude_env(cfg):
    import warp as wp
    from falcons.envs.altitude import AltitudeEnv
    wp.init()
    return AltitudeEnv(N_ENVS, DEVICE, cfg)


def train_altitude(aircraft, total_steps=None, seed=0, ckpt_dir=CKPT_DIR, curves_dir=CURVES_DIR):
    assert aircraft in PLANE_CONFIGS, f"unknown aircraft {aircraft}"
    total_steps = total_steps or TASK_DEFAULTS["altitude"]["total_steps"]

    torch.manual_seed(seed)
    cfg = PLANE_CONFIGS[aircraft]
    spawn_max = cfg["spawn"]
    env = _make_altitude_env(altitude_env_cfg(aircraft, spawn_max))
    agent = Agent(env.num_obs, env.num_act).to(DEVICE)

    def on_iter(it, iters, env, agent):
        env.spawn = SPAWN_START + min(1.0, it / max(1.0, iters * 0.6)) * (spawn_max - SPAWN_START)

    def post_iter(it, iters, agent):
        if it == iters - 1 or (it and it % max(1, iters // 8) == 0):
            r = evaluate_altitude(agent, aircraft)
            log_curve(f"ppo_altitude_{aircraft}_s{seed}", (it + 1) * N_ENVS * ROLLOUT, curves_dir,
                      rob=r[0], rmse=r[1], mean=r[2], overshoot=r[3], band=r[4])
            print(f"  [eval it {it}] robustness={r[0]:.3f} RMSE={r[1]:.3f}m "
                  f"overshoot={r[3]:.3f}m band±2={r[4]:.3f}")

    train_loop(env, agent, total_steps=total_steps, lr0=LR, std0=STD_START,
               std1=TASK_DEFAULTS["altitude"]["std_end"], on_iter=on_iter, post_iter=post_iter,
               log_prefix=f"PPO {aircraft} cfg={cfg}")

    out = ckpt_path("ppo", "altitude", aircraft, seed, ckpt_dir)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), out)
    rob, rmse, mean, over, band = evaluate_altitude(agent, aircraft)
    print(f"EVAL {aircraft} (altitude s{seed}): robustness={rob:.3f} RMSE={rmse:.3f}m "
          f"mean={mean:.3f}m overshoot={over:.3f}m band±2={band:.3f} -> {out}")


# ------------------------------------------------------------------ attitude executor

@torch.no_grad()
def evaluate_attitude(agent, aircraft, steps=2000, settle=200):
    """Robustness (no crash/stall/bank-out) + attitude/speed tracking RMSE over the
    piecewise target segments. Clean obs (no noise). phi RMSE in degrees, hdot RMSE in m/s,
    va RMSE in m/s."""
    from falcons.envs.attitude import AttitudeEnv
    env = AttitudeEnv(N_ENVS, DEVICE, attitude_env_cfg(aircraft, horizon=steps + 1))
    obs = env.reset()
    failed = torch.zeros(N_ENVS, dtype=torch.bool, device=DEVICE)
    phi_sq = torch.zeros(N_ENVS, device=DEVICE)
    hdot_sq = torch.zeros(N_ENVS, device=DEVICE)
    va_sq = torch.zeros(N_ENVS, device=DEVICE)
    cnt = 0
    for t in range(steps):
        obs, r, d, info = env.step(agent.actor_mean(obs).clamp(-1.0, 1.0))
        failed |= (info["reason"] == 1) | (info["reason"] == 2) | (info["reason"] == 3)
        if t >= settle:
            # obs[:,0]=phi_err/(pi/4), obs[:,1]=hdot_err/vz_scale, obs[:,2]=va_err/va_scale
            phi_err = obs[:, 0] * (np.pi / 4)
            hdot_err = obs[:, 1] * env.vz_scale
            va_err = obs[:, 2] * env.va_scale
            phi_sq += phi_err ** 2; hdot_sq += hdot_err ** 2; va_sq += va_err ** 2
            cnt += 1
    rob = 1.0 - failed.float().mean().item()
    phi_rmse = float(np.degrees(torch.sqrt(phi_sq / cnt).mean().item()))
    hdot_rmse = torch.sqrt(hdot_sq / cnt).mean().item()
    va_rmse = torch.sqrt(va_sq / cnt).mean().item()
    settle_time = settle * env.dt   # settle-time proxy: report the fixed settle window (v1)
    return (rob, phi_rmse, hdot_rmse, va_rmse, settle_time)


def train_attitude(aircraft, total_steps=None, seed=0, ckpt_dir=CKPT_DIR, curves_dir=CURVES_DIR):
    import warp as wp
    from falcons.envs.attitude import AttitudeEnv
    total_steps = total_steps or TASK_DEFAULTS["attitude"]["total_steps"]
    wp.init()
    torch.manual_seed(seed)
    env = AttitudeEnv(N_ENVS, DEVICE, attitude_env_cfg(aircraft))
    agent = Agent(env.num_obs, env.num_act).to(DEVICE)

    def post_iter(it, iters, agent):
        if it == iters - 1 or (it and it % max(1, iters // 8) == 0) \
                or (it >= int(iters * 0.5) and it % 200 == 0):
            r = evaluate_attitude(agent, aircraft)
            log_curve(f"ppo_attitude_{aircraft}_s{seed}", (it + 1) * N_ENVS * ROLLOUT, curves_dir,
                      rob=r[0], phi_rmse=r[1], hdot_rmse=r[2], va_rmse=r[3])
            print(f"  [eval it {it}] robustness={r[0]:.3f} phi_rmse={r[1]:.2f}deg "
                  f"hdot_rmse={r[2]:.2f}m/s va_rmse={r[3]:.2f}")

    def on_iter(it, iters, env, agent):
        env.set_curriculum(it / max(1.0, iters * 0.6))   # phi/hdot grow tiny -> full over first 60%

    train_loop(env, agent, total_steps=total_steps, lr0=LR, std0=STD_START,
               std1=TASK_DEFAULTS["attitude"]["std_end"], post_iter=post_iter, on_iter=on_iter,
               log_prefix=f"PPO attitude {aircraft}")

    out = ckpt_path("ppo", "attitude", aircraft, seed, ckpt_dir)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), out)
    r = evaluate_attitude(agent, aircraft)
    print(f"EVAL {aircraft} (attitude s{seed}): robustness={r[0]:.3f} phi_rmse={r[1]:.2f}deg "
          f"hdot_rmse={r[2]:.2f}m/s va_rmse={r[3]:.2f} settle={r[4]:.2f}s -> {out}")
