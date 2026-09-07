"""TD3 (Fujimoto et al. 2018) for the warp envs: the deterministic-actor off-policy baseline that
sits alongside falcons/train/ppo.py (on-policy) and falcons/train/sac.py (max-entropy off-policy),
so all three learning families are evaluated on the same benchmark.

Twin critics + clipped double-Q target + target-policy smoothing + delayed (every `policy_delay`
critic updates) actor & target updates; exploration is a fixed Gaussian on the deterministic action
(no entropy term). Everything else -- replay, critic net, env setup, curriculum, best-eval
checkpointing, evaluate_altitude scoring -- is shared with falcons/train/sac.py, so the comparison
differs only in the algorithm, not the harness.

Run: falcons train --algo td3 --task altitude --plane Airship_V7
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from falcons.controllers.policies import ckpt_path
from falcons.paths import CKPT_DIR
# reuse SAC's replay, critic, mlp, and the eval shim -> identical harness, algo is the only change
from falcons.train.curves import log_curve
from falcons.train.sac import (BATCH, CURVES_DIR, DEVICE, GAMMA, LR, N_ENVS, REPLAY_CAP, STEPS,
                               TAU, UPDATES_PER_STEP, WARMUP)
from falcons.train.sac import Replay, QNet, mlp, _ActorMeanShim

EXPL_NOISE = 0.1       # std of the exploration Gaussian on the deterministic action (in [-1,1])
POLICY_NOISE = 0.2     # std of the target-policy smoothing noise
NOISE_CLIP = 0.5       # clip range for the smoothing noise
POLICY_DELAY = 2       # actor + target updates every this many critic updates


class DetActor(nn.Module):
    """Deterministic policy: tanh-squashed MLP. mean_action == forward (the shim + checkpoint
    loaders call mean_action, matching SquashedActor's interface). `act` = hidden activation
    (ReLU default; nn.Tanh for the PPO-matched jitter test)."""
    def __init__(self, obs_dim, act_dim, act=nn.ReLU):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, act)

    def forward(self, obs):
        return torch.tanh(self.net(obs))

    def mean_action(self, obs):
        return self.forward(obs)


def train(aircraft, task="altitude", steps=None, seed=0, ckpt_dir=CKPT_DIR,
          curves_dir=CURVES_DIR, n_envs=N_ENVS, lr=LR, gamma=GAMMA, tau=TAU, batch=BATCH,
          updates_per_step=UPDATES_PER_STEP, replay_cap=REPLAY_CAP, warmup=WARMUP,
          expl_noise=EXPL_NOISE, policy_noise=POLICY_NOISE, noise_clip=NOISE_CLIP,
          policy_delay=POLICY_DELAY, tanh=False):
    """Train TD3; returns the best-eval record {"score", "state", "msg"}. Signature mirrors
    falcons.train.sac.train (minus the entropy/CAPS knobs, plus TD3's exploration/smoothing/
    delay)."""
    import warp as wp

    steps = steps or STEPS
    torch.manual_seed(seed)
    np.random.seed(seed)
    wp.init()
    if task == "attitude":
        from falcons.envs.attitude import AttitudeEnv
        from falcons.envs.configs import attitude_env_cfg
        env = AttitudeEnv(n_envs, DEVICE, attitude_env_cfg(aircraft))
    else:
        from falcons.envs.altitude import AltitudeEnv
        from falcons.envs.configs import PLANE_CONFIGS, altitude_env_cfg
        spawn = PLANE_CONFIGS[aircraft]["spawn"]
        env = AltitudeEnv(n_envs, DEVICE, altitude_env_cfg(aircraft, spawn))
    O, A = env.num_obs, env.num_act

    ACT = nn.Tanh if tanh else nn.ReLU     # Tanh matches PPO (jitter-isolation test)
    actor, actor_t = DetActor(O, A, ACT).to(DEVICE), DetActor(O, A, ACT).to(DEVICE)
    actor_t.load_state_dict(actor.state_dict())
    q1, q2 = QNet(O, A, ACT).to(DEVICE), QNet(O, A, ACT).to(DEVICE)
    q1_t, q2_t = QNet(O, A, ACT).to(DEVICE), QNet(O, A, ACT).to(DEVICE)
    q1_t.load_state_dict(q1.state_dict()); q2_t.load_state_dict(q2.state_dict())
    for p in list(actor_t.parameters()) + list(q1_t.parameters()) + list(q2_t.parameters()):
        p.requires_grad_(False)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    replay = Replay(replay_cap, O, A, DEVICE)

    obs = env.reset()
    total_env_steps = steps
    iters = total_env_steps // n_envs
    t0 = time.time()
    rew_sum, rew_n = 0.0, 0
    upd = {"n": 0}
    print(f"TD3 {aircraft}: {n_envs} envs, {iters} iters -> {iters * n_envs:,} env-steps\n")

    best = {"score": None, "state": None, "msg": ""}
    curve_tag = f"td3_{task}_{aircraft}_s{seed}"

    def run_eval(label="", gstep=0):
        from falcons.train.tasks import evaluate_altitude, evaluate_attitude
        shim = _ActorMeanShim(actor)
        if task == "attitude":
            r = evaluate_attitude(shim, aircraft)
            score = (r[0], -r[1])
            msg = (f"robustness={r[0]:.3f} phi_rmse={r[1]:.2f}deg "
                   f"hdot_rmse={r[2]:.2f}m/s va_rmse={r[3]:.2f}")
            log_curve(curve_tag, gstep, curves_dir,
                      rob=r[0], phi_rmse=r[1], hdot_rmse=r[2], va_rmse=r[3])
        else:
            r = evaluate_altitude(shim, aircraft)
            score = (r[0], -r[1])
            msg = (f"robustness={r[0]:.3f} RMSE={r[1]:.3f}m mean={r[2]:.3f}m "
                   f"overshoot={r[3]:.3f}m band±2={r[4]:.3f}")
            log_curve(curve_tag, gstep, curves_dir,
                      rob=r[0], rmse=r[1], mean=r[2], overshoot=r[3], band=r[4])
        if best["score"] is None or score > best["score"]:
            best["score"] = score
            best["state"] = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
            best["msg"] = msg
            msg += "  <- new best"
        print(f"  [eval{label}] {msg}", flush=True)

    for it in range(iters):
        gstep = it * n_envs
        if task == "attitude":
            env.set_curriculum(gstep / (0.6 * total_env_steps))
        if gstep < warmup:
            action = torch.empty(n_envs, A, device=DEVICE).uniform_(-1.0, 1.0)
        else:
            with torch.no_grad():
                action = (actor(obs) + torch.randn(n_envs, A, device=DEVICE) * expl_noise)

        nobs, reward, done, info = env.step(action.clamp(-1.0, 1.0))
        terminal_obs = info["terminal_obs"]
        truncated = info["truncated"].bool()
        nxt = torch.where(done.unsqueeze(-1), terminal_obs, nobs)
        mask = 1.0 - (done & ~truncated).float()
        replay.push(obs, action.clamp(-1.0, 1.0), reward, nxt, mask)
        obs = nobs

        if gstep >= warmup:
            for _ in range(updates_per_step):
                b_obs, b_act, b_rew, b_nxt, b_mask = replay.sample(batch)

                with torch.no_grad():
                    noise = (torch.randn_like(b_act) * policy_noise).clamp(-noise_clip, noise_clip)
                    next_act = (actor_t(b_nxt) + noise).clamp(-1.0, 1.0)
                    min_q_next = torch.min(q1_t(b_nxt, next_act), q2_t(b_nxt, next_act))
                    target = b_rew + gamma * b_mask * min_q_next

                q_loss = (nn.functional.mse_loss(q1(b_obs, b_act), target)
                          + nn.functional.mse_loss(q2(b_obs, b_act), target))
                q_opt.zero_grad(set_to_none=True)
                q_loss.backward()
                q_opt.step()

                upd["n"] += 1
                if upd["n"] % policy_delay == 0:      # delayed actor + target updates
                    actor_loss = -q1(b_obs, actor(b_obs)).mean()
                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_opt.step()
                    with torch.no_grad():
                        for net, net_t in ((q1, q1_t), (q2, q2_t), (actor, actor_t)):
                            for p, pt in zip(net.parameters(), net_t.parameters()):
                                pt.mul_(1.0 - tau).add_(tau * p)

        rew_sum += reward.mean().item()
        rew_n += 1
        if it % 500 == 0:
            print(f"it={it} gstep={gstep:,} rew/step={rew_sum / rew_n:6.3f} "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)
            rew_sum, rew_n = 0.0, 0

        if it and it % max(1, iters // 8) == 0 and gstep >= warmup:
            run_eval(label=f" it {it}", gstep=gstep)

    run_eval(label=" final", gstep=iters * n_envs)

    out = ckpt_path("td3", task, aircraft, seed, ckpt_dir)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], out)
    print(f"EVAL {aircraft} (td3 {task} s{seed}, best checkpoint): {best['msg']} -> {out}")
    return best
