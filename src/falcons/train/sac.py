"""SAC for the batched warp envs (altitude + attitude tasks). Off-policy companion to
falcons/train/ppo.py: same env contract, replay-based. Correct tuples come free from
info["terminal_obs"] + info["truncated"] (bootstrap mask = done & ~truncated).

Run: falcons train --algo sac --task altitude --plane Airship_V7
     falcons train --algo sac --task attitude --plane Airship_V7
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from falcons.controllers.policies import ckpt_path
from falcons.paths import CKPT_DIR, RESULTS_DIR
from falcons.train.curves import log_curve

DEVICE = "cuda"
N_ENVS = 256           # off-policy: keep collect:update ratio sane (1024 envs x 1 update = ~29k
                       # updates for 30M steps, ~20x too few for SAC -> rob 0.00). 256 envs x 8
                       # updates/iter gives a healthy update-to-data ratio.
GAMMA, TAU, LR = 0.99, 0.005, 3e-4
BATCH = 8192
WARMUP = 20_000        # random-action transitions before learning
UPDATES_PER_STEP = 8   # gradient updates per env-step batch (raise UTD: SAC needs ~100k-1M updates)
REPLAY_CAP = 1_000_000
STEPS = 30_000_000     # default budget, the one the shipped checkpoints were trained at
CURVES_DIR = RESULTS_DIR / "curves"

# CAPS smoothness (Mysore et al.), ported from falcons/train/ppo.py. OFF by default (the shipped
# checkpoints are baseline SAC). SAC survives the attitude-stream maneuvers but tracks loosely
# (phi RMSE 3-39deg vs PPO 0.4-2.4deg) because its actor is jittery; CAPS regularizes the
# deterministic MEAN action (orthogonal to the entropy exploration term). Defaults larger than
# PPO's 0.2/0.1: SAC's actor loss is scaled by Q-value magnitude (returns over horizon), not
# normalized advantages, so the same weights bite far less.
CAPS_LAMBDA_T = 0.5    # temporal: penalize mean(obs) drifting from mean(next_obs) -> smooth in time
CAPS_LAMBDA_S = 0.25   # spatial: penalize mean(obs) sensitivity to obs noise -> Lipschitz / low jitter
CAPS_SIGMA = 0.05      # spatial-perturbation std (obs are normalized-ish, matches PPO)


def caps_penalty(actor, obs, nxt, mask, lam_t, lam_s, sigma, dims=None):
    """CAPS smoothness term on the actor's deterministic mean action. `mask` (1=bootstrap,
    0=terminal) zeroes temporal pairs that straddle an episode boundary (not time-consecutive).
    `dims` restricts the penalty to a subset of action channels — e.g. [0,1,2] (elev/ail/rud)
    to leave THROTTLE free: smoothing throttle starves airspeed tracking (Va RMSE blew up 4x on
    Volantex when throttle was included)."""
    mean_a = actor.mean_action(obs)
    mean_n = actor.mean_action(nxt)
    mean_s = actor.mean_action(obs + torch.randn_like(obs) * sigma)
    if dims is not None:
        mean_a, mean_n, mean_s = mean_a[:, dims], mean_n[:, dims], mean_s[:, dims]
    temporal = (torch.norm(mean_a - mean_n, dim=1) * mask).mean()
    spatial = torch.norm(mean_a - mean_s, dim=1).mean()
    return lam_t * temporal + lam_s * spatial


class Replay:
    def __init__(self, cap, obs_dim, act_dim, device):
        self.cap, self.device, self.i, self.full = cap, device, 0, False
        self.obs = torch.zeros(cap, obs_dim, device=device)
        self.act = torch.zeros(cap, act_dim, device=device)
        self.rew = torch.zeros(cap, device=device)
        self.nxt = torch.zeros(cap, obs_dim, device=device)
        self.mask = torch.zeros(cap, device=device)      # 1 = bootstrap, 0 = terminal

    def __len__(self):
        return self.cap if self.full else self.i

    def push(self, obs, act, rew, nxt, mask):
        n = obs.shape[0]
        idx = (torch.arange(n, device=self.device) + self.i) % self.cap
        self.obs[idx], self.act[idx], self.rew[idx] = obs, act, rew
        self.nxt[idx], self.mask[idx] = nxt, mask
        self.i = int((self.i + n) % self.cap)
        self.full = self.full or (self.i < n) or len(self) == self.cap

    def sample(self, bs):
        idx = torch.randint(0, len(self), (bs,), device=self.device)
        return self.obs[idx], self.act[idx], self.rew[idx], self.nxt[idx], self.mask[idx]


def mlp(i, o, act=nn.ReLU):
    """2x256 MLP. act defaults to ReLU (the trained checkpoints); pass nn.Tanh to match PPO's
    smooth activation (jitter isolation test — Tanh nets are Lipschitz, ReLU are piecewise-linear
    with sharp kinks that chatter)."""
    return nn.Sequential(nn.Linear(i, 256), act(), nn.Linear(256, 256), act(),
                         nn.Linear(256, o))


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, act=nn.ReLU):
        super().__init__()
        self.q = mlp(obs_dim + act_dim, 1, act)

    def forward(self, o, a):
        return self.q(torch.cat([o, a], -1)).squeeze(-1)


class SquashedActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act=nn.ReLU):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, act)
        self.act_dim = act_dim

    def sample(self, obs):
        mu, logstd = self.net(obs).chunk(2, -1)
        logstd = logstd.clamp(-5, 2)
        dist = torch.distributions.Normal(mu, logstd.exp())
        x = dist.rsample()
        a = torch.tanh(x)
        logp = dist.log_prob(x).sum(-1) - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp

    def mean_action(self, obs):
        mu, _ = self.net(obs).chunk(2, -1)
        return torch.tanh(mu)


class _ActorMeanShim:
    """Wraps a SquashedActor so falcons.train.tasks' evaluators' agent.actor_mean(obs) call works."""
    def __init__(self, actor):
        self.actor = actor

    def actor_mean(self, obs):
        return self.actor.mean_action(obs)


def train(aircraft, task="altitude", steps=None, seed=0, ckpt_dir=CKPT_DIR,
          curves_dir=CURVES_DIR, target_entropy=None, n_envs=N_ENVS, lr=LR, gamma=GAMMA,
          tau=TAU, batch=BATCH, updates_per_step=UPDATES_PER_STEP, replay_cap=REPLAY_CAP,
          warmup=WARMUP, caps=False, caps_t=CAPS_LAMBDA_T, caps_s=CAPS_LAMBDA_S,
          caps_sigma=CAPS_SIGMA, caps_dims=None, tanh=False):
    """Train SAC; returns the best-eval record {"score", "state", "msg"}."""
    import warp as wp

    steps = steps or STEPS
    torch.manual_seed(seed)
    np.random.seed(seed)
    # NB: falcons.train.tasks' env helper hardcodes PPO's N_ENVS (4096); SAC uses its own
    # (smaller) n_envs since gradient steps, not sim throughput, are the bottleneck here. So
    # build the env directly rather than via that helper.
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
    actor = SquashedActor(O, A, ACT).to(DEVICE)
    q1, q2 = QNet(O, A, ACT).to(DEVICE), QNet(O, A, ACT).to(DEVICE)
    q1_t, q2_t = QNet(O, A, ACT).to(DEVICE), QNet(O, A, ACT).to(DEVICE)
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())
    for p in list(q1_t.parameters()) + list(q2_t.parameters()):
        p.requires_grad_(False)

    log_alpha = torch.zeros(1, device=DEVICE, requires_grad=True)
    target_entropy = target_entropy if target_entropy is not None else -float(A)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    alpha_opt = torch.optim.Adam([log_alpha], lr=lr)

    replay = Replay(replay_cap, O, A, DEVICE)

    obs = env.reset()
    total_env_steps = steps
    iters = total_env_steps // n_envs
    t0 = time.time()
    rew_sum, rew_n = 0.0, 0
    print(f"SAC {aircraft}: {n_envs} envs, {iters} iters -> {iters * n_envs:,} env-steps\n")

    # Best-eval checkpointing: SAC drifts late in training (e.g. A0S attitude peaked
    # rob 0.847 at 75% then decayed to 0.571), so the saved policy is the best evaluated
    # one, not the last. Score: (robustness, -primary tracking RMSE).
    best = {"score": None, "state": None, "msg": ""}
    curve_tag = f"sac_{task}_{aircraft}_s{seed}"

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
            # same target-magnitude curriculum as PPO's train_attitude: gentle -> full
            # phi/hdot range over the first 60% of training
            env.set_curriculum(gstep / (0.6 * total_env_steps))
        if gstep < warmup:
            action = torch.empty(n_envs, A, device=DEVICE).uniform_(-1.0, 1.0)
        else:
            with torch.no_grad():
                action, _ = actor.sample(obs)

        nobs, reward, done, info = env.step(action.clamp(-1.0, 1.0))
        terminal_obs = info["terminal_obs"]
        truncated = info["truncated"].bool()
        nxt = torch.where(done.unsqueeze(-1), terminal_obs, nobs)
        mask = 1.0 - (done & ~truncated).float()
        replay.push(obs, action, reward, nxt, mask)
        obs = nobs

        if gstep >= warmup:
            for _ in range(updates_per_step):
                b_obs, b_act, b_rew, b_nxt, b_mask = replay.sample(batch)
                alpha = log_alpha.exp().detach()

                with torch.no_grad():
                    next_act, next_logp = actor.sample(b_nxt)
                    q1_next = q1_t(b_nxt, next_act)
                    q2_next = q2_t(b_nxt, next_act)
                    min_q_next = torch.min(q1_next, q2_next) - alpha * next_logp
                    target = b_rew + gamma * b_mask * min_q_next

                q1_pred = q1(b_obs, b_act)
                q2_pred = q2(b_obs, b_act)
                q_loss = nn.functional.mse_loss(q1_pred, target) + nn.functional.mse_loss(q2_pred, target)
                q_opt.zero_grad(set_to_none=True)
                q_loss.backward()
                q_opt.step()

                new_act, logp = actor.sample(b_obs)
                q1_pi = q1(b_obs, new_act)
                q2_pi = q2(b_obs, new_act)
                min_q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (log_alpha.exp().detach() * logp - min_q_pi).mean()
                if caps:
                    actor_loss = actor_loss + caps_penalty(
                        actor, b_obs, b_nxt, b_mask, caps_t, caps_s, caps_sigma, dims=caps_dims)
                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_opt.step()

                alpha_loss = -(log_alpha * (logp.detach() + target_entropy)).mean()
                alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                alpha_opt.step()

                with torch.no_grad():
                    for p, pt in zip(q1.parameters(), q1_t.parameters()):
                        pt.mul_(1.0 - tau).add_(tau * p)
                    for p, pt in zip(q2.parameters(), q2_t.parameters()):
                        pt.mul_(1.0 - tau).add_(tau * p)

        rew_sum += reward.mean().item()
        rew_n += 1
        if it % 500 == 0:
            elapsed = time.time() - t0
            print(f"it={it} gstep={gstep:,} rew/step={rew_sum / rew_n:6.3f} "
                  f"elapsed={elapsed:.1f}s", flush=True)
            rew_sum, rew_n = 0.0, 0

        if it and it % max(1, iters // 8) == 0 and gstep >= warmup:
            run_eval(label=f" it {it}", gstep=gstep)

    run_eval(label=" final", gstep=iters * n_envs)

    out = ckpt_path("sac", task, aircraft, seed, ckpt_dir)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], out)
    print(f"EVAL {aircraft} (sac {task} s{seed}, best checkpoint): {best['msg']} -> {out}")
    return best
