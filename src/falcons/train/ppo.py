"""Shared CleanRL-style PPO core for the batched warp envs.

One Agent (2x512 tanh MLP actor+critic, scheduled logstd) and one training loop
(rollout -> truncation bootstrap -> GAE -> clipped update + CAPS smoothness) used
by both tasks. Per-task env cfgs live in falcons.envs.configs, task setup in
falcons.train.tasks.

State-dict keys (critic.*, actor_mean.*, logstd) are checkpoint-format: do not rename.
"""
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

DEVICE = "cuda"
N_ENVS = 4096
ROLLOUT = 32
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP = 0.2
EPOCHS = 4
MINIBATCHES = 4
LR = 3e-4
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
CAPS_LAMBDA_T = 0.2          # temporal action smoothness (Chowdhury rate-penalty / CAPS)
CAPS_LAMBDA_S = 0.1          # spatial action smoothness
CAPS_SIGMA = 0.05
STD_START = 1.0
NET = 512


def layer_init(layer, std=np.sqrt(2.0), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, NET)), nn.Tanh(),
            layer_init(nn.Linear(NET, NET)), nn.Tanh(),
            layer_init(nn.Linear(NET, 1), std=1.0))
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, NET)), nn.Tanh(),
            layer_init(nn.Linear(NET, NET)), nn.Tanh(),
            layer_init(nn.Linear(NET, n_act), std=0.01))
        self.register_buffer("logstd", torch.zeros(1, n_act))   # scheduled, not learned

    def set_std(self, std):
        self.logstd.fill_(float(np.log(std)))

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        std = torch.exp(self.logstd).expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.critic(x).squeeze(-1)


def train_loop(env, agent, *, total_steps, lr0=LR, std0=STD_START, std1=0.5,
               caps_t=CAPS_LAMBDA_T, caps_s=CAPS_LAMBDA_S, n_envs=N_ENVS,
               rollout=ROLLOUT, device=DEVICE, on_iter=None, post_iter=None, log_prefix="PPO"):
    """Run the shared PPO loop; returns the trained agent.

    on_iter(it, iters, env, agent): per-iteration curriculum hook (spawn/task magnitude).
    post_iter(it, iters, agent):    after-update hook (periodic eval / learning curve).
    """
    opt = torch.optim.Adam(agent.parameters(), lr=lr0, eps=1e-5)

    N, T, O, A = n_envs, rollout, env.num_obs, env.num_act
    obs_b = torch.zeros(T, N, O, device=device); nobs_b = torch.zeros(T, N, O, device=device)
    act_b = torch.zeros(T, N, A, device=device); logp_b = torch.zeros(T, N, device=device)
    rew_b = torch.zeros(T, N, device=device); done_b = torch.zeros(T, N, device=device)
    val_b = torch.zeros(T, N, device=device)

    obs = env.reset()
    iters = total_steps // (T * N)
    batch = T * N
    mb = batch // MINIBATCHES
    print(f"{log_prefix}: {N}x{T}={batch} batch, {iters} iters -> {iters*batch:,} steps\n")
    t0 = time.time(); gstep = 0
    for it in range(iters):
        agent.set_std(std0 + (it / max(1.0, iters - 1)) * (std1 - std0))
        for g in opt.param_groups:
            g["lr"] = lr0 * (1.0 - it / iters)
        if on_iter is not None:
            on_iter(it, iters, env, agent)
        for t in range(T):
            obs_b[t] = obs
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(obs)
            nobs, reward, done, info = env.step(action.clamp(-1.0, 1.0))
            nobs_b[t] = nobs
            trunc = info["truncated"].bool()
            if trunc.any():
                with torch.no_grad():
                    reward = reward.clone()
                    reward[trunc] += GAMMA * agent.get_value(info["terminal_obs"][trunc])
            act_b[t] = action; logp_b[t] = logp; val_b[t] = value
            rew_b[t] = reward; done_b[t] = done.float()
            obs = nobs; gstep += N

        with torch.no_grad():
            next_val = agent.get_value(obs)
            adv = torch.zeros_like(rew_b); last = 0.0
            for t in reversed(range(T)):
                nnt = 1.0 - done_b[t]
                nv = next_val if t == T - 1 else val_b[t + 1]
                delta = rew_b[t] + GAMMA * nv * nnt - val_b[t]
                adv[t] = last = delta + GAMMA * GAE_LAMBDA * nnt * last
            ret = adv + val_b

        b_obs = obs_b.reshape(-1, O); b_nobs = nobs_b.reshape(-1, O)
        b_nt = (1.0 - done_b).reshape(-1); b_act = act_b.reshape(-1, A)
        b_logp = logp_b.reshape(-1); b_adv = adv.reshape(-1); b_ret = ret.reshape(-1)
        idx = torch.randperm(batch, device=device)
        for _ in range(EPOCHS):
            for s in range(0, batch, mb):
                mi = idx[s:s + mb]
                _, nlogp, ent, nval = agent.get_action_and_value(b_obs[mi], b_act[mi])
                ratio = (nlogp - b_logp[mi]).exp()
                a = b_adv[mi]; a = (a - a.mean()) / (a.std() + 1e-8)
                pg = torch.max(-a * ratio, -a * ratio.clamp(1 - CLIP, 1 + CLIP)).mean()
                vloss = 0.5 * ((nval - b_ret[mi]) ** 2).mean()
                mean_a = agent.actor_mean(b_obs[mi])
                caps_tv = (torch.norm(mean_a - agent.actor_mean(b_nobs[mi]), dim=1) * b_nt[mi]).mean()
                caps_sv = torch.norm(mean_a - agent.actor_mean(
                    b_obs[mi] + torch.randn_like(b_obs[mi]) * CAPS_SIGMA), dim=1).mean()
                loss = pg + VF_COEF * vloss + caps_t * caps_tv + caps_s * caps_sv
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM); opt.step()

        if it % 40 == 0 or it == iters - 1:
            print(f"it {it:3d} | step {gstep:>9,} | ret/step {rew_b.mean().item():6.3f} | "
                  f"{gstep/(time.time()-t0):,.0f} sps")
        if post_iter is not None:
            post_iter(it, iters, agent)

    print(f"\nDONE: {gstep:,} steps in {time.time()-t0:.1f}s")
    return agent
