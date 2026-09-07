"""The one place that knows how checkpoints are named and loaded.

checkpoints/{algo}_{task}_{plane}_s{seed}.pt
  ppo -> falcons.train.ppo.Agent            (.actor_mean)
  sac -> falcons.train.sac.SquashedActor    (.mean_action)
  td3 -> falcons.train.td3.DetActor         (.mean_action)
lqr / mppi are not checkpoints: they are the classical attitude executors, built on an env.
"""
from pathlib import Path

import torch

from falcons.paths import CKPT_DIR

ALGOS = ("ppo", "sac", "td3")
CLASSICAL = ("lqr", "mppi")
TASKS = ("altitude", "attitude")
OBS_ACT = {"altitude": (11, 2), "attitude": (15, 4)}   # observation and action widths per task


def ckpt_path(algo, task, plane, seed=0, ckpt_dir=CKPT_DIR) -> Path:
    return Path(ckpt_dir) / f"{algo}_{task}_{plane}_s{seed}.pt"


def seeds(algo, task, plane, ckpt_dir=CKPT_DIR) -> list:
    """Seeds that exist for this (algo, task, plane), ascending. Classical executors report [0]."""
    if algo in CLASSICAL:
        return [0]
    return sorted(int(p.stem.rsplit("_s", 1)[1]) for p in Path(ckpt_dir).glob(f"{algo}_{task}_{plane}_s*.pt"))


class Policy:
    """Uniform `.act(obs) -> action` over the three actor classes."""
    def __init__(self, fn):
        self.act = fn


def load_policy(algo, task, plane, seed=0, device="cuda", env=None, stream=None, ckpt_dir=CKPT_DIR):
    if algo in CLASSICAL:
        if env is None:
            raise ValueError(f"{algo} is a classical executor and needs env=")
        if task != "attitude":
            raise ValueError("classical executors load through falcons.controllers.{lqr,mppi}.altitude "
                             "for the altitude task")
        if algo == "lqr":
            from falcons.controllers.lqr.attitude import LQRAttitudeExecutor
            return Policy(LQRAttitudeExecutor(plane, env).actor_mean)
        from falcons.controllers.mppi.attitude import MPPIAttitudeExecutor
        if stream is None:
            raise ValueError("mppi needs stream=(phi, hdot, va) for its preview horizon")
        return Policy(MPPIAttitudeExecutor(plane, env, *stream, seed=seed).actor_mean)
    path = ckpt_path(algo, task, plane, seed, ckpt_dir)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint {path}; have seeds {seeds(algo, task, plane, ckpt_dir)}")
    state = torch.load(path, map_location=device)
    n_obs, n_act = OBS_ACT[task]
    if algo == "ppo":
        from falcons.train.ppo import Agent
        net = Agent(n_obs, n_act).to(device); net.load_state_dict(state); net.eval()
        return Policy(net.actor_mean)
    if algo == "sac":
        from falcons.train.sac import SquashedActor
        net = SquashedActor(n_obs, n_act).to(device); net.load_state_dict(state); net.eval()
        return Policy(net.mean_action)
    from falcons.train.td3 import DetActor
    net = DetActor(n_obs, n_act).to(device); net.load_state_dict(state); net.eval()
    return Policy(net.mean_action)
