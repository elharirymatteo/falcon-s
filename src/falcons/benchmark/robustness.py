"""Robustness of the attitude-executor policy (PPO) under four disturbance families -- action
delay, sensor delay, observation noise and Dryden wind -- each swept mild -> medium -> severe with
values grounded in small-UAV hardware and low-altitude turbulence physics. Same maneuver streams
and envelope guard as the clean comparison (benchmark.maneuvers.fly); metrics always come from the
TRUE state, never the disturbed obs.

One executor, PPO seed 0: a disturbance corrupts what the policy sees or commands and never what
is scored, so a cell compares CONDITIONS rather than methods.

Full derivation and citations: docs/disturbance_model.md.
Run: run(planes) -> <results_dir>/robustness.csv
"""
import csv
import os
import zlib
from pathlib import Path

import numpy as np
import torch

from falcons.benchmark.maneuvers import fly, make_env
from falcons.benchmark.streams import DT, MANEUVERS
from falcons.paths import CKPT_DIR, RESULTS_DIR

SEVERITIES = ["mild", "medium", "severe"]

# --- grounded severity grids -------------------------------------------------
# actuator/command transport lag on top of the modeled servo dynamics: FC compute + link + servo
# deadtime, at DT = 0.01 s = 2/5/10 steps
ACTION_DELAY_MS = {"mild": 20, "medium": 50, "severe": 100}
# obs latency (EKF fusion), 2/4/8 steps; 20 ms is the figure the clean comparison assumes
SENSOR_DELAY_MS = {"mild": 20, "medium": 40, "severe": 80}
# Dryden 6-component gust field, dialed to a target longitudinal intensity Iu = sigma_u / Va;
# a literal MIL W20 of 15-45 kt would exceed airspeed for the 15-28 m/s class
WIND_IU = {"mild": 0.05, "medium": 0.10, "severe": 0.15}     # target sigma_u / Va
# regridded finer: the first sweep (0.10/0.20/0.30) crashed every method at medium+, losing all
# resolution. Iu<=0.15 is the survivable band where the executors actually separate; 0.05-0.15
# spans light -> moderate low-altitude turbulence for the 15-28 m/s class.
# physical sensor-noise floors (SI), mapped to obs-space per channel below
NOISE = {  # (attitude rad, rate rad/s, Va m/s, alpha/beta rad, climb m/s)
    "mild":   dict(att=np.radians(0.5), rate=np.radians(1.0), va=0.2, ab=np.radians(0.5), vz=0.10),
    "medium": dict(att=np.radians(1.0), rate=np.radians(2.0), va=0.5, ab=np.radians(1.0), vz=0.25),
    "severe": dict(att=np.radians(2.0), rate=np.radians(5.0), va=1.0, ab=np.radians(2.0), vz=0.50),
}


def steps_from_ms(ms):
    return int(round(ms / 1000.0 / DT))


def obs_noise_sigma(sev, va_scale, vz_scale):
    """15-dim obs-space std for a severity. Channels 11-14 (prev action) get no sensor noise."""
    n = NOISE[sev]
    q = PI4 = np.pi / 4.0
    s = np.zeros(15, dtype=np.float32)
    s[0] = n["att"] / q          # roll error
    s[1] = n["vz"] / vz_scale    # climb error
    s[2] = n["va"] / va_scale    # Va error
    s[3] = n["att"] / q          # roll
    s[4] = n["att"] / q          # pitch
    s[5] = s[6] = s[7] = n["rate"] / 2.0     # body rates
    s[8] = n["ab"] / 0.349       # alpha
    s[9] = n["ab"] / 0.349       # beta
    s[10] = n["vz"] / vz_scale   # vz
    return s


# --- disturbance wrappers (stateful; one instance per run) -------------------
class ActionDelay:
    """Actuator receives the command from k steps ago (neutral buffer before the loop fills)."""
    def __init__(self, fn, k):
        self.fn, self.k, self.buf = fn, k, []

    def __call__(self, obs):
        a = self.fn(obs)
        self.buf.append(a)
        return self.buf.pop(0) if len(self.buf) > self.k else torch.zeros_like(a)


class SensorDelay:
    """Policy sees the obs from k steps ago (holds the initial obs before the buffer fills).
    obs MUST be cloned: the harness hands out wp.to_torch(e._obs), a view of the one warp obs
    buffer that is overwritten in place every step — buffering the view alone makes the delay a
    silent no-op (identical metrics at every severity)."""
    def __init__(self, fn, k):
        self.fn, self.k, self.buf = fn, k, []

    def __call__(self, obs):
        obs = obs.clone()
        self.buf.append(obs)
        old = self.buf.pop(0) if len(self.buf) > self.k else self.buf[0]
        return self.fn(old)


class ObsNoise:
    """Gaussian sensor noise from a generator of its own, seeded per cell by `noise_seed` so the
    draw is a property of the (aircraft, severity) cell rather than of the process. The golden rows
    this family is gated against were drawn from torch's unseeded global CUDA generator, so they
    are a single draw that no re-run reproduces (repeat-to-repeat spread 0.018 deg of bank RMSE),
    which is why TOLERANCE bounds obs_noise instead of demanding equality."""
    def __init__(self, fn, sigma, seed):
        self.fn = fn
        self.sigma = torch.tensor(sigma, device="cuda")
        self.gen = torch.Generator(device="cuda")
        self.gen.manual_seed(int(seed))

    def __call__(self, obs):
        n = torch.randn(obs.shape, generator=self.gen, device=obs.device, dtype=obs.dtype)
        return self.fn(obs + n * self.sigma)


def noise_seed(aircraft, sev):
    """Stable 32-bit seed for one sensor-noise cell. crc32 rather than hash(): Python salts str
    hashes per process, which would make the draw a property of the process again."""
    return zlib.crc32(f"{aircraft}/obs_noise/{sev}".encode())


PI4 = np.pi / 4.0


def calibrate_turb(aircraft):
    """Realized sigma_u is linear in W20 (Dryden filter). Measure the constant k = sigma_u/W20
    once per airframe so we can set W20 = Iu*Va / k to hit a target intensity."""
    import warp as wp
    e = make_env(aircraft, 700, turbulence="light")
    e.reset()
    W20 = 20.0
    e.set_turb_W20(W20)
    us = []
    a = torch.zeros(1, 4, device="cuda")
    for t in range(700):
        e.step(a)
        lin, _ = e.model.wind_model.turbulence_model.get_gusts_numpy()
        us.append(lin[0, 0])
    sig_u = float(np.std(us[150:]))
    return sig_u / W20, float(e.base_vel)


METHOD = "PPO"      # the one executor; the name the table prints


def controller(plane, ckpt_dir=CKPT_DIR):
    """The study's one executor: the PPO attitude policy at seed 0. A second executor is not an
    option here either -- the shipped checkpoint set carries no SAC+CAPS weights."""
    from falcons.controllers.policies import load_policy
    return load_policy("ppo", "attitude", plane, 0, ckpt_dir=ckpt_dir).act


def _hand_over(ctrl):
    """fly() builds the rollout env itself and asks for the controller only afterwards, so a
    controller is passed as a factory f(env, phi_s, hdot_s, va_s). The disturbance wrappers read
    nothing from the env, so the factory just hands the already-built wrapper back; it is called
    once per rollout, which is exactly the lifetime the delay buffers assume."""
    return lambda env, phi_s, hdot_s, va_s, draw=0: ctrl


def eval_cell(aircraft, method, fn, dist, sev, turb_k=None, va=None, va_scale=None, vz_scale=None):
    """One (disturbance, severity) cell: mean over maneuvers of survival + tracking RMSE."""
    survs, phis, hds, vas = [], [], [], []
    for mv in MANEUVERS:
        kw = {}
        if dist == "nominal":
            ctrl = fn
        elif dist == "action_delay":
            ctrl = ActionDelay(fn, steps_from_ms(ACTION_DELAY_MS[sev]))
        elif dist == "sensor_delay":
            ctrl = SensorDelay(fn, steps_from_ms(SENSOR_DELAY_MS[sev]))
        elif dist == "obs_noise":
            ctrl = ObsNoise(fn, obs_noise_sigma(sev, va_scale, vz_scale),
                            noise_seed(aircraft, sev))
        elif dist == "wind":
            ctrl = fn
            kw = dict(turbulence="light", turb_W20=WIND_IU[sev] * va / turb_k)
        r = fly(aircraft, mv, _hand_over(ctrl), **kw)
        survs.append(r["survival"])
        if r["phi_rmse"] is not None:
            phis.append(r["phi_rmse"]); hds.append(r["hdot_rmse"]); vas.append(r["va_rmse"])
    agg = lambda xs: float(np.mean(xs)) if xs else None
    return dict(aircraft=aircraft, method=method, disturbance=dist, severity=sev,
                survival=float(np.mean(survs)), n_survive=int(sum(survs)), n=len(MANEUVERS),
                phi_rmse=agg(phis), hdot_rmse=agg(hds), va_rmse=agg(vas))


def run(planes, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR):
    """One row per (aircraft, disturbance, severity) for the one executor: the nominal cell, then
    the four families at three severities each, every cell a mean over the four maneuvers."""
    import warp as wp
    wp.init()
    os.makedirs(results_dir, exist_ok=True)
    rows = []
    for ac in planes:
        turb_k, va = calibrate_turb(ac)
        va_scale = vz_scale = max(0.5 * va, 5.0)
        print(f"\n=== {ac}  (Va={va:.1f}, turb k=sigma_u/W20={turb_k:.4f}) ===", flush=True)
        fn = controller(ac, ckpt_dir)
        base = eval_cell(ac, METHOD, fn, "nominal", "-")
        rows.append(base)
        print(f"  {METHOD:9s} nominal            surv {base['survival']:.2f}  "
              f"phi {base['phi_rmse']}", flush=True)
        for dist in ["action_delay", "sensor_delay", "obs_noise", "wind"]:
            for sev in SEVERITIES:
                c = eval_cell(ac, METHOD, fn, dist, sev, turb_k=turb_k, va=va,
                              va_scale=va_scale, vz_scale=vz_scale)
                rows.append(c)
                pr = f"{c['phi_rmse']:.1f}" if c['phi_rmse'] is not None else "crash"
                print(f"  {METHOD:9s} {dist:13s} {sev:7s} surv {c['survival']:.2f}  "
                      f"phi {pr}", flush=True)
    return _write_table(rows, Path(results_dir) / "robustness.csv")


def _write_table(rows, out):
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["aircraft", "method", "disturbance", "severity",
            "survival", "n_survive", "phi_rmse", "hdot_rmse", "va_rmse"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.3f}" if isinstance(r[k], float) and r[k] is not None else r[k])
                        for k in w.fieldnames})
    print(f"wrote {out}")
    return out


# the four-panel per-airframe figure is not here: Task 8 ports it to figures.py.
