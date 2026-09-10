"""Batched GPU attitude+speed executor env: direct-actuator RL tracks local goals
(phi*, hdot*, Va*). No autopilot wrapper -- RL owns all four surfaces. Reusable
low-level skill; a later guidance layer feeds it a stream of goals.
"""
import numpy as np
import torch
import warp as wp

from falcons.sim.warp.aircraft import Aircraft
from falcons.aircraft.params import load_params
from falcons.sim.warp.termination import check_termination_batch
from falcons.aircraft.config import AircraftConfig

TRIM_THROTTLE = 0.45
PI = 3.14159265358979


@wp.func
def _clip5(x: wp.float32) -> wp.float32:
    return wp.clamp(x, -5.0, 5.0)


@wp.kernel
def compute_att_obs(orientation: wp.array(dtype=wp.quatf),
                    angular_vel: wp.array(dtype=wp.vec3f),
                    linear_vel: wp.array(dtype=wp.vec3f),
                    Va: wp.array(dtype=wp.float32),
                    alpha: wp.array(dtype=wp.float32),
                    beta: wp.array(dtype=wp.float32),
                    phi_tgt: wp.array(dtype=wp.float32),
                    hdot_tgt: wp.array(dtype=wp.float32),
                    va_tgt: wp.array(dtype=wp.float32),
                    prev_action: wp.array(dtype=wp.float32, ndim=2),
                    va_scale: wp.float32, vz_scale: wp.float32,
                    obs: wp.array(dtype=wp.float32, ndim=2)):
    tid = wp.tid()
    q = orientation[tid]
    qx = q[0]; qy = q[1]; qz = q[2]; qw = q[3]
    roll = wp.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch = wp.asin(wp.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
    w = angular_vel[tid]
    vw = wp.quat_rotate(q, linear_vel[tid])
    vz = vw[2]
    obs[tid, 0] = _clip5((roll - phi_tgt[tid]) / (PI / 4.0))
    obs[tid, 1] = _clip5(((0.0 - vz) - hdot_tgt[tid]) / vz_scale)   # climb_rate = -vz
    obs[tid, 2] = _clip5((Va[tid] - va_tgt[tid]) / va_scale)
    obs[tid, 3] = _clip5(roll / (PI / 4.0))
    obs[tid, 4] = _clip5(pitch / (PI / 4.0))
    obs[tid, 5] = _clip5(w[0] / 2.0)
    obs[tid, 6] = _clip5(w[1] / 2.0)
    obs[tid, 7] = _clip5(w[2] / 2.0)
    obs[tid, 8] = _clip5(alpha[tid] / 0.349)
    obs[tid, 9] = _clip5(beta[tid] / 0.349)
    obs[tid, 10] = _clip5(vz / vz_scale)
    obs[tid, 11] = _clip5(prev_action[tid, 0])
    obs[tid, 12] = _clip5(prev_action[tid, 1])
    obs[tid, 13] = _clip5(prev_action[tid, 2])
    obs[tid, 14] = _clip5(prev_action[tid, 3])


@wp.kernel
def compute_att_reward(orientation: wp.array(dtype=wp.quatf),
                       linear_vel: wp.array(dtype=wp.vec3f),
                       Va: wp.array(dtype=wp.float32),
                       action: wp.array(dtype=wp.float32, ndim=2),
                       prev_action: wp.array(dtype=wp.float32, ndim=2),
                       phi_tgt: wp.array(dtype=wp.float32),
                       hdot_tgt: wp.array(dtype=wp.float32),
                       va_tgt: wp.array(dtype=wp.float32),
                       va_safety: wp.float32,
                       w_phi: wp.float32, sig_phi: wp.float32,
                       w_hdot: wp.float32, sig_hdot: wp.float32,
                       w_va: wp.float32, sig_va: wp.float32,
                       w_energy: wp.float32, w_sm: wp.float32,
                       reward: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    q = orientation[tid]
    qx = q[0]; qy = q[1]; qz = q[2]; qw = q[3]
    roll = wp.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    vw = wp.quat_rotate(q, linear_vel[tid])
    vz = vw[2]
    pe = (roll - phi_tgt[tid]) / sig_phi
    he = ((0.0 - vz) - hdot_tgt[tid]) / sig_hdot   # climb_rate = -vz
    ve = (Va[tid] - va_tgt[tid]) / sig_va
    r = w_phi * wp.exp(-0.5 * pe * pe)
    r += w_hdot * wp.exp(-0.5 * he * he)
    r += w_va * wp.exp(-0.5 * ve * ve)
    deficit = (va_safety - Va[tid]) / va_safety
    if deficit > 0.0:
        r -= w_energy * deficit * deficit
    d0 = action[tid, 0] - prev_action[tid, 0]
    d1 = action[tid, 1] - prev_action[tid, 1]
    d2 = action[tid, 2] - prev_action[tid, 2]
    d3 = action[tid, 3] - prev_action[tid, 3]
    r -= w_sm * (d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3)
    reward[tid] = r


@wp.kernel
def _select_segment(step: wp.array(dtype=wp.int32), horizon: wp.int32,
                    phi3: wp.array(dtype=wp.float32, ndim=2),
                    hdot3: wp.array(dtype=wp.float32, ndim=2),
                    va3: wp.array(dtype=wp.float32, ndim=2),
                    phi_tgt: wp.array(dtype=wp.float32),
                    hdot_tgt: wp.array(dtype=wp.float32),
                    va_tgt: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    seg = wp.min(2, (step[tid] * 3) // horizon)
    phi_tgt[tid] = phi3[tid, seg]
    hdot_tgt[tid] = hdot3[tid, seg]
    va_tgt[tid] = va3[tid, seg]


@wp.kernel
def _att_bank_term(orientation: wp.array(dtype=wp.quatf), bank_limit: wp.float32,
                   terminated: wp.array(dtype=wp.int32), reason: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    q = orientation[tid]
    roll = wp.atan2(2.0 * (q[3] * q[0] + q[1] * q[2]), 1.0 - 2.0 * (q[0] * q[0] + q[1] * q[1]))
    if reason[tid] == 0 and wp.abs(roll) > bank_limit:
        terminated[tid] = 1
        reason[tid] = 3


@wp.kernel
def _att_nan_guard(Va: wp.array(dtype=wp.float32), position: wp.array(dtype=wp.vec3f),
                   terminated: wp.array(dtype=wp.int32), reason: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    v = Va[tid]; p = position[tid][2]
    if not (v == v) or not (p == p):
        terminated[tid] = 1
        reason[tid] = 1


@wp.kernel
def _att_inc_truncate(step: wp.array(dtype=wp.int32), horizon: wp.int32,
                      terminated: wp.array(dtype=wp.int32),
                      truncated: wp.array(dtype=wp.int32), done: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    step[tid] = step[tid] + 1
    tr = wp.int32(0)
    if step[tid] >= horizon:
        tr = 1
    truncated[tid] = tr
    d = wp.int32(0)
    if terminated[tid] == 1 or tr == 1:
        d = 1
    done[tid] = d


@wp.kernel
def _att_apply_terminal(reason: wp.array(dtype=wp.int32), crash_pen: wp.float32,
                        stall_pen: wp.float32, bank_pen: wp.float32,
                        reward: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    if reason[tid] == 1:
        reward[tid] = reward[tid] - crash_pen
    elif reason[tid] == 2:
        reward[tid] = reward[tid] - stall_pen
    elif reason[tid] == 3:
        reward[tid] = reward[tid] - bank_pen


@wp.kernel
def _att_masked_reset(done: wp.array(dtype=wp.int32), seed: wp.int32,
                      spawn_alt: wp.float32, base_vel: wp.float32, trim_thr: wp.float32,
                      phi_max: wp.float32, hdot_max: wp.float32, va_lo: wp.float32, va_hi: wp.float32,
                      position: wp.array(dtype=wp.vec3f), linear_vel: wp.array(dtype=wp.vec3f),
                      angular_vel: wp.array(dtype=wp.vec3f), orientation: wp.array(dtype=wp.quatf),
                      elevator: wp.array(dtype=wp.float32), aileron: wp.array(dtype=wp.float32),
                      rudder: wp.array(dtype=wp.float32),
                      throttle_l: wp.array(dtype=wp.float32), throttle_r: wp.array(dtype=wp.float32),
                      elevator_dot: wp.array(dtype=wp.float32), aileron_dot: wp.array(dtype=wp.float32),
                      rudder_dot: wp.array(dtype=wp.float32),
                      step: wp.array(dtype=wp.int32), prev: wp.array(dtype=wp.float32, ndim=2),
                      phi3: wp.array(dtype=wp.float32, ndim=2), hdot3: wp.array(dtype=wp.float32, ndim=2),
                      va3: wp.array(dtype=wp.float32, ndim=2)):
    tid = wp.tid()
    if done[tid] == 0:
        return
    position[tid] = wp.vec3f(0.0, 0.0, -spawn_alt)
    linear_vel[tid] = wp.vec3f(base_vel, 0.0, 0.0)
    angular_vel[tid] = wp.vec3f(0.0, 0.0, 0.0)
    orientation[tid] = wp.quatf(0.0, 0.0, 0.0, 1.0)
    elevator[tid] = 0.0; aileron[tid] = 0.0; rudder[tid] = 0.0
    throttle_l[tid] = trim_thr; throttle_r[tid] = trim_thr
    elevator_dot[tid] = 0.0; aileron_dot[tid] = 0.0; rudder_dot[tid] = 0.0
    step[tid] = 0
    prev[tid, 0] = 0.0; prev[tid, 1] = 0.0; prev[tid, 2] = 0.0; prev[tid, 3] = 0.0
    st = wp.rand_init(seed, tid)
    for k in range(3):
        phi3[tid, k] = phi_max * (2.0 * wp.randf(st) - 1.0)
        hdot3[tid, k] = hdot_max * (2.0 * wp.randf(st) - 1.0)
        va3[tid, k] = va_lo + (va_hi - va_lo) * wp.randf(st)


class AttitudeEnv:
    """15-D obs, 4-D action [elevator, aileron, rudder, throttle]. RL tracks piecewise
    (phi*, hdot*, Va*). Direct actuator, no inner-loop wrapper."""

    def __init__(self, num_envs, device="cuda", cfg=None):
        cfg = cfg or {}
        self.n = num_envs
        self.device = device
        self.horizon = cfg.get("horizon", 2000)
        self.aircraft = cfg.get("aircraft", "Airship_V7")
        self.dt = 0.01
        self.spawn_alt = cfg.get("spawn_alt", 60.0)

        raw = AircraftConfig(self.aircraft).load()
        self.alpha_soft_deg = raw["aero_params"]["alpha_soft_deg"]
        self.alpha_limit = float(np.radians(raw["aero_params"]["alpha_max_deg"]))
        self.base_vel = cfg.get("base_vel", float(raw["default_initial_state"]["linear_vel"][0]))
        self.va_min = cfg.get("va_min", 0.6 * self.base_vel)
        self.va_scale = max(0.5 * self.base_vel, 5.0)
        self.vz_scale = max(0.5 * self.base_vel, 5.0)
        self.phi_max = float(np.radians(cfg.get("phi_max_deg", 45.0)))
        self.hdot_max = cfg.get("hdot_max", 4.0)   # m/s
        self.phi_max_full = self.phi_max
        self.hdot_max_full = self.hdot_max
        self.phi_min = float(np.radians(cfg.get("phi_min_deg", 5.0)))
        self.hdot_min = cfg.get("hdot_min", 0.5)   # m/s
        self.va_lo = cfg.get("va_lo", 0.85) * self.base_vel
        self.va_hi = cfg.get("va_hi", 1.15) * self.base_vel

        self.w_phi = cfg.get("w_phi", 2.0); self.sig_phi = cfg.get("sig_phi", 0.15)
        self.w_hdot = cfg.get("w_hdot", 2.0); self.sig_hdot = cfg.get("sig_hdot", 1.0)
        self.w_va = cfg.get("w_va", 1.0); self.sig_va = cfg.get("sig_va", 3.0)
        self.w_energy = cfg.get("w_energy", 2.0); self.w_sm = cfg.get("smooth_weight", 0.1)
        self.va_safety = cfg.get("va_safety", 0.75 * self.base_vel)
        self.bank_limit = float(np.radians(cfg.get("bank_limit_deg", 70.0)))
        self.crash_pen = cfg.get("crash_penalty", 40.0)
        self.stall_pen = cfg.get("stall_penalty", 30.0)
        self.bank_pen = cfg.get("bank_penalty", 30.0)

        co = load_params(self.aircraft)
        AP = co["aero_params"].as_warp_struct()
        VP = co["vehicle_params"]
        VP.WP = VP.WP.as_warp_struct(); VP.J = VP.J.as_warp_struct(); VP.PP = VP.PP.as_warp_struct()
        if hasattr(VP, "sensor_system"):
            VP.sensor_system = None
        EP = co["environment_params"]
        # Opt-in Dryden turbulence for robustness eval (mirrors envs/altitude.py). Off by default:
        # training never sees gusts. cfg["turbulence"] = "light"|"moderate"|"severe" seeds the base
        # W20; set_turb_W20/set_turb_scale then dial severity at eval time.
        self.turbulence = cfg.get("turbulence", None)
        if self.turbulence:
            EP.turbulence = {"enable": True, "model": "dryden", "intensity": self.turbulence}
        config = {"AP": AP, "VP": VP, "CL": co["control_limits"], "EP": EP}
        self.model = Aircraft(num_envs, device, config, save_history=False)
        self.model.solver_type = 1
        tm = getattr(self.model.wind_model, "turbulence_model", None)
        self._turb_W20_full = getattr(tm, "W_20_ms", 0.0)

        d, n = device, num_envs
        self._step = wp.zeros(n, dtype=wp.int32, device=d)
        self._prev = wp.zeros((n, 4), dtype=wp.float32, device=d)
        self._act = wp.zeros((n, 4), dtype=wp.float32, device=d)
        self._obs = wp.zeros((n, 15), dtype=wp.float32, device=d)
        self._rew = wp.zeros(n, dtype=wp.float32, device=d)
        self._term = wp.zeros(n, dtype=wp.int32, device=d)
        self._trunc = wp.zeros(n, dtype=wp.int32, device=d)
        self._done = wp.zeros(n, dtype=wp.int32, device=d)
        self._reason = wp.zeros(n, dtype=wp.int32, device=d)
        self._phi_tgt = wp.zeros(n, dtype=wp.float32, device=d)
        self._hdot_tgt = wp.zeros(n, dtype=wp.float32, device=d)
        self._va_tgt = wp.zeros(n, dtype=wp.float32, device=d)
        self._phi3 = wp.zeros((n, 3), dtype=wp.float32, device=d)
        self._hdot3 = wp.zeros((n, 3), dtype=wp.float32, device=d)
        self._va3 = wp.zeros((n, 3), dtype=wp.float32, device=d)
        self._seed = 0
        self.num_obs = 15
        self.num_act = 4

    def set_curriculum(self, frac):
        # grow the attitude-target magnitude from a gentle floor to the full range over the first
        # 60% of training: full +-45deg bank / +-4m/s climb-rate from step 0 cold-starts into loss-of-
        # control/stall on hard slews; small targets first give a survivable gradient to climb.
        f = min(1.0, max(0.0, frac))
        self.phi_max = self.phi_min + f * (self.phi_max_full - self.phi_min)
        self.hdot_max = self.hdot_min + f * (self.hdot_max_full - self.hdot_min)

    def set_turb_scale(self, frac):
        """Scale Dryden gust intensity to `frac` of the configured level (robustness eval)."""
        tm = getattr(self.model.wind_model, "turbulence_model", None)
        if tm is not None:
            tm.W_20_ms = self._turb_W20_full * float(frac)

    def set_turb_W20(self, w20_ms):
        """Set the Dryden W20 (wind speed at 20 ft, m/s) directly. Lets the harness dial gusts to
        a target turbulence intensity Iu = sigma_u/Va per airframe (sigma_u ~ 0.154*W20 at ~60 m)."""
        tm = getattr(self.model.wind_model, "turbulence_model", None)
        if tm is not None:
            tm.W_20_ms = float(w20_ms)

    def _sample_targets(self, n):
        phi = (self.phi_max * (2.0 * np.random.rand(n, 3) - 1.0)).astype(np.float32)
        hdot = (self.hdot_max * (2.0 * np.random.rand(n, 3) - 1.0)).astype(np.float32)
        va = np.random.uniform(self.va_lo, self.va_hi, (n, 3)).astype(np.float32)
        return phi, hdot, va

    def _select_launch(self):
        wp.launch(_select_segment, dim=self.n, inputs=[
            self._step, wp.int32(self.horizon), self._phi3, self._hdot3, self._va3,
            self._phi_tgt, self._hdot_tgt, self._va_tgt], device=self.device)

    def _obs_launch(self):
        s = self.model._state
        wp.launch(compute_att_obs, dim=self.n, inputs=[
            s["orientation"], s["angular_vel"], s["linear_vel"],
            self.model._Va, self.model._alpha, self.model._beta,
            self._phi_tgt, self._hdot_tgt, self._va_tgt,
            self._prev, self.va_scale, self.vz_scale, self._obs], device=self.device)

    def _reward_launch(self):
        wp.launch(compute_att_reward, dim=self.n, inputs=[
            self.model._state["orientation"], self.model._state["linear_vel"],
            self.model._Va, self._act, self._prev,
            self._phi_tgt, self._hdot_tgt, self._va_tgt, self.va_safety,
            self.w_phi, self.sig_phi, self.w_hdot, self.sig_hdot,
            self.w_va, self.sig_va, self.w_energy, self.w_sm, self._rew], device=self.device)

    def reset(self):
        n = self.n
        pos = np.zeros((n, 3), dtype=np.float32); pos[:, 2] = -self.spawn_alt
        vel = np.zeros((n, 3), dtype=np.float32); vel[:, 0] = self.base_vel
        ori = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1))
        self.model.reset(
            {"position": pos, "linear_vel": vel,
             "angular_vel": np.zeros((n, 3), dtype=np.float32), "orientation": ori},
            {"elevator": 0.0, "aileron": 0.0, "rudder": 0.0,
             "throttle_left": TRIM_THROTTLE, "throttle_right": TRIM_THROTTLE,
             "elevator_dot": 0.0, "aileron_dot": 0.0, "rudder_dot": 0.0})
        self._step.zero_(); self._prev.zero_()
        phi3, hdot3, va3 = self._sample_targets(n)
        wp.copy(self._phi3, wp.array(phi3, device=self.device))
        wp.copy(self._hdot3, wp.array(hdot3, device=self.device))
        wp.copy(self._va3, wp.array(va3, device=self.device))
        self._select_launch()
        self._obs_launch()
        return wp.to_torch(self._obs)

    def step(self, action):
        act = action.contiguous().to(torch.float32)
        wp.copy(self._act, wp.from_torch(act, dtype=wp.float32))
        anp = act.detach().cpu().numpy()
        self.model.step({"elevator": anp[:, 0], "aileron": anp[:, 1], "rudder": anp[:, 2],
                         "throttle_left": anp[:, 3], "throttle_right": anp[:, 3]})
        s = self.model._state
        self._reward_launch()

        self._term.zero_(); self._reason.zero_()
        wp.launch(check_termination_batch, dim=self.n, inputs=[
            s["position"], self.model._alpha, self.model._Va,
            self.alpha_limit, self.va_min, self._term, self._reason],
            device=self.device)
        wp.launch(_att_bank_term, dim=self.n, inputs=[
            s["orientation"], self.bank_limit, self._term, self._reason], device=self.device)
        wp.launch(_att_nan_guard, dim=self.n, inputs=[
            self.model._Va, s["position"], self._term, self._reason], device=self.device)
        wp.launch(_att_inc_truncate, dim=self.n, inputs=[
            self._step, wp.int32(self.horizon), self._term, self._trunc, self._done], device=self.device)
        wp.launch(_att_apply_terminal, dim=self.n, inputs=[
            self._reason, self.crash_pen, self.stall_pen, self.bank_pen, self._rew], device=self.device)

        reward = torch.nan_to_num(wp.to_torch(self._rew)).clone()
        done = wp.to_torch(self._done).clone()
        truncated = wp.to_torch(self._trunc).clone()

        self._select_launch(); self._obs_launch()
        terminal_obs = torch.nan_to_num(wp.to_torch(self._obs)).clone()
        wp.copy(self._prev, self._act)

        self._seed += 1
        a = self.model._actuator_states
        wp.launch(_att_masked_reset, dim=self.n, inputs=[
            self._done, self._seed, self.spawn_alt, self.base_vel, TRIM_THROTTLE,
            self.phi_max, self.hdot_max, self.va_lo, self.va_hi,
            s["position"], s["linear_vel"], s["angular_vel"], s["orientation"],
            a["elevator"], a["aileron"], a["rudder"], a["throttle_left"], a["throttle_right"],
            a["elevator_dot"], a["aileron_dot"], a["rudder_dot"],
            self._step, self._prev, self._phi3, self._hdot3, self._va3], device=self.device)

        self._select_launch(); self._obs_launch()
        obs = torch.nan_to_num(wp.to_torch(self._obs))
        info = {"terminal_obs": terminal_obs, "truncated": truncated,
                "reason": wp.to_torch(self._reason).clone()}
        return obs, reward, done.bool(), info
