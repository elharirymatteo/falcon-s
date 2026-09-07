"""Batched GPU altitude-keeping env: Aircraft + warp kernels + masked per-env
auto-reset. Obs/actions/rewards stay on GPU (torch views via wp.to_torch).

Target modes: "dynamic" (random target in [lo,hi] per episode) or "fixed". With ref_rate > 0
the tracked sub-target ramps toward the true target at ref_rate m/s (rate-limited reference),
keeping the altitude error small so the policy never over-pitches into stall.

Each kernel is parity-tested vs the numpy sim in src/utils/tests/.
"""
import numpy as np
import torch
import warp as wp

from falcons.sim.warp.aircraft import Aircraft
from falcons.aircraft.params import load_params
from falcons.sim.warp.termination import check_termination_batch
from falcons.sim.warp.altitude_obs import compute_obs
from falcons.sim.warp.altitude_reward import compute_reward
from falcons.aircraft.config import AircraftConfig

TRIM_ELEV = 0.0
TRIM_THROTTLE = 0.45


@wp.kernel
def _apply_terminal(reason: wp.array(dtype=wp.int32),
                    truncated: wp.array(dtype=wp.int32),
                    position: wp.array(dtype=wp.vec3f),
                    target_altitude: wp.array(dtype=wp.float32),
                    zone: wp.float32,
                    crash_pen: wp.float32,
                    stall_pen: wp.float32,
                    horizon_bonus: wp.float32,
                    reward: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    if reason[tid] == 1:
        reward[tid] = reward[tid] - crash_pen
    elif reason[tid] == 2:
        reward[tid] = reward[tid] - stall_pen
    elif truncated[tid] == 1:
        h_error = -position[tid][2] - target_altitude[tid]
        if wp.abs(h_error) < zone:
            reward[tid] = reward[tid] + horizon_bonus


@wp.kernel
def _masked_reset(done: wp.array(dtype=wp.int32),
                  seed: wp.int32,
                  target: wp.float32,
                  spawn: wp.float32,
                  trim_throttle: wp.float32,
                  position: wp.array(dtype=wp.vec3f),
                  linear_vel: wp.array(dtype=wp.vec3f),
                  angular_vel: wp.array(dtype=wp.vec3f),
                  orientation: wp.array(dtype=wp.quatf),
                  elevator_state: wp.array(dtype=wp.float32),
                  aileron_state: wp.array(dtype=wp.float32),
                  rudder_state: wp.array(dtype=wp.float32),
                  throttle_left_state: wp.array(dtype=wp.float32),
                  throttle_right_state: wp.array(dtype=wp.float32),
                  elevator_dot: wp.array(dtype=wp.float32),
                  aileron_dot: wp.array(dtype=wp.float32),
                  rudder_dot: wp.array(dtype=wp.float32),
                  step_count: wp.array(dtype=wp.int32),
                  prev_action: wp.array(dtype=wp.float32, ndim=2),
                  target_altitude: wp.array(dtype=wp.float32),
                  integral_error: wp.array(dtype=wp.float32),
                  mode: wp.int32,
                  tgt_lo: wp.float32,
                  tgt_hi: wp.float32,
                  base_vel: wp.float32,
                  ramp_on: wp.int32,
                  true_target: wp.array(dtype=wp.float32),
                  alt_min: wp.float32,
                  spawn_above: wp.int32,
                  va_mode: wp.int32,
                  va_lo: wp.float32,
                  va_hi: wp.float32,
                  va_tgt: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    if done[tid] == 1:
        st = wp.rand_init(seed, tid)
        if mode == 1:  # dynamic: random constant target in [lo, hi]
            tgt = tgt_lo + (tgt_hi - tgt_lo) * wp.randf(wp.rand_init(seed + 104729, tid))
        else:          # fixed
            tgt = target
        if va_mode == 1:  # A3 (Va,h): random airspeed setpoint per episode
            va_tgt[tid] = va_lo + (va_hi - va_lo) * wp.randf(wp.rand_init(seed + 15485863, tid))
        if spawn_above == 1:
            alt = wp.max(tgt + spawn * wp.randf(st), alt_min)  # descent-only: [tgt, tgt+spawn], never below target
        else:
            alt = wp.max(tgt + spawn * (2.0 * wp.randf(st) - 1.0), alt_min)  # floor task: spawn above the soft zone
        position[tid] = wp.vec3f(0.0, 0.0, -alt)
        true_target[tid] = tgt
        if ramp_on == 1:
            tgt = alt  # sub-target starts at the aircraft; ramps to true_target
        linear_vel[tid] = wp.vec3f(base_vel, 0.0, 0.0)
        angular_vel[tid] = wp.vec3f(0.0, 0.0, 0.0)
        orientation[tid] = wp.quatf(0.0, 0.0, 0.0, 1.0)
        elevator_state[tid] = 0.0
        aileron_state[tid] = 0.0
        rudder_state[tid] = 0.0
        throttle_left_state[tid] = trim_throttle
        throttle_right_state[tid] = trim_throttle
        elevator_dot[tid] = 0.0
        aileron_dot[tid] = 0.0
        rudder_dot[tid] = 0.0
        step_count[tid] = 0
        prev_action[tid, 0] = 0.0
        prev_action[tid, 1] = 0.0
        target_altitude[tid] = tgt
        integral_error[tid] = 0.0


@wp.kernel
def _floor_term(position: wp.array(dtype=wp.vec3f),
                h_floor: wp.float32,
                terminated: wp.array(dtype=wp.int32),
                reason: wp.array(dtype=wp.int32)):
    # A5 floor-constrained task: altitude floor breach terminates (CaT hard constraint).
    # reason 3 = floor (crash/stall from check_termination_batch keep priority).
    tid = wp.tid()
    if reason[tid] == 0 and -position[tid][2] < h_floor:
        terminated[tid] = 1
        reason[tid] = 3


@wp.kernel
def _floor_shaped(position: wp.array(dtype=wp.vec3f),
                  reason: wp.array(dtype=wp.int32),
                  h_floor: wp.float32,
                  margin: wp.float32,
                  w_floor: wp.float32,
                  floor_pen: wp.float32,
                  reward: wp.array(dtype=wp.float32)):
    # shaped-reward baseline: linear barrier inside the soft margin + terminal penalty on
    # breach (the hand-tuned weights CaT replaces; both 0 in cat mode).
    tid = wp.tid()
    depth = (h_floor + margin - (-position[tid][2])) / margin
    reward[tid] = reward[tid] - w_floor * wp.clamp(depth, 0.0, 2.0)
    if reason[tid] == 3:
        reward[tid] = reward[tid] - floor_pen


@wp.kernel
def _floor_obs(position: wp.array(dtype=wp.vec3f),
               h_floor: wp.float32,
               margin: wp.float32,
               obs: wp.array(dtype=wp.float32, ndim=2)):
    # 12th obs channel: clearance above the soft-margin boundary (else targets 12m vs 30m
    # look identical through h_err and the policy cannot know how close the floor is).
    tid = wp.tid()
    obs[tid, 11] = wp.clamp((-position[tid][2] - h_floor - margin) / 10.0, -3.0, 3.0)


@wp.kernel
def _inc_and_truncate(step_count: wp.array(dtype=wp.int32),
                      horizon: wp.int32,
                      terminated: wp.array(dtype=wp.int32),
                      truncated: wp.array(dtype=wp.int32),
                      done: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    step_count[tid] = step_count[tid] + 1
    if step_count[tid] >= horizon:
        truncated[tid] = 1
    else:
        truncated[tid] = 0
    if terminated[tid] == 1 or truncated[tid] == 1:
        done[tid] = 1
    else:
        done[tid] = 0


@wp.kernel
def _set_prev(action: wp.array(dtype=wp.float32, ndim=2),
              prev_action: wp.array(dtype=wp.float32, ndim=2)):
    tid = wp.tid()
    prev_action[tid, 0] = action[tid, 0]
    prev_action[tid, 1] = action[tid, 1]


@wp.kernel
def _ramp_reference(true_target: wp.array(dtype=wp.float32),
                    dt: wp.float32,
                    rate: wp.float32,
                    target_altitude: wp.array(dtype=wp.float32),
                    ref_rate_out: wp.array(dtype=wp.float32)):
    # sub-target ramps toward the true target at <= rate m/s -> error stays small -> no over-pitch.
    tid = wp.tid()
    diff = true_target[tid] - target_altitude[tid]
    step = wp.clamp(diff, -rate * dt, rate * dt)
    target_altitude[tid] = target_altitude[tid] + step
    ref_rate_out[tid] = step / dt


@wp.kernel
def _va_from_vel(vel: wp.array(dtype=wp.vec3f), va_out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    va_out[tid] = wp.length(vel[tid])


@wp.kernel
def _kin_alpha(vel: wp.array(dtype=wp.vec3f), alpha_out: wp.array(dtype=wp.float32)):
    # deploy: EKF2 has no AoA sensor -> kinematic proxy atan2(w_body, u_body) (= aero alpha in still
    # air). Used for the obs only; the sim keeps true aero alpha for termination.
    tid = wp.tid()
    alpha_out[tid] = wp.atan2(vel[tid][2], vel[tid][0])


@wp.kernel
def _inner_pitch(action: wp.array(dtype=wp.float32, ndim=2),    # [pitch_sp_norm, throttle_norm]
                 orientation: wp.array(dtype=wp.quatf),
                 angular_vel: wp.array(dtype=wp.vec3f),
                 pitch_max: wp.float32, kp: wp.float32, kd: wp.float32,
                 surf: wp.array(dtype=wp.float32, ndim=2)):     # out [elevator, throttle]
    # PX4/SAS-surrogate inner loop: RL commands a pitch setpoint; a PD on pitch+q owns the elevator
    # so the unstable plant is stabilized by the inner loop (RL does guidance only).
    tid = wp.tid()
    q = orientation[tid]
    pitch = wp.asin(wp.clamp(2.0 * (q[3] * q[1] - q[2] * q[0]), -1.0, 1.0))
    pitch_sp = pitch_max * wp.clamp(action[tid, 0], -1.0, 1.0)
    # +elevator pitches nose DOWN on this airframe, so the restoring/damping signs are inverted
    # vs a standard +pitch convention: elev = Kp*(pitch - pitch_sp) + Kd*q.
    elev = kp * (pitch - pitch_sp) + kd * angular_vel[tid][1]
    surf[tid, 0] = wp.clamp(elev, -1.0, 1.0)
    surf[tid, 1] = action[tid, 1]


@wp.kernel
def _nan_guard(position: wp.array(dtype=wp.vec3f), Va: wp.array(dtype=wp.float32),
               terminated: wp.array(dtype=wp.int32), reason: wp.array(dtype=wp.int32)):
    # a NaN state (e.g. a gust pushing past the poly-aero valid range) won't trip the alpha
    # check (NaN > limit is False); force-crash it so masked_reset recovers the env.
    tid = wp.tid()
    p = position[tid][2]
    v = Va[tid]
    if not (p == p) or not (v == v):
        terminated[tid] = 1
        reason[tid] = 1


@wp.kernel
def _update_integral(position: wp.array(dtype=wp.vec3f),
                     target_altitude: wp.array(dtype=wp.float32),
                     dt: wp.float32,
                     clamp: wp.float32,
                     integral_error: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    v = integral_error[tid] + (-position[tid][2] - target_altitude[tid]) * dt
    integral_error[tid] = wp.clamp(v, -clamp, clamp)


class AltitudeEnv:
    """11-dim obs: [h_err, pitch_rate, Va_err, pitch, vz, alpha, elev, elev_rate, thr,
    integral(h_err), ref_rate]; 2-d action [elevator, throttle].
    Floor task (h_floor set): +12th obs channel = clearance above floor soft margin."""

    def __init__(self, num_envs, device="cuda", cfg=None):
        cfg = cfg or {}
        self.n = num_envs
        self.device = device
        self.horizon = cfg.get("horizon", 2000)
        self.target = cfg.get("target_altitude", 50.0)
        self.spawn = cfg.get("spawning_distance", 1.0)
        self.in_ground_effect = bool(cfg.get("in_ground_effect", True))   # WIG ablation: GE lift/drag on/off
        self.spawn_above = bool(cfg.get("spawn_above", False))   # flare: descent-only spawn, never below target
        w = cfg.get("reward_weights", [5.0, 2.0, 0.2, 0.1])
        self.w_alt, self.w_va, self.w_pr, self.w_sm = w
        self.sigma = cfg.get("altitude_sigma", 10.0)
        self.zone = cfg.get("target_altitude_zone", 3.0)
        self.va_band = cfg.get("va_band", 4.0)
        self.recovery = 8.0
        self.ov_w = cfg.get("overshoot_weight", 2.0)
        self.ov_thr = cfg.get("overshoot_threshold", 5.0)
        self.crash_pen = cfg.get("crash_penalty", 40.0)
        self.stall_pen = cfg.get("stall_penalty", 30.0)
        self.horizon_bonus = 10.0
        # stall-safety shaping (lit-derived robustness)
        self.alpha_safety_w = cfg.get("alpha_safety_weight", 1.0)
        self.alpha_safety_start = cfg.get("alpha_safety_start", 0.7)
        self.va_safety_w = cfg.get("va_safety_weight", 1.0)
        self.va_safety_start = cfg.get("va_safety_start", 15.0)
        self.va_stall = 10.0
        self.g = 9.81
        self.energy_w = cfg.get("energy_weight", 3.0)
        self.energy_sigma = cfg.get("energy_sigma", 5.0)
        self.damp_w = cfg.get("damp_weight", 0.0)
        self.damp_zone = cfg.get("damp_zone", 10.0)
        self.pr_cap = cfg.get("pr_cap", 1.0)
        self.effort_w = cfg.get("effort_weight", 0.0)   # ungated action-rate penalty (damp transient oscillation); 0 = off
        self.ref_rate = cfg.get("ref_rate", 0.0)        # m/s; >0 = rate-limited reference
        # disturbance eval (benchmark.robustness): noisy/delayed sensors + Dryden turbulence
        self.enable_sensors = bool(cfg.get("enable_sensors", False))   # route obs through sensors
        self.turbulence = cfg.get("turbulence", None)                  # None | "light" | "moderate" ...
        self.sensor_delay = cfg.get("sensor_delay", 0.0)               # s (e.g. 0.02 = 20 ms)
        # deploy-ready (PX4): RL outputs a pitch setpoint, a PD inner loop owns the elevator (the
        # stabilizing inner loop our turbulence analysis showed is needed); obs uses EKF2-realizable
        # states (sensored + kinematic-alpha proxy) under sim-to-real DR.
        self.deploy = bool(cfg.get("deploy", False))
        self.kin_alpha = bool(cfg.get("kin_alpha", False))   # EKF2 has no AoA -> kinematic proxy obs
        self.obs_noise = float(cfg.get("obs_noise", 0.0))    # train-time obs perturbation (deploy robustness)
        # transfer obs: zero these channels so the policy learns to track without them. Used to drop
        # the proxy/servo-modeled channels (5 alpha, 6 elevator-state, 7 elev-rate, 8 throttle) so the
        # deployment obs is EKF2-direct only -> no servo reconstruction -> no climb-stall.
        self._mask = cfg.get("obs_mask", [])
        self.pitch_max = float(np.radians(cfg.get("pitch_sp_max_deg", 20.0)))
        self.kp_pitch = cfg.get("inner_kp", 4.0)
        self.kd_pitch = cfg.get("inner_kd", 1.0)
        if self.deploy:
            self.enable_sensors = bool(cfg.get("deploy_sensors", True)) # EKF2 sensor routing (stage 2)
            self.sensor_delay = cfg.get("sensor_delay", 0.04) if self.enable_sensors else 0.0
            # NOTE: no turbulence here -- the pitch-PD inner loop cannot stabilize this open-loop-
            # unstable airframe under gusts (that needs full-state/thrust LQR; see disturbance notes).
            # Deploy targets sim-to-real of the calm-air case: sensor noise + latency + the portable
            # setpoint architecture.
        self.dt = 0.01
        self.i_clamp = cfg.get("i_clamp", 25.0)   # integral-error clamp; lower = less windup (A0S limit-cycle fix)
        self.mode = cfg.get("target_mode", "fixed")      # "fixed" | "dynamic"
        self.mode_id = {"fixed": 0, "dynamic": 1}[self.mode]
        self.tgt_lo = cfg.get("target_lo", 30.0)
        self.tgt_hi = cfg.get("target_hi", 90.0)
        self.va_min = 10.0
        self._seed = 0
        # CaT (Constraints-as-Terminations, Chane-Sane IROS'24): expose per-step constraint
        # violations c_i(s,a) in info["constraints"] (positive = violated). Safety/style shaping
        # moves out of the reward (weights -> 0 in the cat cfg) and into these constraints;
        # existing crash/stall terminations act as the hard constraints (delta = 1).
        self.cat = bool(cfg.get("cat", False))
        self.cat_alpha_onset = cfg.get("cat_alpha_onset", 0.7)      # fraction of stall alpha
        self.cat_pr_lim = cfg.get("cat_pr_lim", 2.0)                # |pitch rate| cap [rad/s]
        self.cat_p_scale = cfg.get("cat_p_scale", None)             # per-constraint severity tiers (None = uniform)
        self.success_va = bool(cfg.get("success_va", False))        # SCoCaT: success requires Va band too
        # A5 floor-constrained task: hard altitude floor (terminates, reason 3) + soft margin
        # above it. Shaped baseline uses floor_weight barrier + floor_penalty; cat mode zeroes
        # both and exposes the margin as a 5th constraint instead.
        self.h_floor = cfg.get("h_floor", None)                     # None = floor off
        self.floor_margin = cfg.get("floor_margin", 3.0)
        self.floor_w = cfg.get("floor_weight", 0.0)
        self.floor_pen = cfg.get("floor_penalty", 40.0)
        # Flare task: sink-rate constraint below sink_gate altitude (6th constraint).
        self.sink_max = cfg.get("sink_max", None)                   # None = sink constraint off
        self.sink_gate = cfg.get("sink_gate", 5.0)
        self.num_constraints = (5 if self.h_floor is not None else 4) + (1 if self.sink_max is not None else 0)

        # per-aircraft params (cross-plane generalization)
        self.aircraft = cfg.get("aircraft", "Airship_V7")
        raw = AircraftConfig(self.aircraft).load()
        self.stall_deg = raw["aero_params"]["stall_angle_deg"]
        self.alpha_limit = float(np.radians(1.5 * self.stall_deg))
        self.wing_cg_z = float(raw["vehicle_params"]["wing"]["cg_offset_vector"][2])
        self.base_vel = cfg.get("base_vel", float(raw["default_initial_state"]["linear_vel"][0]))
        self.target_va = cfg.get("target_airspeed", self.base_vel)
        # airspeed thresholds scale with trim: the V7-tuned 10/15 break at A0S's 13 m/s trim
        # (va_safety_start=15 sits ABOVE trim -> the airspeed reward penalizes constantly ->
        # the policy fights to speed up -> pitch oscillation -> alpha stall).
        self.va_min = cfg.get("va_min", 0.6 * self.base_vel)            # hard stall airspeed
        self.va_stall = self.va_min
        self.va_safety_start = cfg.get("va_safety_start", 0.75 * self.base_vel)
        # per-plane velocity obs scales (so vz/Va_err don't saturate for fast planes)
        self.va_scale = max(0.5 * self.base_vel, 5.0)
        self.vz_scale = max(0.5 * self.base_vel, 5.0)

        # A3 (Va,h) joint-tracking task: per-episode random airspeed setpoint in
        # [va_lo, va_hi] (fractions of trim). "fixed" (default) keeps target_va at trim.
        self.va_mode = cfg.get("va_mode", "fixed")           # "fixed" | "dynamic"
        self.va_mode_id = {"fixed": 0, "dynamic": 1}[self.va_mode]
        self.va_lo = cfg.get("va_lo", 0.9) * self.base_vel
        self.va_hi = cfg.get("va_hi", 1.25) * self.base_vel
        self.va_constraint = cfg.get("va_constraint", "band")   # "band" (default) | "envelope" (ablated, worse)

        co = load_params(self.aircraft)
        self.elev_min, self.elev_max = [float(v) for v in co["control_limits"].elevator_limits]
        self.thr_min, self.thr_max = [float(v) for v in co["control_limits"].throttle_limits]
        AP = co["aero_params"].as_warp_struct()
        VP = co["vehicle_params"]
        VP.WP = VP.WP.as_warp_struct()
        VP.J = VP.J.as_warp_struct()
        VP.PP = VP.PP.as_warp_struct()
        if not self.enable_sensors and hasattr(VP, "sensor_system"):
            VP.sensor_system = None                       # nominal: ground-truth obs, no sensors
        elif self.enable_sensors and self.sensor_delay > 0.0:
            self._set_sensor_delay(VP.sensor_system, self.sensor_delay)
        EP = co["environment_params"]
        if self.turbulence:
            EP.turbulence = {"enable": True, "model": "dryden", "intensity": self.turbulence}
        config = {"AP": AP, "VP": VP, "CL": co["control_limits"], "EP": EP}
        self.model = Aircraft(num_envs, device, config, save_history=False,
                              in_ground_effect=self.in_ground_effect)
        self.model.solver_type = 1                       # fixed-step DP5 stages with forces frozen over the step (~1st-order in force dynamics)
        tm = getattr(self.model.wind_model, "turbulence_model", None)
        self._turb_W20_full = getattr(tm, "W_20_ms", 0.0)   # base gust scale (for curriculum)

        self.num_obs = 12 if self.h_floor is not None else 11
        d, n = device, num_envs
        self._step = wp.zeros(n, dtype=wp.int32, device=d)
        self._prev = wp.zeros((n, 2), dtype=wp.float32, device=d)
        self._act = wp.zeros((n, 2), dtype=wp.float32, device=d)
        self._obs = wp.zeros((n, self.num_obs), dtype=wp.float32, device=d)
        self._rew = wp.zeros(n, dtype=wp.float32, device=d)
        self._term = wp.zeros(n, dtype=wp.int32, device=d)
        self._trunc = wp.zeros(n, dtype=wp.int32, device=d)
        self._done = wp.zeros(n, dtype=wp.int32, device=d)
        self._reason = wp.zeros(n, dtype=wp.int32, device=d)
        self._target = wp.zeros(n, dtype=wp.float32, device=d)
        self._ierr = wp.zeros(n, dtype=wp.float32, device=d)
        self._refrate = wp.zeros(n, dtype=wp.float32, device=d)
        self._true_target = wp.zeros(n, dtype=wp.float32, device=d)
        self._va_sensed = wp.zeros(n, dtype=wp.float32, device=d)
        self._va_tgt = wp.full(n, self.target_va, dtype=wp.float32, device=d)  # per-env Va setpoint
        self._surf = wp.zeros((n, 2), dtype=wp.float32, device=d)   # deploy: inner-loop surfaces
        self._kin = wp.zeros(n, dtype=wp.float32, device=d)         # deploy: kinematic-alpha proxy

        self.num_act = 2

    def set_turb_scale(self, frac):
        """Curriculum knob: scale Dryden gust intensity to `frac` of the configured level."""
        tm = getattr(self.model.wind_model, "turbulence_model", None)
        if tm is not None:
            tm.W_20_ms = self._turb_W20_full * float(frac)

    @staticmethod
    def _set_sensor_delay(sensor_system, delay):
        # realistic PX4: only GPS position/velocity carries fusion latency; the IMU (attitude/gyro)
        # feeding the inner attitude loop stays fresh (delaying it phase-lags the PD -> instability).
        for group in sensor_system.values():
            for scfg in group.values():
                scfg["delay"] = delay if scfg.get("type") in ("position", "velocity") else 0.0

    def _obs_launch(self):
        a = self.model._actuator_states
        if self.enable_sensors:
            m = self.model.get_sensor_measurements()      # noisy / delayed
            pos, vel = m["gps_position"], m["gps_velocity"]
            ang, ori = m["gyroscope"], m["attitude_sensor"]
            wp.launch(_va_from_vel, dim=self.n, inputs=[vel, self._va_sensed], device=self.device)
            va = self._va_sensed
        else:
            s = self.model._state
            pos, vel, ang, ori = s["position"], s["linear_vel"], s["angular_vel"], s["orientation"]
            va = self.model._Va
        if self.deploy or self.kin_alpha:                  # EKF2 has no AoA: kinematic proxy
            wp.launch(_kin_alpha, dim=self.n, inputs=[vel, self._kin], device=self.device)
            alpha_obs = self._kin
        else:
            alpha_obs = self.model._alpha
        wp.launch(compute_obs, dim=self.n, inputs=[
            pos, vel, ang, ori,
            va, alpha_obs,                                 # aero alpha, or kinematic proxy in deploy
            a["elevator"], a["elevator_dot"], a["throttle_left"],
            self._ierr, self._refrate, self._target, self._va_tgt,
            self.elev_min, self.elev_max, self.thr_min, self.thr_max,
            self.va_scale, self.vz_scale, self._obs],
            device=self.device)
        if self.h_floor is not None:
            wp.launch(_floor_obs, dim=self.n, inputs=[
                pos, self.h_floor, self.floor_margin, self._obs], device=self.device)

    def reset(self):
        n = self.n
        if self.mode == "dynamic":
            tgt0 = np.random.uniform(self.tgt_lo, self.tgt_hi, n).astype(np.float32)
        else:
            tgt0 = np.full(n, self.target, dtype=np.float32)
        if self.spawn_above:
            alt = tgt0 + self.spawn * np.random.rand(n)          # descent-only: [tgt, tgt+spawn]
        else:
            alt = tgt0 + self.spawn * (2.0 * np.random.rand(n) - 1.0)
        if self.h_floor is not None:
            alt = np.maximum(alt, self.h_floor + self.floor_margin)
        pos = np.zeros((n, 3), dtype=np.float32)
        pos[:, 2] = -alt
        vel = np.zeros((n, 3), dtype=np.float32)
        vel[:, 0] = self.base_vel
        ori = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1))
        self.model.reset(
            {"position": pos, "linear_vel": vel,
             "angular_vel": np.zeros((n, 3), dtype=np.float32), "orientation": ori},
            {"elevator": TRIM_ELEV, "aileron": 0.0, "rudder": 0.0,
             "throttle_left": TRIM_THROTTLE, "throttle_right": TRIM_THROTTLE,
             "elevator_dot": 0.0, "aileron_dot": 0.0, "rudder_dot": 0.0})
        self._step.zero_()
        self._prev.zero_()
        self._ierr.zero_()
        va0 = (np.random.uniform(self.va_lo, self.va_hi, n) if self.va_mode == "dynamic"
               else np.full(n, self.target_va)).astype(np.float32)
        wp.copy(self._va_tgt, wp.array(va0, device=self.device))
        wp.copy(self._true_target, wp.array(tgt0.astype(np.float32), device=self.device))
        # ramp mode: sub-target starts at the aircraft altitude, then ramps to true target
        sub = alt.astype(np.float32) if self.ref_rate > 0.0 else tgt0.astype(np.float32)
        wp.copy(self._target, wp.array(sub, device=self.device))
        self._refrate.zero_()
        self._obs_launch()
        obs = wp.to_torch(self._obs)
        if self._mask:
            obs = obs.clone(); obs[:, self._mask] = 0.0
        return obs

    def step(self, action):
        # action: torch (n,2) cuda float32 in [-1,1]
        act = action.contiguous().to(torch.float32)
        wp.copy(self._act, wp.from_torch(act, dtype=wp.float32))
        if self.deploy:                                    # RL pitch-setpoint -> PD inner loop -> elevator
            wp.launch(_inner_pitch, dim=self.n, inputs=[
                self._act, self.model._state["orientation"], self.model._state["angular_vel"],
                self.pitch_max, self.kp_pitch, self.kd_pitch, self._surf], device=self.device)
            anp = wp.to_torch(self._surf).detach().cpu().numpy()
        else:
            anp = act.detach().cpu().numpy()
        zeros = np.zeros(self.n, dtype=np.float32)
        self.model.step({"elevator": anp[:, 0], "aileron": zeros, "rudder": zeros,
                         "throttle_left": anp[:, 1], "throttle_right": anp[:, 1]})

        s = self.model._state
        if self.ref_rate > 0.0:
            wp.launch(_ramp_reference, dim=self.n, inputs=[
                self._true_target, self.dt, self.ref_rate, self._target, self._refrate],
                device=self.device)
        wp.launch(_update_integral, dim=self.n, inputs=[
            s["position"], self._target, self.dt, self.i_clamp, self._ierr], device=self.device)

        wp.launch(compute_reward, dim=self.n, inputs=[
            s["position"], s["linear_vel"], s["angular_vel"], self.model._Va, self.model._alpha,
            self._act, self._prev,
            self._target, self._va_tgt, self.w_alt, self.w_va, self.w_pr, self.w_sm,
            self.sigma, self.zone, self.va_band, self.recovery, self.ov_w, self.ov_thr,
            0.01, self.alpha_safety_w, self.alpha_safety_start, self.stall_deg,
            self.va_safety_w, self.va_safety_start, self.va_stall,
            self.g, self.energy_w, self.energy_sigma,
            self.damp_w, self.damp_zone, self.pr_cap, self.effort_w, self._rew], device=self.device)

        wp.launch(check_termination_batch, dim=self.n, inputs=[
            s["position"], self.model._alpha, self.model._Va,
            self.wing_cg_z, self.alpha_limit, self.va_min, self._term, self._reason],
            device=self.device)
        if self.h_floor is not None:
            wp.launch(_floor_term, dim=self.n, inputs=[
                s["position"], self.h_floor, self._term, self._reason], device=self.device)
        wp.launch(_nan_guard, dim=self.n, inputs=[
            s["position"], self.model._Va, self._term, self._reason], device=self.device)
        wp.launch(_inc_and_truncate, dim=self.n, inputs=[
            self._step, self.horizon, self._term, self._trunc, self._done], device=self.device)
        wp.launch(_apply_terminal, dim=self.n, inputs=[
            self._reason, self._trunc, s["position"], self._target, self.zone,
            self.crash_pen, self.stall_pen, self.horizon_bonus, self._rew], device=self.device)
        if self.h_floor is not None and (self.floor_w > 0.0 or self.floor_pen > 0.0):
            wp.launch(_floor_shaped, dim=self.n, inputs=[
                s["position"], self._reason, self.h_floor, self.floor_margin,
                self.floor_w, self.floor_pen, self._rew], device=self.device)

        reward = torch.nan_to_num(wp.to_torch(self._rew)).clone()
        done = wp.to_torch(self._done).clone()
        truncated = wp.to_torch(self._trunc).clone()

        constraints = None
        success = None
        if self.cat:
            # Safety soft constraints (pre-reset state, same tensors the reward kernel saw):
            # [alpha margin, low airspeed, airspeed band, pitch rate].
            # - Va band IS safety on this open-loop-unstable airframe (airspeed = energy =
            #   stall margin): dropping it collapses robustness 1.00 -> 0.21.
            # - Overshoot stays OUT: as a constraint it biases the policy to park below the
            #   target (undershoot is constraint-free); precision is the reward's job.
            # - Action-rate stays OUT: it fires on exploration noise (std 0.5-1.0 -> |da|~1 >>
            #   any sane limit), putting a constant delta on every step -> effective horizon
            #   collapses to ~5 steps and tracking dies. CAPS (on the actor MEAN) owns
            #   smoothness instead — orthogonal to CaT.
            ang_t = wp.to_torch(s["angular_vel"])
            va_t = wp.to_torch(self.model._Va); al_t = wp.to_torch(self.model._alpha)
            alpha_deg = al_t.abs() * (180.0 / np.pi)
            va_tgt_t = wp.to_torch(self._va_tgt)
            if self.va_mode == "dynamic" and self.va_constraint == "envelope":
                # ablated alternative: constrain Va to the commanded envelope [va_lo, va_hi]
                # only. Looks principled (setpoint precision = reward's job) but LOSES: without
                # the dense band signal the energy-coupled V7 tracks far worse (h-RMSE 2.8 vs
                # 1.5, Va-RMSE 4.3 vs 2.7). On an energy-critical airframe, deviation from the
                # commanded airspeed IS a safety quantity — the band stays the default.
                va_band_c = torch.maximum(self.va_lo - va_t, va_t - self.va_hi)
            else:
                va_band_c = (va_t - va_tgt_t).abs() - self.va_band
            cons = [
                alpha_deg - self.cat_alpha_onset * self.stall_deg,
                self.va_safety_start - va_t,
                va_band_c,
                ang_t[:, 1].abs() - self.cat_pr_lim,
            ]
            if self.h_floor is not None:
                # soft floor margin (hard floor breach already terminates, reason 3)
                h_t = -wp.to_torch(s["position"])[:, 2]
                cons.append(self.h_floor + self.floor_margin - h_t)
            if self.sink_max is not None:
                # flare task: cap sink rate near the ground (world-frame vz, + = sinking/down)
                alt = -wp.to_torch(s["position"])[:, 2]
                vz_t = wp.to_torch(s["linear_vel"])[:, 2]
                gate = (alt < self.sink_gate).float()
                cons.append((vz_t - self.sink_max) * gate)
            constraints = torch.nan_to_num(torch.stack(cons, dim=1)).clone()
            # SCoCaT dense success indicator: within the target tolerance band (vs the TRUE
            # target, not the ramping sub-target). Consumed by the success critic only.
            # A3 (va_mode dynamic): joint success requires the Va band too.
            # success_va: altitude-only success lets the policy park in-band while bleeding
            # airspeed into a stall (success advantages outweigh survival damping — the
            # bonus-competition failure through the advantage channel); require the Va band.
            h_all = -wp.to_torch(s["position"])[:, 2]
            success = (h_all - wp.to_torch(self._true_target)).abs() < self.zone
            if self.va_mode == "dynamic" or self.success_va:
                success &= (va_t - va_tgt_t).abs() < self.va_band
            success = success.float().clone()

        self._obs_launch()                               # terminal obs (for bootstrap) pre-reset
        terminal_obs = torch.nan_to_num(wp.to_torch(self._obs)).clone()
        if self._mask:
            terminal_obs[:, self._mask] = 0.0
        wp.launch(_set_prev, dim=self.n, inputs=[self._act, self._prev], device=self.device)

        self._seed += 1
        a = self.model._actuator_states
        wp.launch(_masked_reset, dim=self.n, inputs=[
            self._done, self._seed, self.target, self.spawn, TRIM_THROTTLE,
            s["position"], s["linear_vel"], s["angular_vel"], s["orientation"],
            a["elevator"], a["aileron"], a["rudder"],
            a["throttle_left"], a["throttle_right"],
            a["elevator_dot"], a["aileron_dot"], a["rudder_dot"],
            self._step, self._prev, self._target, self._ierr,
            wp.int32(self.mode_id), self.tgt_lo, self.tgt_hi, self.base_vel,
            wp.int32(1 if self.ref_rate > 0.0 else 0), self._true_target,
            wp.float32(0.0 if self.h_floor is None else self.h_floor + self.floor_margin),
            wp.int32(1 if self.spawn_above else 0),
            wp.int32(self.va_mode_id), wp.float32(self.va_lo), wp.float32(self.va_hi),
            self._va_tgt],
            device=self.device)

        self._obs_launch()
        obs = torch.nan_to_num(wp.to_torch(self._obs))
        if self.obs_noise > 0.0:
            # noise ONLY on the reconstructed actuator channels (6 elev, 7 elev-rate, 8 throttle):
            # the flight-stack plugin estimates these from a servo model. h_err/pitch/rates/Va come
            # straight from EKF2 and are accurate, so leave them clean (else tracking collapses).
            obs = obs.clone()
            obs[:, 6:9] = obs[:, 6:9] + torch.randn_like(obs[:, 6:9]) * self.obs_noise
        if self._mask:
            obs = obs.clone(); obs[:, self._mask] = 0.0
        info = {"terminal_obs": terminal_obs, "truncated": truncated,
                "reason": wp.to_torch(self._reason).clone()}
        if constraints is not None:
            info["constraints"] = constraints
            info["success"] = success
        return obs, reward, done.bool(), info
