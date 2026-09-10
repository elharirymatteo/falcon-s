"""MPPI attitude-speed executor: the same sampling MPC as `MPPIAltitudeControl`, but its cost
tracks a (phi*, hdot*, Va*) command STREAM instead of a position trajectory -- i.e. it solves the
same task as the RL attitude executor (`envs/attitude.py`), so the maneuver-tracking figures are
directly comparable across MPPI / LQR / PPO / SAC / TD3.

Run through: `falcons eval --algo mppi --task attitude --plane <AC> --maneuver <maneuver>`.
"""
import numpy as np
import torch
import warp as wp

from falcons.aircraft.params import load_params
from falcons.controllers.mppi.control import MPPIAltitudeControl
from falcons.controllers.mppi.kernels import _evaluate_trajectories_attitude

MPPI_SPECS = {
    "number_of_sampled_trajectories": 1000,
    "number_of_iterations_per_sample": 100,
    "noise_variance_elevator": 0.10,
    "noise_variance_aileron": 0.08,
    "noise_variance_rudder": 0.08,
    "noise_variance_throttle_both": 0.10,
    "temperature": 3.0,
}


class MPPIAttitudeControl(MPPIAltitudeControl):
    """MPPI planner scoring sampled rollouts against the attitude command stream.

    The only change from the altitude adapter in `falcons.controllers.mppi.altitude` is the cost:
    |phi - phi*| + |hdot - hdot*| + |Va - Va*| + stall/ground guards + sideslip + turn-rate damping
    (`_evaluate_trajectories_attitude`). Pure RATE tracking, exactly like the RL reward -- no
    altitude-position feedback, so it drifts in altitude the same way the RL executor does."""

    def __init__(self, specs, config, phi_s, hdot_s, va_s, alpha_limit):
        super().__init__(specs, {"trajectory_type": "constant_altitude"}, config)
        self.phi_ref = wp.array(np.asarray(phi_s, np.float32), dtype=wp.float32, device=self._device)
        self.hdot_ref = wp.array(np.asarray(hdot_s, np.float32), dtype=wp.float32, device=self._device)
        self.va_ref = wp.array(np.asarray(va_s, np.float32), dtype=wp.float32, device=self._device)
        self.alpha_limit = float(alpha_limit)
        self.step0 = 0

    def evaluate_trajectories(self):
        wp.launch(
            kernel=_evaluate_trajectories_attitude,
            dim=self.number_of_sampled_trajectories * self.number_of_iterations_per_sample,
            inputs=[
                self.number_of_iterations_per_sample,
                self.position_memory,
                self.linear_vel_memory,
                self.angular_vel_memory,
                self.orientation_memory,
                self.alpha_memory,
                self.beta_memory,
                self.costs,
                self.step0,
                self.phi_ref,
                self.hdot_ref,
                self.va_ref,
                self.alpha_limit,
            ],
            device=self._device,
        )

    def MPPI_step(self):
        action = super().MPPI_step()
        self.step0 += 1
        return action


class MPPIAttitudeExecutor:
    """`.actor_mean(obs) -> normalized [elevator, aileron, rudder, throttle]`, so the attitude
    eval plotters drive MPPI through the exact same rollout loop as a learned policy. `obs` is
    ignored: MPPI re-plans from the plant state of `env` each step."""

    def __init__(self, aircraft, env, phi_s, hdot_s, va_s, seed=0):
        co = load_params(aircraft)
        AP = co["aero_params"].as_warp_struct()
        VP = co["vehicle_params"]
        VP.WP = VP.WP.as_warp_struct()
        VP.J = VP.J.as_warp_struct()
        VP.PP = VP.PP.as_warp_struct()
        EP = co["environment_params"]
        # the planner's internal model is the nominal one: no sensor noise, no wind (the plant
        # `env` keeps whatever turbulence it was configured with).
        if hasattr(VP, "sensor_system"):
            VP.sensor_system = None
        if hasattr(EP, "constant_wind"):
            EP.constant_wind = (0.0, 0.0, 0.0)
        if hasattr(EP, "turbulence") and EP.turbulence:
            EP.turbulence = {"enable": False}

        self.env = env
        self.mppi = MPPIAttitudeControl(MPPI_SPECS,
                                        {"AP": AP, "VP": VP, "CL": co["control_limits"], "EP": EP},
                                        phi_s, hdot_s, va_s, VP.alpha_max)
        # Pin the sampling noise. The warp aircraft base otherwise seeds this from entropy. This
        # alone does not make a rollout reproducible: the rollout-cost reduction in kernels.py uses
        # wp.atomic_add, whose thread ordering is not deterministic, so a single rollout is one
        # draw. Callers fly several draws with different seeds and report the spread.
        if seed is not None:
            self.mppi.seed(int(seed))
        s = env.model._state
        a = env.model._actuator_states
        self.mppi.reset(self._state_dict(s), self._action_dict(a))

    @staticmethod
    def _state_dict(s):
        return {"position": s["position"].numpy()[0],
                "linear_vel": s["linear_vel"].numpy()[0],
                "angular_vel": s["angular_vel"].numpy()[0],
                "orientation": s["orientation"].numpy()[0]}

    @staticmethod
    def _action_dict(a):
        return {k: float(a[k].numpy()[0]) for k in
                ("elevator", "aileron", "rudder", "throttle_left", "throttle_right",
                 "elevator_dot", "aileron_dot", "rudder_dot")}

    def actor_mean(self, obs):
        # The per-step `reset` refills the whole horizon with the current actuator command instead
        # of warm-starting from the previous solution. That is not an oversight: measured on the V7
        # circle, warm-starting instead makes the plan drift (bank RMSE 18.1 deg, departs at 5.9 s)
        # because the sampler explores around an unregularized mean sequence that saturates, while
        # the refill holds 1.45 deg.
        self.mppi.reset(self._state_dict(self.env.model._state),
                        self._action_dict(self.env.model._actuator_states))
        u = self.mppi.MPPI_step()
        a = np.array([[u["elevator"], u["aileron"], u["rudder"], u["throttle_left"]]], np.float32)
        return torch.from_numpy(a).to(obs.device if hasattr(obs, "device") else "cuda")
