"""LQR attitude-speed executor: the benchmark's re-derived LQR gains driven by a
(phi*, hdot*, Va*) command stream instead of a position trajectory, so the maneuver-tracking
figures are directly comparable against the RL attitude executors. `actor_mean` documents how the
three commanded channels enter the reference state K was designed for.

Run through: `falcons eval --algo lqr --task attitude --plane <AC> --maneuver <maneuver>`.
"""
import numpy as np
import torch

from falcons.controllers.lqr.control import LQR_Control

G = 9.81


def _euler_to_quat_wfirst(roll, pitch, yaw):
    """[w, x, y, z] for a Z-Y-X (yaw-pitch-roll) rotation -- the convention LQR_Control's
    reference quaternion / get_quaternion_error use."""
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return np.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


class LQRAttitudeExecutor:
    """`.actor_mean(obs) -> normalized [elevator, aileron, rudder, throttle]`, so the attitude
    eval plotters drive LQR through the same rollout loop as a learned policy. `obs` is ignored:
    the control law reads the plant state of `env` directly."""

    def __init__(self, aircraft, env, dt=0.01):
        # LQR_Control owns the gains, the actuator-state estimators, the integral action and the
        # command normalization; only its own CPU sim (`step()`) is unused here.
        self.lqr = LQR_Control(aircraft_name=aircraft,
                               targets={"trajectory_type": "constant_altitude"},
                               save_history=False, use_sensor_noise=False, use_estimator=False)
        self.env = env
        self.dt = dt
        self.nm = self.lqr.n_motors
        self.ns = self.lqr.n_aero_surfaces
        self.prev_u = np.zeros(self.nm + self.ns)
        self.alt_ref = float(env.spawn_alt)
        # Trim feed-forward: u = u_trim - K(x - ref). The gains linearize the plant AROUND trim,
        # so without u_trim the law commands zero throttle at zero error -- on a position-tracking
        # trajectory the altitude integrator eventually rebuilds it, but an attitude executor that
        # holds its altitude reference has no such error to integrate, and the aircraft dumps its
        # energy in the first second. Motor units of `u` are the physical throttle (0-1, see
        # normalize_actuator_commands), so the env's spawn (trim) throttle drops straight in.
        self.u_trim = np.zeros(self.nm + self.ns)
        self.u_trim[:self.nm] = float(env.model._actuator_states["throttle_left"].numpy()[0])

    def _targets(self):
        import warp as wp
        return (float(wp.to_torch(self.env._phi_tgt).cpu().numpy()[0]),
                float(wp.to_torch(self.env._hdot_tgt).cpu().numpy()[0]),
                float(wp.to_torch(self.env._va_tgt).cpu().numpy()[0]))

    def actor_mean(self, obs):
        """How the command stream maps onto the LQR's reference state. The gains are fixed, so the
        command has to enter through channels K was designed for (see scripts/derive_lqr_gains.py):

          phi*   -> reference QUATERNION = roll(phi*) about the CURRENT heading (yaw is copied from
                    the aircraft, pitch reference is 0). Copying yaw is what makes a sustained turn
                    possible: a fixed yaw reference would make the quaternion channel fight the turn
                    it just commanded. Feed-forward coordinated-turn body rates
                    (p, q, r) = (0, psidot sin phi, psidot cos phi), psidot = g tan(phi*) / Va*, go
                    into the angular-velocity reference.
          hdot*  -> reference ALTITUDE, integrated from the spawn altitude
                    (z_ref = -(alt0 + int hdot*)). This LQR has no climb-rate channel of its own;
                    its altitude gains + z integrator are what track a climb, exactly as in the
                    benchmark's ramp_ascent scenario. Note this gives LQR altitude-error feedback
                    that the RL rate executor does not have (the RL equivalent is the `_althold`
                    variant).
          Va*    -> reference body velocity [Va*, 0, 0].

        Along-track and lateral position errors are nulled (x_ref, y_ref = current x, y): the LQR's
        position gains would otherwise pull the aircraft back onto its start line and cancel the
        turn."""
        s = self.env.model._state
        p = s["position"].numpy()[0].astype(float)
        vb = s["linear_vel"].numpy()[0].astype(float)
        wb = s["angular_vel"].numpy()[0].astype(float)
        qx, qy, qz, qw = s["orientation"].numpy()[0].astype(float)
        q_wfirst = np.array([qw, qx, qy, qz])
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))

        phi_c, hdot_c, va_c = self._targets()
        self.alt_ref += hdot_c * self.dt

        # actuator/motor estimator states advance on the PREVIOUS command (as in
        # LQR_Control.compute_action)
        if self.ns:
            self.lqr.second_order_actuator_dynamics(self.prev_u[self.nm:self.nm + self.ns], self.dt)
        if self.nm:
            self.lqr.first_order_actuator_dynamics(self.prev_u[:self.nm], self.dt)

        pos_ref = np.array([p[0], p[1], -self.alt_ref])       # only the altitude error is tracked
        x_int = self.lqr.integral_action(pos_ref - p)

        q_ref = _euler_to_quat_wfirst(phi_c, 0.0, yaw)
        # shortest-path (w >= 0) error: the plant quaternion integrates continuously, so once a
        # sustained turn accumulates more than 180deg of heading it sits on the opposite sheet of
        # the double cover from the euler-built reference. Without this the attitude feedback
        # silently changes sign mid-turn and the aircraft departs.
        q_err = self.lqr.get_quaternion_error(q_wfirst, q_ref)
        q_err = -q_err if q_err[0] < 0 else q_err
        q_err = q_err[1:]

        psidot = G * np.tan(phi_c) / max(va_c, 1.0)            # coordinated-turn rate
        vel_ref = np.array([va_c, 0.0, 0.0])
        angvel_ref = np.array([0.0, psidot * np.sin(phi_c), psidot * np.cos(phi_c)])

        x = np.concatenate([self.lqr.actuator_states_est, self.lqr.motor_states_est,
                            vb, wb, q_err, p, x_int])
        ref = np.concatenate([np.zeros(self.lqr.n_aero_states + self.nm),
                              vel_ref, angvel_ref, np.zeros(3), pos_ref, np.zeros(3)])

        u = self.u_trim - self.lqr.K_LQR @ (x - ref)
        self.prev_u = u.copy()
        na = self.lqr.normalize_actuator_commands(u)           # [motors..., elev, ail, rud]

        a = np.array([[na[self.nm], na[self.nm + 1], na[self.nm + 2], na[0]]], np.float32)
        return torch.from_numpy(a).to(obs.device if hasattr(obs, "device") else "cuda")
