#!/usr/bin/env python3
"""Fly a FALCON-S altitude-keeping PPO policy in X-Plane over the XPlaneConnect bridge.

The policy is a checkpoint from `checkpoints/ppo_altitude_<plane>_s<seed>.pt`. It was trained
against the Warp plant, so everything between the network and the aeroplane has to be rebuilt
here: the eleven-element observation, the actuator models whose states two of those elements
report, the rate-limited altitude reference and the clamped integral term. All of it is
transcribed from `falcons/sim/warp/altitude_obs.py`, `falcons/envs/altitude.py` and
`falcons/envs/configs.py`, and the actuator models are the plant's own classes rather than a
second implementation of them.

    --mock   flies the FALCON-S CPU plant instead of X-Plane, through the same code path. It
             needs no X-Plane and no GPU, and it is how the shim below is tested: if altitude
             hold works there, the observation assembly and the actuator emulation are right,
             because the policy is back on the dynamics it was trained on.

    default  connects to X-Plane. The X-Plane axis, sign and unit conventions in XPlaneBridge
             are NOT flight-tested -- see the README before trusting a flight, and expect to
             adjust ELEVATOR_SIGN and the velocity mapping on the first one.

Two caveats worth stating in the code as well as the README. The policy commands the elevator
and the throttle only; ailerons and rudder are left at zero, exactly as in training, so nothing
holds the wings level and a lateral upset is not the policy's to correct. And X-Plane will be
flying whichever aeroplane you have loaded, not the FALCON-S airframe the policy was trained on,
which makes this a transfer test rather than a validation.
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))     # tools/xplane, for `import xpc`

# --- X-Plane conventions: the one place to flip a reversed axis -------------------------------
# sendCTRL takes [elevator, aileron, rudder, throttle]. ELEVATOR_SIGN maps FALCON-S's
# positive-deflection convention onto X-Plane's stick sign; it is a starting point rather than a
# measured result. Aileron and rudder are commanded zero throughout, as in training.
ELEVATOR_SIGN = -1.0


def euler_to_quat(roll, pitch, yaw):
    """ZYX Euler -> [w, x, y, z], the FALCON-S convention (see falcons/sim/cpu/physics)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr,
                     cy * sp * cr + sy * cp * sr, sy * cp * cr - cy * sp * sr])


def quat_to_body(q, v_ned):
    """Rotate an NED vector into body axes with q = [w, x, y, z]."""
    w, x, y, z = q
    R = np.array([                                   # body <- NED
        [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
        [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]])
    return R @ np.asarray(v_ned)


# --- the two things the loop can fly ----------------------------------------------------------

class XPlaneBridge:
    """State out of X-Plane and stick positions back in, over XPlaneConnect."""

    def __init__(self, host="localhost", port=49009):
        import xpc
        self._c = xpc.XPlaneConnect(xpHost=host, xpPort=port)
        try:
            self._c.getDREF("sim/test/test_float")       # is anything answering on that port?
        except (TimeoutError, OSError) as error:
            raise SystemExit(
                f"no answer from X-Plane at {host}:{port} ({error}).\n"
                f"  - is X-Plane running, with the XPlaneConnect plugin in "
                f"Resources/plugins and an aeroplane loaded?\n"
                f"  - to try the loop without X-Plane at all, run with --mock.") from None
        self._origin = None

    def read(self):
        posi = self._c.getPOSI()             # [lat, lon, alt_m, pitch, roll, heading, gear]
        altitude = posi[2]
        pitch, roll, yaw = (math.radians(a) for a in (posi[3], posi[4], posi[5]))
        quat = euler_to_quat(roll, pitch, yaw)

        # X-Plane's local frame is x=east, y=up, z=south, so NED = [-vz, vx, -vy]. Going through
        # the local velocities and the attitude keeps the mapping derivable, rather than trusting
        # the semantics of the acf-axis force DREFs.
        vx, vy, vz = (self._c.getDREF(f"sim/flightmodel/position/local_v{a}")[0] for a in "xyz")
        vel_ned = np.array([-vz, vx, -vy])
        local_x, local_z = (self._c.getDREF(f"sim/flightmodel/position/local_{a}")[0] for a in "xz")
        if self._origin is None:
            self._origin = (local_x, local_z)
        north, east = -(local_z - self._origin[1]), local_x - self._origin[0]

        rates = np.radians([self._c.getDREF(f"sim/flightmodel/position/{a}")[0] for a in "PQR"])
        return {"position": np.array([north, east, -altitude]),
                "linear_vel": quat_to_body(quat, vel_ned),
                "angular_vel": rates,
                "orientation": quat,
                "Va": self._c.getDREF("sim/flightmodel/position/true_airspeed")[0],
                "alpha": math.radians(self._c.getDREF("sim/flightmodel/position/alpha")[0])}

    def apply(self, action):
        """X-Plane has no model of our servos, so it gets the servo's output position."""
        self._c.sendCTRL([ELEVATOR_SIGN * action["elevator_pos"], 0.0, 0.0,
                          float(np.clip(action["throttle_pos"], 0.0, 1.0))])

    def neutral(self):
        self._c.sendCTRL([0.0, 0.0, 0.0, 0.0])

    def close(self):
        self._c.close()


class MockPlant:
    """The FALCON-S CPU plant behind the same interface, for a run without X-Plane.

    This exercises the shim, not the X-Plane conventions: here the state arrives in FALCON-S's
    own frame already, so a sign error in XPlaneBridge would not show up.
    """

    def __init__(self, plane, altitude, airspeed):
        from falcons.sim.cpu.aircraft import Aircraft
        self.ac = Aircraft(plane)
        self.dt = self.ac.get_time_step()
        self.ac.reset(init_state={"position": np.array([0.0, 0.0, -altitude]),
                                  "linear_vel": np.array([airspeed, 0.0, 0.0]),
                                  "angular_vel": np.zeros(3),
                                  "orientation": np.array([1.0, 0.0, 0.0, 0.0])})
        self._cmd = (0.0, -1.0)

    def read(self):
        s = self.ac.state
        vel = s["linear_vel"]
        return {"position": s["position"].copy(), "linear_vel": vel.copy(),
                "angular_vel": s["angular_vel"].copy(), "orientation": s["orientation"].copy(),
                "Va": float(np.linalg.norm(vel)),
                "alpha": float(math.atan2(vel[2], vel[0]))}

    def apply(self, action):
        """The plant models the actuators itself, so it gets the commands, not the positions --
        sending the shim's servo output here would put two lags in series."""
        self._cmd = (action["elevator_cmd"], action["throttle_cmd"])

    def advance(self):
        elevator, throttle = self._cmd
        self.ac.step(np.array([elevator, 0.0, 0.0]),
                     np.array([throttle] * self.ac.get_motor_action_size()))

    def neutral(self):
        pass

    def close(self):
        pass


# --- the deployment shim ----------------------------------------------------------------------

class AltitudeHoldPolicy:
    """The eleven-element observation, the actuator states it reports, and the policy.

    Observation, scales and clip are transcribed from falcons/sim/warp/altitude_obs.py; the
    reference ramp and the integral clamp from falcons/envs/altitude.py.
    """

    CLIP = 3.0
    SCALE = dict(h=25.0, q=2.0, pitch=math.pi / 4, alpha=math.radians(20.0),
                 elev=1.0, elev_dot=100.0, thr=1.0, ierr=10.0, ref_rate=5.0)

    def __init__(self, plane, target_altitude, seed=0, device="cpu", checkpoints=None):
        from falcons.aircraft.config import AircraftConfig
        from falcons.controllers.policies import load_policy
        from falcons.envs.configs import altitude_env_cfg
        from falcons.paths import CKPT_DIR
        from falcons.sim.cpu.actuators import ActuatorSystem

        config = AircraftConfig(plane).load()
        vehicle = config["vehicle_params"]
        cfg = altitude_env_cfg(plane, spawn=0.0)
        self.dt = config["environment_params"].get("dt", 0.01)

        # Same scales the env computes from the airframe's trim speed.
        self.target_va = float(config["default_initial_state"]["linear_vel"][0])
        self.va_scale = self.vz_scale = max(0.5 * self.target_va, 5.0)
        self.ref_rate = float(cfg["ref_rate"])           # m/s; 0 = step straight to the target
        self.i_clamp = float(cfg["i_clamp"])

        elevator = vehicle["actuator_system"]["aero_surfaces"]["elevator"]
        self.elev_min, self.elev_max = elevator["min_deflection"], elevator["max_deflection"]
        motor = next(iter(vehicle["actuator_system"]["motors"].values()))
        self.thr_min, self.thr_max = motor.get("min_throttle", 0.0), motor.get("max_throttle", 1.0)

        # The plant's own actuator models, so the two actuator observations are the states the
        # policy was trained to read rather than an approximation of them.
        self.actuators = ActuatorSystem(self.dt)
        for group, entries in vehicle["actuator_system"].items():
            self.actuators.add_actuator_group(group, entries)
        self.n_motors = len(vehicle["actuator_system"]["motors"])

        self.policy = load_policy("ppo", "altitude", plane, seed, device=device,
                                  ckpt_dir=checkpoints or CKPT_DIR)
        self.device = device
        self.target = float(target_altitude)
        self.reset(target_altitude)

    def reset(self, target_altitude, altitude=None):
        self.target = float(target_altitude)
        # With a rate-limited reference the tracked sub-target starts where the aeroplane is and
        # ramps; without one it is the target from the first step.
        self.sub_target = float(altitude) if (self.ref_rate > 0 and altitude is not None) \
            else float(target_altitude)
        self.ierr = 0.0
        self.ref_rate_now = 0.0
        self.actuators.reset_all()
        self.elev_state = np.zeros(2)                    # [deflection deg, rate deg/s]
        self.throttle_state = 0.0                        # [0, 1]

    def observe(self, state):
        q = state["orientation"]
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q[0] * q[2] - q[3] * q[1]))))
        elev_norm = (self.elev_state[0] - self.elev_min) / (self.elev_max - self.elev_min) * 2 - 1
        thr_norm = (self.throttle_state - self.thr_min) / (self.thr_max - self.thr_min) * 2 - 1
        raw = [(-state["position"][2] - self.sub_target) / self.SCALE["h"],
               state["angular_vel"][1] / self.SCALE["q"],
               (state["Va"] - self.target_va) / self.va_scale,
               pitch / self.SCALE["pitch"],
               state["linear_vel"][2] / self.vz_scale,
               state["alpha"] / self.SCALE["alpha"],
               elev_norm / self.SCALE["elev"],
               self.elev_state[1] / self.SCALE["elev_dot"],
               thr_norm / self.SCALE["thr"],
               self.ierr / self.SCALE["ierr"],
               self.ref_rate_now / self.SCALE["ref_rate"]]
        return np.clip(np.asarray(raw, dtype=np.float32), -self.CLIP, self.CLIP)

    def act(self, state, dt):
        """One control step: observation -> action -> the commands to send, states advanced."""
        import torch

        obs = self.observe(state)
        with torch.no_grad():
            action = self.policy.act(torch.as_tensor(obs, device=self.device).unsqueeze(0))
        elevator_cmd, throttle_cmd = (float(v) for v in np.clip(action.cpu().numpy()[0], -1.0, 1.0))

        # Advance the actuator states with the commands just issued, and the reference and the
        # integral with the altitude just measured -- the order the env uses.
        surfaces = self.actuators.apply_dynamics("aero_surfaces",
                                                 np.array([elevator_cmd, 0.0, 0.0]))
        motors = self.actuators.apply_dynamics("motors", np.array([throttle_cmd] * self.n_motors))
        servo = self.actuators.actuator_groups["aero_surfaces"].actuators[0]
        self.elev_state = np.asarray(servo.state, dtype=float).copy()
        self.throttle_state = float(motors[0])

        if self.ref_rate > 0.0:
            step = np.clip(self.target - self.sub_target, -self.ref_rate * dt, self.ref_rate * dt)
            self.sub_target += step
            self.ref_rate_now = step / dt
        else:
            self.sub_target, self.ref_rate_now = self.target, 0.0
        self.ierr = float(np.clip(self.ierr + (-state["position"][2] - self.sub_target) * dt,
                                  -self.i_clamp, self.i_clamp))

        # Both forms, because what to send depends on what the far side models: the commands
        # for something that has its own actuator dynamics, the servo positions for something
        # that does not.
        elevator_pos = (surfaces[0] - self.elev_min) / (self.elev_max - self.elev_min) * 2 - 1
        return {"elevator_cmd": elevator_cmd, "throttle_cmd": throttle_cmd,
                "elevator_pos": float(np.clip(elevator_pos, -1.0, 1.0)),
                "throttle_pos": float(self.throttle_state), "obs": obs}


# --- the loop ---------------------------------------------------------------------------------

def fly(args):
    policy = AltitudeHoldPolicy(args.plane, args.target, seed=args.seed, device=args.device,
                                checkpoints=args.checkpoints)
    dt = args.dt or policy.dt
    if args.mock:
        vehicle = MockPlant(args.plane, args.start, policy.target_va)
        dt = vehicle.dt
    else:
        vehicle = XPlaneBridge(args.host, args.port)
    policy.reset(args.target, altitude=-vehicle.read()["position"][2])

    log = None
    if args.log:
        log = csv.writer(open(args.log, "w", newline=""))
        log.writerow(["t", "h", "target", "sub_target", "Va", "pitch_deg", "alpha_deg",
                      "elevator_norm", "throttle"])
    print(f"{'t':>6} {'h':>8} {'sub tgt':>8} {'Va':>7} {'pitch':>7} {'elev':>7} {'thr':>6}")
    t, next_tick, next_report = 0.0, time.time(), 0.0
    try:
        while t < args.seconds:
            state = vehicle.read()
            action = policy.act(state, dt)
            vehicle.apply(action)
            if args.mock:
                vehicle.advance()
            else:
                next_tick += dt                          # real time, X-Plane runs on its own clock
                time.sleep(max(0.0, next_tick - time.time()))
            h = -state["position"][2]
            q = state["orientation"]
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (q[0] * q[2] - q[3] * q[1])))))
            if log:
                log.writerow([f"{t:.3f}", f"{h:.3f}", f"{policy.target:.3f}",
                              f"{policy.sub_target:.3f}", f"{state['Va']:.3f}", f"{pitch:.3f}",
                              f"{math.degrees(state['alpha']):.3f}",
                              f"{action['elevator_pos']:.4f}", f"{action['throttle_pos']:.4f}"])
            if t + 1e-9 >= next_report:
                next_report += args.report
                print(f"{t:6.1f} {h:8.2f} {policy.sub_target:8.2f} {state['Va']:7.2f} "
                      f"{pitch:7.2f} {action['elevator_pos']:7.3f} {action['throttle_pos']:6.3f}")
            t += dt
            if h < 0.0:
                print(f"below ground at t = {t:.2f} s"); break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        vehicle.neutral()
        vehicle.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plane", default="Volantex_Ranger",
                   help="airframe whose PPO checkpoint to fly; needs "
                        "checkpoints/ppo_altitude_<plane>_s<seed>.pt")
    p.add_argument("--seed", type=int, default=0, help="checkpoint seed")
    p.add_argument("--target", type=float, default=50.0, help="altitude to hold [m]")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--mock", action="store_true",
                   help="fly the FALCON-S CPU plant instead of X-Plane, no X-Plane needed")
    p.add_argument("--start", type=float, default=40.0, help="--mock only: starting altitude [m]")
    p.add_argument("--host", default="localhost"), p.add_argument("--port", type=int, default=49009)
    p.add_argument("--dt", type=float, default=None,
                   help="control period [s]; default is the airframe's")
    p.add_argument("--device", default="cpu", help="torch device for the policy")
    p.add_argument("--checkpoints", default=None)
    p.add_argument("--log", default=None, help="write a CSV of the flight here")
    p.add_argument("--report", type=float, default=2.0, help="print every this many seconds")
    fly(p.parse_args())


if __name__ == "__main__":
    main()
