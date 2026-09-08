#!/usr/bin/env python3
"""Fly a FALCON-S airframe in FALCON-S and in JSBSim from the same initial condition, with every
control at zero and the throttle shut, and report where the two disagree.

Five checks, cheapest first, so that a failure in an early one explains the later ones:

  1 coefficients   JSBSim's tables against the FALCON-S polynomial they were built from.
                   Isolates table interpolation error from everything else.
  2 aero loads     body-frame aerodynamic force and moment at matched (alpha, beta, V).
                   Run at beta = 0 and at beta != 0: sideslip is where this tool found the
                   wind-to-body rotation carrying the wrong sign of beta, and the second column
                   is what keeps that from coming back.
  3 rollout OGE    the validation proper: from an arbitrary altitude and attitude, high enough
                   that FALCON-S's ground effect is inactive. Flown twice, at the plant's
                   configured step and at a refined one, because the plant's own time-step error
                   is first order and larger than anything else measured here.
  4 ground effect  FALCON-S's lift and drag against JSBSim's over a height sweep. They part
                   company below h/b ~ 1 by exactly the mu_l and mu_d the model predicts.
  5 rollout IGE    the same rollout started low. Divergence here is the ground-effect model
                   doing its job, not a failure.

Nothing here writes to the FALCON-S package; it only reads airframe data and steps the CPU plant.
"""

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd


@contextlib.contextmanager
def quiet():
    """Swallow the greetings both simulators print when they start up."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


FT = 0.3048                     # m per ft
LBS = 4.4482216152605           # N per lbf
LBSFT = 1.3558179483314004      # N*m per lbf*ft
OMEGA_EARTH = 7.2921150e-5      # rad/s


PLANET = Path(__file__).parent / "nonrotating_planet.xml"


def effective_gravity(fdm, rotating: bool = False) -> float:
    """The gravity to give FALCON-S so that both agree on the vertical specific force.

    On the non-rotating planet this tool loads by default it is simply JSBSim's gravitation. On
    the real rotating earth — `--rotating-earth`, kept for measuring what the flat-earth
    assumption costs — the vehicle also feels 0.034 m/s^2 of centrifugal acceleration, ten times
    the difference between JSBSim's gravitation and FALCON-S's 9.81, so that comes off. Earth
    rotation's other term, Coriolis at 2*OMEGA*V, has a direction and cannot be absorbed into a
    scalar at all; on the rotating earth it is the noise floor of every rollout comparison.
    """
    gravitation = fdm["accelerations/gravity-ft_sec2"] * FT
    if not rotating:
        return gravitation
    return gravitation - OMEGA_EARTH**2 * fdm["position/radius-to-vehicle-ft"] * FT


# ---------------------------------------------------------------- attitude helpers

def euler_to_quat(phi: float, theta: float, psi: float) -> np.ndarray:
    """ZYX Euler angles (rad) -> quaternion [w, x, y, z], FALCON-S's convention.

    Check 3 verifies this against JSBSim by comparing NED velocities, which only agree if the
    quaternion here means the same rotation as the Euler angles handed to JSBSim.
    """
    cf, sf = np.cos(phi / 2), np.sin(phi / 2)
    ct, st = np.cos(theta / 2), np.sin(theta / 2)
    cp, sp = np.cos(psi / 2), np.sin(psi / 2)
    return np.array([cp * ct * cf + sp * st * sf,
                     cp * ct * sf - sp * st * cf,
                     cp * st * cf + sp * ct * sf,
                     sp * ct * cf - cp * st * sf])


def quat_to_euler_deg(q: np.ndarray) -> np.ndarray:
    """Quaternion [w, x, y, z] -> phi, theta, psi in degrees, inverse of euler_to_quat."""
    w, x, y, z = q / np.linalg.norm(q)
    phi = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    theta = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    psi = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.degrees([phi, theta, psi])


def wrap_deg(difference: np.ndarray) -> np.ndarray:
    """Angle differences into (-180, 180]. Alpha and psi both wrap during a tumble."""
    return (np.asarray(difference) + 180.0) % 360.0 - 180.0


def attitude_error_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Angle of the rotation taking one attitude to the other. No Euler singularities."""
    dot = abs(float(np.dot(q_a / np.linalg.norm(q_a), q_b / np.linalg.norm(q_b))))
    return float(np.degrees(2.0 * np.arccos(min(1.0, dot))))


def body_velocity(V: float, alpha: float, beta: float) -> np.ndarray:
    """[u, v, w] giving exactly this airspeed, angle of attack and sideslip in both simulators.

    FALCON-S takes alpha = atan2(w, u) and beta = asin(v/Va); JSBSim takes the same alpha but
    beta = atan2(v, hypot(u, w)). Those two betas are algebraically equal, so one velocity serves
    both, and the report prints each simulator's own alpha and beta as a check that they do.
    """
    return np.array([V * np.cos(beta) * np.cos(alpha),
                     V * np.sin(beta),
                     V * np.cos(beta) * np.sin(alpha)])


# ---------------------------------------------------------------- the two simulators

def open_jsbsim(plane: str, aircraft_dir: Path, dt: float, rotating: bool = False):
    import jsbsim
    with quiet():
        fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    fdm.set_aircraft_path(str(aircraft_dir.resolve()))
    # Before the aircraft: swap the planet for one that does not rotate, which is what lets the
    # two simulators agree exactly instead of only to a Coriolis floor. See nonrotating_planet.xml.
    if not rotating and not fdm.load_planet(str(PLANET.resolve()), False):
        raise SystemExit(f"JSBSim could not load the planet definition {PLANET}")
    if not fdm.load_model(f"{plane}_falcons"):
        raise SystemExit(f"JSBSim could not load {plane}_falcons from {aircraft_dir}. "
                         f"Run gen_jsbsim.py --plane {plane} first.")
    fdm.set_dt(dt)
    return fdm


def jsbsim_ic(fdm, h: float, euler: np.ndarray, uvw: np.ndarray, pqr: np.ndarray) -> None:
    """Put JSBSim at a state, at the equator so that local down is the gravity direction."""
    fdm["ic/lat-gc-deg"] = 0.0
    fdm["ic/long-gc-deg"] = 0.0
    fdm["ic/h-sl-ft"] = h / FT
    fdm["ic/phi-deg"], fdm["ic/theta-deg"], fdm["ic/psi-true-deg"] = np.degrees(euler)
    fdm["ic/u-fps"], fdm["ic/v-fps"], fdm["ic/w-fps"] = uvw / FT
    fdm["ic/p-rad_sec"], fdm["ic/q-rad_sec"], fdm["ic/r-rad_sec"] = pqr
    fdm.run_ic()
    for control in ("elevator", "aileron", "rudder"):
        fdm[f"fcs/{control}-cmd-norm"] = 0.0


def open_falcons(plane: str, gravity: float):
    """The CPU plant, with gravity overridden to the value JSBSim uses at the test altitude.

    FALCON-S's g is the constant 9.81 from the airframe config, JSBSim's comes from GM and the
    radius. Left alone the difference is 0.013 m/s^2, which integrates into more than a
    centimetre over a couple of seconds and would sit on top of everything this tool is trying
    to measure.
    """
    from falcons.sim.cpu.aircraft import Aircraft
    with quiet():
        aircraft = Aircraft(plane)
    aircraft.EP["g"] = gravity
    return aircraft


def falcons_polynomial(plane: str):
    """The plant's own aerodynamics object, for coefficients and ground-effect factors."""
    from falcons.aircraft.config import AircraftConfig
    from falcons.sim.cpu.physics.aerodynamics import PolynomialAerodynamics
    config = AircraftConfig(plane).load()
    poly = pd.read_csv(config["aero_params"]["poly_params_file"])
    aero = PolynomialAerodynamics(config["aero_params"], config["vehicle_params"],
                                  config["environment_params"], poly)
    return aero, config


def falcons_aero_loads(coeffs: dict, Q: float, wing: dict, alpha: float, beta: float):
    """Body-frame aerodynamic force and moment, transcribed from the CPU plant.

    Mirrors AircraftForcesAndMoments.compute_aerodynamic_{forces,moments}_from_coeffs in
    falcons/sim/cpu/physics/forces_moments.py, including its wind-to-body rotation. Keep the
    rotation below identical to the plant's: check 2 is what caught the sideslip sign it used to
    carry, and it can only catch the next one if this is a faithful copy.
    """
    S, b, c = wing["area"], wing["span"], wing["mac"]
    D, Y, L = Q * S * coeffs["CD"], Q * S * coeffs["CY"], Q * S * coeffs["CL"]
    ca, sa, cb, sb = np.cos(alpha), np.sin(alpha), np.cos(beta), np.sin(beta)
    wind_to_body = np.array([[ca * cb, -ca * sb, -sa],
                             [sb, cb, 0.0],
                             [sa * cb, -sa * sb, ca]])
    force = wind_to_body @ np.array([-D, Y, -L])
    moment = np.array([coeffs["Cl"] * Q * S * b,
                       coeffs["Cm"] * Q * S * c,
                       coeffs["Cn"] * Q * S * b])
    return force, moment


# ---------------------------------------------------------------- checks

# Which angle each coefficient is a function of, for the sweeps and the plots.
COEFFICIENT_ANGLE = {"CD": "alpha", "CL": "alpha", "CMy": "alpha",
                     "CY": "beta", "CMx": "beta", "CMz": "beta"}


def check_coefficients(fdm, aero, altitude: float, limit: float = 20.0,
                       step: float = 0.37) -> pd.DataFrame:
    """Sweep each coefficient along the angle it depends on, table against polynomial.

    The step is deliberately not a round number: landing on the table's own nodes would read
    back the polynomial exactly and measure nothing. Sweeping rather than sampling at random
    means the result can be plotted as two curves, where a table that had gone wrong anywhere
    would be obvious instead of buried in scatter.
    """
    angles = np.arange(-limit, limit + 1e-9, step)
    rows = []
    for sweep in ("alpha", "beta"):
        names = [name for name, angle in COEFFICIENT_ANGLE.items() if angle == sweep]
        for angle_deg in angles:
            alpha, beta = np.radians([angle_deg if sweep == "alpha" else 0.0,
                                      angle_deg if sweep == "beta" else 0.0])
            jsbsim_ic(fdm, altitude, np.zeros(3), body_velocity(50.0, alpha, beta), np.zeros(3))
            row = {"sweep": sweep, "angle_deg": angle_deg}
            for name in names:
                reference = aero.get_coefficient(name, alpha, beta, np.zeros(3))
                table = fdm[f"aero/coeff/{name}"]
                row[f"{name}_falcons"] = reference
                row[f"{name}_jsbsim"] = table
                row[f"{name}_ref"] = reference
                row[name] = table - reference          # the difference the statistics read
            rows.append(row)
    return pd.DataFrame(rows)


def check_aero_loads(fdm, aero, config, altitude: float, beta_deg: float) -> pd.DataFrame:
    wing = config["vehicle_params"]["wing"]
    state = {"position": np.array([0.0, 0.0, -altitude])}
    rows = []
    for alpha_deg in np.arange(-12.0, 12.1, 2.0):
        alpha, beta = np.radians([alpha_deg, beta_deg])
        uvw = body_velocity(50.0, alpha, beta)
        jsbsim_ic(fdm, altitude, np.zeros(3), uvw, np.zeros(3))
        coeffs = aero.get_coefficients_with_ground_effect(alpha, beta, np.zeros(3), state)
        Q = 0.5 * fdm["atmosphere/rho-slugs_ft3"] * 515.378818 * np.dot(uvw, uvw)
        force, moment = falcons_aero_loads(coeffs, Q, wing, alpha, beta)
        jsb_force = np.array([fdm[f"forces/fb{ax}-aero-lbs"] for ax in "xyz"]) * LBS
        jsb_moment = np.array([fdm[f"moments/{ax}-aero-lbsft"] for ax in "lmn"]) * LBSFT
        rows.append({"alpha_deg": alpha_deg,
                     **{f"dF{ax}_N": d for ax, d in zip("xyz", jsb_force - force)},
                     **{f"dM{ax}_Nm": d for ax, d in zip("xyz", jsb_moment - moment)},
                     # The loads themselves, so the plots can show the error against the thing
                     # it is an error in rather than leaving the reader to supply the scale.
                     **{f"F{ax}_N": v for ax, v in zip("xyz", force)},
                     **{f"M{ax}_Nm": v for ax, v in zip("xyz", moment)},
                     "F_mag_N": float(np.linalg.norm(force)),
                     "M_mag_Nm": float(np.linalg.norm(moment)),
                     "My_Nm": moment[1]})
    return pd.DataFrame(rows)


def check_ground_effect(fdm, aero, span: float) -> pd.DataFrame:
    """FALCON-S lift and drag against JSBSim's, which has no ground effect, over height."""
    alpha = np.radians(4.0)
    uvw = body_velocity(50.0, alpha, 0.0)
    jsbsim_ic(fdm, 200.0, np.zeros(3), uvw, np.zeros(3))
    CL_jsb, CD_jsb = fdm["aero/coeff/CL"], fdm["aero/coeff/CD"]
    rows = []
    for h_over_b in [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]:
        state = {"position": np.array([0.0, 0.0, -h_over_b * span])}
        coeffs = aero.get_coefficients_with_ground_effect(alpha, 0.0, np.zeros(3), state)
        rows.append({"h_over_b": h_over_b, "mu_l": coeffs["mu_l"], "mu_d": coeffs["mu_d"],
                     "CL_ratio": coeffs["CL"] / CL_jsb, "CD_ratio": coeffs["CD"] / CD_jsb,
                     "CL_falcons": coeffs["CL"], "CL_jsbsim": CL_jsb})
    return pd.DataFrame(rows)


def rollout(plane: str, aircraft_dir: Path, altitude: float, euler: np.ndarray,
            uvw: np.ndarray, pqr: np.ndarray, seconds: float, jsb_dt: float,
            plant_dt: float = None, rotating: bool = False) -> pd.DataFrame:
    """Step both simulators from one state and tabulate the difference at every FALCON-S step.

    plant_dt overrides the airframe config's step. Worth doing: the CPU plant's error is first
    order in its step and, at the configured 0.01 s, dominates everything else this tool
    measures. Running it refined separates the plant's time-step error from any disagreement
    with JSBSim.
    """
    from falcons.sim.cpu.physics.quaternion_math import quat_rotate_vector

    fdm = open_jsbsim(plane, aircraft_dir, min(jsb_dt, plant_dt or jsb_dt), rotating)
    jsbsim_ic(fdm, altitude, euler, uvw, pqr)
    aircraft = open_falcons(plane, effective_gravity(fdm, rotating))
    if plant_dt:
        aircraft.EP["dt"] = plant_dt
    dt = aircraft.get_time_step()
    # A whole number of JSBSim steps per plant step, at or below the step asked for, so the two
    # are always sampled at the same instants whatever the plant's step happens to be.
    substeps = max(1, int(round(dt / jsb_dt)))
    fdm.set_dt(dt / substeps)
    jsbsim_ic(fdm, altitude, euler, uvw, pqr)
    aircraft.reset(init_state={"position": np.array([0.0, 0.0, -altitude]),
                               "linear_vel": uvw.copy(),
                               "angular_vel": pqr.copy(),
                               "orientation": euler_to_quat(*euler)})
    zero_surfaces = np.zeros(aircraft.get_aero_action_size())
    throttle_shut = -np.ones(aircraft.get_motor_action_size())

    rows, stopped = [], None
    for step in range(int(round(seconds / dt)) + 1):
        if step:
            try:
                aircraft.step(zero_surfaces, throttle_shut)
            except ValueError as error:
                # The plant guards against runaway rates. With no controls and no thrust some
                # airframes — the airships, whose Cm is an order of magnitude the Navion's
                # against a much smaller inertia — tumble hard enough to trip it. Report the
                # rollout up to that point rather than dying on it.
                stopped = f"the plant refused to continue: {error}"
                break
            for _ in range(substeps):
                fdm.run()
        state = aircraft.state
        f_pos, f_vel, f_quat = state["position"], state["linear_vel"], state["orientation"]
        h_jsb = fdm["position/h-sl-meters"]
        j_vel = np.array([fdm["velocities/u-fps"], fdm["velocities/v-fps"],
                          fdm["velocities/w-fps"]]) * FT
        # An uncontrolled airframe with a large Cm against a small inertia — the airships —
        # tumbles until one simulator or the other loses all meaning. Stop at that point rather
        # than tabulating numbers nobody can read.
        speeds = [float(np.linalg.norm(f_vel)), float(np.linalg.norm(j_vel))]
        if not (np.isfinite(f_pos).all() and np.isfinite(speeds).all() and np.isfinite(h_jsb)):
            stopped = "a state went non-finite"
            break
        if max(speeds) > 10.0 * np.linalg.norm(uvw):
            stopped = f"the rollout left any physical range, {max(speeds):.0f} m/s"
            break
        j_ned = np.array([fdm["velocities/v-north-fps"], fdm["velocities/v-east-fps"],
                          fdm["velocities/v-down-fps"]]) * FT
        f_ned = quat_rotate_vector(f_quat, f_vel, i_to_b=False)
        q_jsb = euler_to_quat(fdm["attitude/phi-rad"], fdm["attitude/theta-rad"],
                              fdm["attitude/psi-rad"])
        f_euler = quat_to_euler_deg(f_quat)
        j_euler = np.degrees([fdm["attitude/phi-rad"], fdm["attitude/theta-rad"],
                              fdm["attitude/psi-rad"]])
        f_rates = np.degrees(state["angular_vel"])
        j_rates = np.degrees([fdm["velocities/p-rad_sec"], fdm["velocities/q-rad_sec"],
                              fdm["velocities/r-rad_sec"]])
        rows.append({
            "t": step * dt,
            # Every state both simulators can be asked for, each side kept separately so the
            # plots can show the trajectories and not only the difference.
            "h_falcons": -f_pos[2], "h_jsbsim": h_jsb, "dh": -f_pos[2] - h_jsb,
            "V_falcons": speeds[0], "V_jsbsim": speeds[1], "dV": speeds[0] - speeds[1],
            "u_falcons": f_vel[0], "u_jsbsim": j_vel[0],
            "v_falcons": f_vel[1], "v_jsbsim": j_vel[1],
            "w_falcons": f_vel[2], "w_jsbsim": j_vel[2],
            "alpha_falcons_deg": np.degrees(np.arctan2(f_vel[2], f_vel[0])),
            "alpha_jsbsim_deg": fdm["aero/alpha-deg"],
            "beta_falcons_deg": np.degrees(np.arcsin(f_vel[1] / max(speeds[0], 1e-9))),
            "beta_jsbsim_deg": fdm["aero/beta-deg"],
            **{f"{name}_falcons_deg": f_euler[i] for i, name in enumerate("phi theta psi".split())},
            **{f"{name}_jsbsim_deg": j_euler[i] for i, name in enumerate("phi theta psi".split())},
            **{f"{name}_falcons_degs": f_rates[i] for i, name in enumerate("pqr")},
            **{f"{name}_jsbsim_degs": j_rates[i] for i, name in enumerate("pqr")},
            # Ground track magnitude: FALCON-S is flat-earth NED from the spawn point, JSBSim
            # reports great-circle distance from it, so magnitudes are what compare.
            "track_falcons": float(np.linalg.norm(f_pos[:2])),
            "track_jsbsim": fdm["position/distance-from-start-mag-mt"],
            "dtrack": float(np.linalg.norm(f_pos[:2])
                            - fdm["position/distance-from-start-mag-mt"]),
            "dvel_body": float(np.linalg.norm(f_vel - j_vel)),
            # Same velocity seen in NED. Nonzero here with dvel_body near zero would mean the
            # quaternion built by euler_to_quat is not the attitude JSBSim was given.
            "dvel_ned": float(np.linalg.norm(f_ned - j_ned)),
            "datt_deg": attitude_error_deg(f_quat, q_jsb),
        })
        if -f_pos[2] < 0 or fdm["position/h-sl-meters"] < 0:
            stopped = "one of the two reached the ground"
            break
    frame = pd.DataFrame(rows)
    frame.attrs["stopped"] = stopped
    return frame


# ---------------------------------------------------------------- statistics

# The rollout channels worth a per-state line: label, column stem, unit, whether it is an angle
# that has to be differenced modulo 360.
CHANNELS = [
    ("h", "h", "m", False),
    ("V", "V", "m/s", False),
    ("u", "u", "m/s", False),
    ("v", "v", "m/s", False),
    ("w", "w", "m/s", False),
    ("track", "track", "m", False),
    ("alpha", "alpha", "deg", True),
    ("beta", "beta", "deg", True),
    ("phi", "phi", "deg", True),
    ("theta", "theta", "deg", True),
    ("psi", "psi", "deg", True),
    ("p", "p", "deg/s", False),
    ("q", "q", "deg/s", False),
    ("r", "r", "deg/s", False),
]


def channel_columns(stem: str, unit: str) -> tuple:
    suffix = {"deg": "_deg", "deg/s": "_degs"}.get(unit, "")
    return f"{stem}_falcons{suffix}", f"{stem}_jsbsim{suffix}"


def statistics(difference: np.ndarray, reference: np.ndarray = None,
               span_floor: float = 1e-6) -> dict:
    """Bias, RMS and worst case of a difference, and what fraction of the signal that is.

    Bias separated from RMS on purpose: a constant offset (a mismatched gravity, a units slip)
    and a growing divergence (an unstable mode amplifying round-off) are different faults, and
    the two statistics tell them apart. `span` is the reference channel's own peak-to-peak over
    the same window, so `max %` says whether the disagreement matters at the scale of the motion.
    """
    difference = np.asarray(difference, dtype=float)
    difference = difference[np.isfinite(difference)]
    if difference.size == 0:
        return {"bias": np.nan, "rms": np.nan, "max": np.nan, "span": np.nan, "max_pct": np.nan}
    stats = {"bias": float(difference.mean()),
             "rms": float(np.sqrt(np.mean(difference**2))),
             "max": float(np.abs(difference).max())}
    span = np.nan
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        reference = reference[np.isfinite(reference)]
        if reference.size:
            span = float(reference.max() - reference.min())
    stats["span"] = span
    # A channel that never moves has no scale to be a percentage of. Sideslip and roll rate sit
    # at machine zero in both simulators once JSBSim's planet is not rotating, and a ratio of two
    # numbers at 1e-13 says nothing at all.
    stats["max_pct"] = 100.0 * stats["max"] / span if span and span > span_floor else np.nan
    return stats


def print_statistics(rows: list, value_width: int = 11) -> None:
    """rows: (label, unit, stats dict). Prints one aligned block."""
    print(f"  {'state':<10} {'unit':<7} {'bias':>{value_width}} {'rms':>{value_width}} "
          f"{'max':>{value_width}} {'max % of span':>14}")
    for label, unit, stats in rows:
        percent = "-" if not np.isfinite(stats["max_pct"]) else f"{stats['max_pct']:.3g}"
        print(f"  {label:<10} {unit:<7} {stats['bias']:>{value_width}.3e} "
              f"{stats['rms']:>{value_width}.3e} {stats['max']:>{value_width}.3e} "
              f"{percent:>14}")
    if any(label.endswith("*") for label, _, _ in rows):
        print("  * theta passes through +-90 deg in this run, where phi and psi are not "
              "separately\n    defined: the two simulators take opposite branches and the row "
              "means nothing. Read\n    the attitude row, which comes from the quaternions, "
              "instead.")


def rollout_statistics(frame: pd.DataFrame, steep: float = 80.0) -> list:
    # Near theta = +-90 the phi and psi rows stop meaning anything, so they get starred and
    # print_statistics explains why.
    gimbal = frame["theta_falcons_deg"].abs().max() > steep
    rows = []
    for label, stem, unit, is_angle in CHANNELS:
        falcons, jsbsim = channel_columns(stem, unit)
        if falcons not in frame or jsbsim not in frame:
            continue
        difference = frame[falcons] - frame[jsbsim]
        if is_angle:
            difference = wrap_deg(difference)
        if gimbal and label in ("phi", "psi"):
            label += "*"
        rows.append((label, unit, statistics(difference, frame[falcons])))
    # The attitude as one number, straight from the quaternions, immune to all of that.
    rows.append(("attitude", "deg", statistics(frame["datt_deg"])))
    return rows


def trust_horizon(frame: pd.DataFrame, fraction: float = 0.01) -> float:
    """When the velocity difference first passes `fraction` of the initial airspeed.

    Past this the two are no longer comparable in any quantitative sense. An uncontrolled,
    undamped airframe tumbles, and a tumble amplifies geometrically whatever difference is
    already there — the plant's time-step error, or on the rotating earth the Coriolis term
    JSBSim has and FALCON-S does not. Nothing after this line is a defect.
    """
    threshold = fraction * frame["V_falcons"].iloc[0]
    exceeded = frame[frame["dvel_body"] > threshold]
    return float(exceeded["t"].iloc[0]) if len(exceeded) else np.nan


def growth_time(frame: pd.DataFrame, column: str = "dvel_body") -> float:
    """Time for the difference to grow by a factor of e, from a fit of its logarithm.

    The rollouts tumble, so the disagreement grows geometrically rather than accumulating
    linearly. One time constant says more about how far a rollout can be trusted than any single
    error figure: at ten times this, expect ten e-folds.
    """
    usable = frame[(frame["t"] > 0) & (frame[column].abs() > 0)]
    if len(usable) < 5:
        return np.nan
    slope = np.polyfit(usable["t"], np.log(usable[column].abs()), 1)[0]
    return 1.0 / slope if slope > 0 else np.nan


def summarise(name: str, frame: pd.DataFrame, columns: list) -> str:
    worst = {c: frame[c].abs().max() for c in columns}
    body = "  ".join(f"{c} {v:.3e}" for c, v in worst.items())
    return f"  {name:<22} {body}"


def write_rollout(frame: pd.DataFrame, path: Path, every: float = 0.01) -> None:
    """Save a rollout on a fixed time grid, whatever step it was flown at.

    The refined run takes twenty steps for every one of the configured run's; written out in full
    it is an eight megabyte file saying the same thing as a four hundred kilobyte one.
    """
    step = max(1, int(round(every / (frame["t"].iloc[1] - frame["t"].iloc[0]))))
    pd.concat([frame.iloc[::step], frame.iloc[[-1]]]).drop_duplicates().to_csv(path, index=False)


def print_rollout(frame: pd.DataFrame, every: float = 0.5) -> None:
    """Difference against time, not just its maximum.

    These airframes have no pitch trim at zero elevator and the CPU plant applies no rate
    damping, so the motion is a divergent tumble that magnifies any difference exponentially.
    The number that means something is how small the disagreement is early on, and how fast it
    grows after; a single maximum over the whole run hides both.
    """
    print("     t [s]      dh [m]  dtrack [m]  dvel [m/s]  datt [deg]   alpha [deg]  beta [deg]")
    step = max(1, int(round(every / (frame["t"].iloc[1] - frame["t"].iloc[0]))))
    for _, row in pd.concat([frame.iloc[::step], frame.iloc[[-1]]]).drop_duplicates().iterrows():
        print(f"  {row['t']:8.2f}  {row['dh']:10.2e}  {row['dtrack']:10.2e}  "
              f"{row['dvel_body']:10.2e}  {row['datt_deg']:10.2e}  "
              f"{row['alpha_falcons_deg']:11.2f}  {row['beta_falcons_deg']:10.2f}")
    if frame.attrs.get("stopped"):
        print(f"  ended at {frame['t'].iloc[-1]:.2f} s: {frame.attrs['stopped']}")


def report_growth(frame: pd.DataFrame) -> None:
    tau, horizon = growth_time(frame), trust_horizon(frame)
    if np.isfinite(tau):
        print(f"  the velocity difference e-folds every {tau:.2f} s, so it is 10x larger "
              f"{tau * np.log(10):.2f} s later")
    else:
        print("  the velocity difference does not grow geometrically over this window")
    if np.isfinite(horizon):
        print(f"  it passes 1 % of the initial airspeed at t = {horizon:.2f} s; the tumble "
              f"amplifies whatever\n  difference already exists, so past there the two are no "
              f"longer quantitatively comparable")
    else:
        print("  it stays under 1 % of the initial airspeed for the whole window")


def plant_only_rollout(plane: str, gravity: float, altitude: float, euler: np.ndarray,
                       uvw: np.ndarray, pqr: np.ndarray, seconds: float,
                       plant_dt: float) -> np.ndarray:
    """The plant alone, returning [u, v, w, h] at the end. For measuring its own step error."""
    aircraft = open_falcons(plane, gravity)
    aircraft.EP["dt"] = plant_dt
    aircraft.reset(init_state={"position": np.array([0.0, 0.0, -altitude]),
                               "linear_vel": uvw.copy(), "angular_vel": pqr.copy(),
                               "orientation": euler_to_quat(*euler)})
    zero_surfaces = np.zeros(aircraft.get_aero_action_size())
    throttle_shut = -np.ones(aircraft.get_motor_action_size())
    for _ in range(int(round(seconds / plant_dt))):
        aircraft.step(zero_surfaces, throttle_shut)
    return np.concatenate([aircraft.state["linear_vel"], [-aircraft.state["position"][2]]])


def report_convergence(plane: str, gravity: float, altitude: float, euler: np.ndarray,
                       uvw: np.ndarray, pqr: np.ndarray, dt: float, probe: float,
                       reference_divisor: int = 64) -> list:
    """Halve the plant's step and read off its observed order of accuracy.

    Measured against the plant's own solution at a much finer step, not against JSBSim, so the
    number is a statement about the integrator alone with no model difference mixed in. A scheme
    advertised as RK45 should show order four or better; order one means the step behaves like
    explicit Euler, which is worth knowing before trusting a rollout at 10 ms.
    """
    reference = plant_only_rollout(plane, gravity, altitude, euler, uvw, pqr, probe,
                                   dt / reference_divisor)
    print(f"\n  the plant against its own solution at dt/{reference_divisor}, at t = {probe:g} s:")
    print(f"  {'dt [s]':>10} {'velocity error [m/s]':>22} {'altitude error [m]':>20} "
          f"{'order':>7}")
    ladder, previous = [], None
    for divisor in (1, 2, 4, 8):
        final = plant_only_rollout(plane, gravity, altitude, euler, uvw, pqr, probe, dt / divisor)
        error = float(np.linalg.norm(final[:3] - reference[:3]))
        order = f"{np.log2(previous / error):.2f}" if previous and error > 0 else ""
        print(f"  {dt / divisor:>10.5f} {error:>22.3e} {final[3] - reference[3]:>20.3e} "
              f"{order:>7}")
        ladder.append((dt / divisor, error, float(final[3] - reference[3])))
        previous = error
    return ladder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plane", default="Navion")
    parser.add_argument("--aircraft", type=Path, default=Path(__file__).parent / "aircraft")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    parser.add_argument("--seconds", type=float, default=6.0,
                        help="rollout length. With JSBSim's planet not rotating there is no "
                             "earth-rotation seed for the tumble to amplify, so a long run stays "
                             "comparable; the printed horizon says if one stops being so")
    parser.add_argument("--altitude", type=float, default=200.0,
                        help="out-of-ground-effect spawn altitude, m")
    parser.add_argument("--jsb-dt", type=float, default=0.001,
                        help="JSBSim step, s. Ten times finer than the plant's so that JSBSim's "
                             "integration error stays below the model difference being measured")
    parser.add_argument("--attitude", type=float, nargs=3, default=[0.0, -10.0, 30.0],
                        metavar=("PHI", "THETA", "PSI"),
                        help="initial attitude, degrees. Wings level by default, which keeps "
                             "sideslip at zero and the comparison longitudinal; roll it and the "
                             "wind-to-body sign difference of check 2 enters the rollout too")
    parser.add_argument("--speed", type=float, default=50.0, help="initial airspeed, m/s")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="initial angle of attack, degrees")
    parser.add_argument("--rates", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar=("P", "Q", "R"), help="initial body rates, rad/s")
    parser.add_argument("--ige-h-over-b", type=float, default=0.2,
                        help="height over span for the in-ground-effect rollout")
    parser.add_argument("--rotating-earth", action="store_true",
                        help="leave JSBSim on the real rotating earth instead of the "
                             "non-rotating planet this tool loads. Turns Coriolis back on, which "
                             "puts a floor under every rollout comparison; useful only for "
                             "measuring what FALCON-S's flat-earth assumption costs")
    parser.add_argument("--refine", type=int, default=20,
                        help="also fly the out-of-ground-effect rollout with the plant's step "
                             "divided by this, to separate its time-step error from model error. "
                             "1 to skip")
    parser.add_argument("--plot", action="store_true",
                        help="write the four paper figures, as PDF and PNG")
    parser.add_argument("--diagnostic-plots", action="store_true",
                        help="also write the every-channel figures, which are for working out "
                             "where an unexpected number came from rather than for a paper")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    aero, config = falcons_polynomial(args.plane)
    span = config["vehicle_params"]["wing"]["span"]
    euler = np.radians(args.attitude)
    uvw = body_velocity(args.speed, np.radians(args.alpha), 0.0)
    pqr = np.array(args.rates)

    fdm = open_jsbsim(args.plane, args.aircraft, args.jsb_dt, args.rotating_earth)
    jsbsim_ic(fdm, args.altitude, euler, uvw, pqr)
    gravity = effective_gravity(fdm, args.rotating_earth)
    rho_jsb = fdm["atmosphere/rho-slugs_ft3"] * 515.378818
    aircraft = open_falcons(args.plane, gravity)
    aircraft.state = {"position": np.array([0.0, 0.0, -args.altitude])}
    rho_falcons = aircraft.get_air_density()

    print(f"\nFALCON-S {args.plane} against JSBSim {fdm.get_version()}")
    print(f"  spawn                  h {args.altitude:.0f} m, attitude {args.attitude} deg, "
          f"V {args.speed:.0f} m/s at alpha {args.alpha:.1f} deg, rates {args.rates} rad/s")
    print(f"  controls               zero deflection, throttle shut")
    print(f"  planet                 " + ("the real rotating earth"
          if args.rotating_earth else "earth with rotation and J2 set to zero"))
    print(f"  gravity                {gravity:.6f} m/s^2, JSBSim's, forced on both")
    print(f"  density at {args.altitude:.0f} m       "
          f"FALCON-S {rho_falcons:.6f}  JSBSim {rho_jsb:.6f} kg/m^3  "
          f"({abs(rho_falcons / rho_jsb - 1) * 100:.3f} %)")
    if args.rotating_earth:
        print(f"  rollout noise floor    {2 * OMEGA_EARTH * args.speed:.2e} m/s^2 of Coriolis, "
              f"which JSBSim has and FALCON-S does not")
    else:
        print(f"  rollout noise floor    no Coriolis on either side. What is left is JSBSim's "
              f"gravity falling off\n                         with altitude, "
              f"{2 * gravity * 100.0 / 6.371e6:.1e} m/s^2 per 100 m of climb, "
              f"against FALCON-S's constant g")

    print("\n1 coefficients: JSBSim table minus FALCON-S polynomial, swept off the table nodes")
    coefficients = check_coefficients(fdm, aero, args.altitude)
    coefficients.to_csv(args.out / f"{args.plane}_coefficients.csv", index=False)
    print_statistics([(name, "-", statistics(coefficients[name], coefficients[f"{name}_ref"]))
                      for name in ["CD", "CY", "CL", "CMx", "CMy", "CMz"]])
    print("  Bilinear interpolation of a curved function: a chord sits on one side of the arc it "
          "cuts,\n  so a bias of the same order as the rms is the expected signature and only its "
          "size is\n  news. Shrink it with gen_jsbsim.py --angle-step.")

    print("\n2 aero loads at matched (alpha, beta, V): JSBSim minus FALCON-S")
    for beta_deg in (0.0, 10.0):
        loads = check_aero_loads(fdm, aero, config, args.altitude, beta_deg)
        loads.to_csv(args.out / f"{args.plane}_loads_beta{beta_deg:.0f}.csv", index=False)
        print(f"  beta = {beta_deg:.0f} deg")
        print_statistics(
            [(axis, unit, statistics(loads[f"d{axis}_{'N' if unit == 'N' else 'Nm'}"],
                                     loads[reference]))
             for axis, unit, reference in [("Fx", "N", "Fz_N"), ("Fy", "N", "Fz_N"),
                                           ("Fz", "N", "Fz_N"), ("Mx", "N*m", "My_Nm"),
                                           ("My", "N*m", "My_Nm"), ("Mz", "N*m", "My_Nm")]])
    print("  Spans are the lift and pitching moment over the same alpha sweep, so `max %` is the "
          "error\n  against the load the airframe actually carries. The two beta blocks should "
          "agree with each\n  other: a sideslip-only discrepancy means the wind-to-body rotation "
          "again.")

    print(f"\n3 rollout out of ground effect, {args.seconds:.0f} s from h = {args.altitude:.0f} m")
    oge = rollout(args.plane, args.aircraft, args.altitude, euler, uvw, pqr,
                  args.seconds, args.jsb_dt, rotating=args.rotating_earth)
    write_rollout(oge, args.out / f"{args.plane}_rollout_oge.csv")
    print(f"  plant at its configured step, {aircraft.get_time_step():g} s")
    print_rollout(oge)
    print()
    print_statistics(rollout_statistics(oge))
    report_growth(oge)

    refined = None
    if args.refine > 1:
        plant_dt = aircraft.get_time_step() / args.refine
        refined = rollout(args.plane, args.aircraft, args.altitude, euler, uvw, pqr,
                          args.seconds, args.jsb_dt, plant_dt=plant_dt,
                          rotating=args.rotating_earth)
        write_rollout(refined, args.out / f"{args.plane}_rollout_oge_refined.csv")
        print(f"\n  plant at {plant_dt:g} s. Everything this recovers was the plant's own "
              f"time-step error,\n  not a disagreement with JSBSim")
        print_statistics(rollout_statistics(refined))
        report_growth(refined)
    ladder = report_convergence(args.plane, gravity, args.altitude, euler, uvw, pqr,
                                aircraft.get_time_step(), min(0.5, args.seconds))

    print("\n4 ground effect against a reference that has none")
    sweep = check_ground_effect(fdm, aero, span)
    sweep.to_csv(args.out / f"{args.plane}_ground_effect.csv", index=False)
    print("  h/b     mu_l     mu_d    CL ratio  CD ratio")
    for _, row in sweep.iterrows():
        print(f"  {row['h_over_b']:<6.2f} {row['mu_l']:<8.4f} {row['mu_d']:<7.4f} "
              f"{row['CL_ratio']:<9.4f} {row['CD_ratio']:.4f}")

    low = args.ige_h_over_b * span
    print(f"\n5 rollout in ground effect, {args.seconds:.0f} s from h = {low:.2f} m "
          f"(h/b = {args.ige_h_over_b:.2f}), wings level, divergence expected.\n"
          f"  Either simulator reaching the ground ends the run; neither models the contact")
    ige = rollout(args.plane, args.aircraft, low, np.radians([0.0, 0.0, 0.0]), uvw,
                  np.zeros(3), args.seconds, args.jsb_dt, rotating=args.rotating_earth)
    write_rollout(ige, args.out / f"{args.plane}_rollout_ige.csv")
    print_rollout(ige)
    print()
    print_statistics(rollout_statistics(ige))
    print(f"  after {ige['t'].iloc[-1]:.2f} s h is {ige['h_falcons'].iloc[-1]:.2f} m in FALCON-S "
          f"against {ige['h_jsbsim'].iloc[-1]:.2f} m in JSBSim. Every line here is the "
          f"ground-effect\n  model, and the altitude bias is the sign it holds the aircraft up.")

    print(f"\ncsv in {args.out}")
    if args.plot:
        for name in plot_all(args.out, args.plane, coefficients, oge, refined, ige, sweep,
                             ladder, config, args):
            print(f"plot {name}")


# ---------------------------------------------------------------- figures

# Four figures, one per claim a validation section has to make: the trajectories agree, the
# aerodynamic model transferred, ground effect is the deliberate difference and it matches its
# own factors, and the plant's time step is what limits the agreement. Everything is written as
# PDF for embedding and PNG for looking at.
#
# Sized for a two-column paper: 7.0 in across the text block, 3.4 in for a single column.
PAPER_STYLE = {
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "lines.linewidth": 1.1, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.4, "axes.axisbelow": True, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "legend.frameon": False,
}
FALCONS_STYLE = dict(color="tab:blue", ls="-")
JSBSIM_STYLE = dict(color="tab:orange", ls=(0, (4, 2)))


def _save(figure, path: Path) -> Path:
    """Write PDF next to PNG. The PDF is the one that goes in the paper."""
    import matplotlib.pyplot as plt
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"))
    plt.close(figure)
    return path.with_suffix(".pdf")


def figure_rollout(path: Path, oge: pd.DataFrame, refined: pd.DataFrame,
                   plant_dt: float, refined_dt: float) -> Path:
    """Claim one: the two simulators fly the same aeroplane.

    Four states across the top, the difference in each underneath on a log axis. The top row is
    the evidence that the comparison is a real manoeuvre rather than a trim point; the bottom row
    is the accuracy, and the two curves in it differ only in FALCON-S's own time step.
    """
    import matplotlib.pyplot as plt
    panels = [("altitude", "h", "m", False), ("airspeed", "V", "m/s", False),
              ("angle of attack, unwrapped", "alpha", "deg", True),
              ("pitch rate", "q", "deg/s", False)]
    figure, axes = plt.subplots(2, 4, figsize=(7.0, 3.4), sharex=True,
                                gridspec_kw={"height_ratios": [1.6, 1]})
    for column, (title, stem, unit, is_angle) in enumerate(panels):
        top, bottom = axes[0][column], axes[1][column]
        falcons, jsbsim = channel_columns(stem, unit)

        def series(frame, column_name, unwrap=is_angle):
            """Unwrapped, for alpha. These airframes have no pitch trim at zero elevator, so an
            uncontrolled run tumbles and alpha sawtooths through +-180 several times. Unwrapping
            turns that into a monotone climb whose slope is the tumble rate, which is both easier
            to read and free of the wrap spikes that would otherwise litter the residual."""
            values = frame[column_name].to_numpy()
            return np.degrees(np.unwrap(np.radians(values))) if unwrap else values

        top.plot(oge["t"], series(oge, falcons), label="FALCON-S", **FALCONS_STYLE)
        top.plot(oge["t"], series(oge, jsbsim), label="JSBSim", **JSBSIM_STYLE)
        top.set_ylabel(f"{stem} [{unit}]")
        top.set_title(title)
        worst = 0.0
        for frame, dt, style in [(oge, plant_dt, dict(color="tab:blue", ls="-")),
                                 (refined, refined_dt, dict(color="tab:green", ls=(0, (4, 2))))]:
            if frame is None:
                continue
            difference = pd.Series(series(frame, falcons) - series(frame, jsbsim),
                                   index=frame.index)
            moving = frame["t"] > 0                 # t = 0 is identical by construction
            bottom.semilogy(frame["t"][moving], np.abs(difference)[moving],
                            label=f"plant dt {dt * 1e3:g} ms", **style)
            worst = max(worst, float(np.abs(difference)[moving].max()))
        # Five decades below the worst case. Left to autoscale, the axis stretches to the
        # machine-zero start and the part worth reading collapses into the top line.
        bottom.set_ylim(worst / 1e5, worst * 4)
        bottom.set(xlabel="t [s]", ylabel=f"|difference| [{unit}]")
    axes[0][0].legend(loc="best")
    axes[1][0].legend(loc="best")
    figure.tight_layout(pad=0.4)
    return _save(figure, path)


def figure_aero(path: Path, coefficients: pd.DataFrame, out: Path, plane: str) -> Path:
    """Claim two: the aerodynamic model transferred.

    Left, the coefficients themselves from both sources — the check that a table was built right
    at all. Middle, what is left over, which is bilinear interpolation and nothing else. Right,
    the body-frame load error against the load being carried, at zero sideslip and at ten degrees
    of it. Both columns now sit at round-off; when the plant's wind-to-body rotation still had
    beta the wrong way round, the sideslip forces stood seven decades above the rest of the
    panel while the moments did not move, which is what localised the fault.
    """
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))

    alpha_sweep = coefficients[coefficients["sweep"] == "alpha"].sort_values("angle_deg")
    for name, colour in [("CL", "tab:blue"), ("CD", "tab:red"), ("CMy", "tab:purple")]:
        axes[0].plot(alpha_sweep["angle_deg"], alpha_sweep[f"{name}_falcons"],
                     color=colour, label=name)
        # Markers rather than a second line: two lines on top of each other look like one, and
        # the point of the panel is that they coincide.
        axes[0].plot(alpha_sweep["angle_deg"][::12], alpha_sweep[f"{name}_jsbsim"][::12],
                     color=colour, ls="none", marker="o", ms=2.6, mfc="none", mew=0.7)
    axes[0].set(xlabel="alpha [deg]", ylabel="coefficient")
    axes[0].set_title("FALCON-S polynomial (line), JSBSim table (circles)")
    axes[0].legend(loc="upper left")

    for name in ("CD", "CL", "CMy", "CY", "CMx", "CMz"):
        sweep = coefficients[coefficients["sweep"] == COEFFICIENT_ANGLE[name]]
        sweep = sweep.sort_values("angle_deg")
        axes[1].semilogy(sweep["angle_deg"], sweep[name].abs(), lw=0.7, label=name)
    axes[1].set(xlabel="alpha or beta [deg]", ylabel="|table - polynomial|")
    axes[1].set_title("table interpolation error")
    axes[1].legend(ncol=2, loc="lower center", framealpha=1.0, facecolor="white",
                   edgecolor="0.8", frameon=True)

    # Colour carries the quantity and marker the sideslip, which the key below reads off in four
    # entries instead of the six their cross product would need. The y range is fixed from
    # round-off up to the whole load with a line at 1 %, instead of autoscaling: every curve now
    # sits in the round-off band, and an axis zoomed into that band shows floating-point wiggle as
    # though it were structure. Fixed, the panel reads as "six decades under anything that
    # matters", leaves room for the key, and would show a future regression climbing towards the
    # line rather than quietly rescaling it away.
    worst = 0.0
    for beta_deg, filled in ((0.0, True), (10.0, False)):
        loads = pd.read_csv(out / f"{plane}_loads_beta{beta_deg:.0f}.csv")
        for name, keys, magnitude, colour in [
                ("force", ("dFx_N", "dFy_N", "dFz_N"), "F_mag_N", "tab:blue"),
                ("moment", ("dMx_Nm", "dMy_Nm", "dMz_Nm"), "M_mag_Nm", "tab:red")]:
            error = np.sqrt(sum(loads[key]**2 for key in keys)) / loads[magnitude]
            axes[2].semilogy(loads["alpha_deg"], error, color=colour, marker="o" if filled else "s",
                             ms=3.4, ls="-" if filled else (0, (3, 2)),
                             mfc=colour if filled else "none")
            worst = max(worst, float(error.max()))
    axes[2].axhline(1e-2, color="0.45", lw=0.8, ls=(0, (1, 2)))
    axes[2].text(0.97, 1.4e-2, "1 % of the load", transform=axes[2].get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=7, color="0.35")
    axes[2].set_ylim(1e-10, max(3.0, worst * 3.0))     # never clip a regression off the top
    axes[2].set(xlabel="alpha [deg]", ylabel="|error| / |load|")
    axes[2].set_title("load error at matched states")
    # A factorised key: colour names the quantity, marker names the sideslip. The cross product
    # would be four entries saying two things, and the dotted line is labelled where it is drawn.
    from matplotlib.lines import Line2D
    axes[2].legend(handles=[Line2D([], [], color="tab:blue", label="force"),
                            Line2D([], [], color="tab:red", label="moment"),
                            Line2D([], [], color="0.35", ls="none", marker="o", ms=3.4,
                                   label="beta = 0"),
                            Line2D([], [], color="0.35", ls="none", marker="s", ms=3.4,
                                   mfc="none", label="beta = 10 deg")],
                   # In the empty decades between the curves and the 1 % line, so it
                   # crosses neither.
                   loc="center left", ncol=1, handletextpad=0.5, labelspacing=0.3,
                   borderpad=0.3)

    figure.tight_layout(pad=0.4)
    return _save(figure, path)


def figure_ground_effect(path: Path, sweep: pd.DataFrame, ige: pd.DataFrame,
                         span: float, h_over_b: float) -> Path:
    """Claim three: ground effect is the one difference that is meant to be there.

    Left, the ratio the two simulators are measured to differ by against what FALCON-S's model
    asks for — mu_l for lift and mu_d * mu_l^2 for drag, since that is how the plant composes
    them. Right, what it does to a trajectory a fifth of a span off the ground.
    """
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))

    axes[0].plot(sweep["h_over_b"], sweep["mu_l"], color="tab:blue", label="model: mu_l")
    axes[0].plot(sweep["h_over_b"], sweep["mu_d"] * sweep["mu_l"]**2, color="tab:red",
                 label="model: mu_d mu_l$^2$")
    axes[0].plot(sweep["h_over_b"], sweep["CL_ratio"], "o", ms=4, mfc="none",
                 color="tab:blue", label="measured: CL ratio")
    axes[0].plot(sweep["h_over_b"], sweep["CD_ratio"], "s", ms=4, mfc="none",
                 color="tab:red", label="measured: CD ratio")
    axes[0].axhline(1.0, color="0.6", lw=0.6)
    axes[0].set(xlabel="h / b", ylabel="FALCON-S / JSBSim")
    axes[0].set_title("ground-effect factors, model against measured")
    axes[0].legend(loc="center right")

    # Shade where the model is actually doing something. These airframes climb away, so ground
    # effect acts as an impulse early on and the offset it leaves is carried for the rest of the
    # run; without the band the panel looks like a disagreement that never stops growing.
    axes[1].axhspan(0.0, span, color="0.85", lw=0, label="ground effect: h/b < 1")
    axes[1].plot(ige["t"], ige["h_falcons"], label="FALCON-S, ground effect", **FALCONS_STYLE)
    axes[1].plot(ige["t"], ige["h_jsbsim"], label="JSBSim, none", **JSBSIM_STYLE)
    axes[1].set(xlabel="t [s]", ylabel="h [m]")
    axes[1].set_title(f"same release from h/b = {h_over_b:g}")
    # Opaque and clear of the shaded band, which the legend patch would otherwise vanish into.
    axes[1].legend(loc="upper left", framealpha=1.0, facecolor="white", edgecolor="0.8",
                   frameon=True)

    figure.tight_layout(pad=0.4)
    return _save(figure, path)


def figure_time_step(path: Path, ladder: list, plant_dt: float) -> Path:
    """Claim four: what limits the agreement is FALCON-S's own step, and it is first order.

    Errors are against the plant's own solution at a much finer step, so JSBSim is not involved
    and the slope is a statement about the integrator alone.
    """
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(1, 1, figsize=(3.4, 2.6))
    steps = np.array([row[0] for row in ladder]) * 1e3          # milliseconds, for legible ticks
    velocity = np.array([row[1] for row in ladder])
    altitude = np.abs([row[2] for row in ladder])
    axis.loglog(steps, velocity, "o-", color="tab:blue", ms=4, label="velocity [m/s]")
    axis.loglog(steps, altitude, "s-", color="tab:red", ms=4, label="altitude [m]")
    axis.loglog(steps, velocity[0] * steps / steps[0], color="0.5", lw=0.8, ls=(0, (4, 2)),
                label="first order")
    order = np.polyfit(np.log(steps), np.log(velocity), 1)[0]
    axis.set_xticks(steps)
    axis.set_xticklabels([f"{step:g}" for step in steps])
    axis.minorticks_off()
    axis.set(xlabel="FALCON-S time step [ms]", ylabel="error against dt/64 solution")
    axis.set_title(f"observed order {order:.2f}, configured step {plant_dt * 1e3:g} ms")
    axis.legend(loc="best")
    figure.tight_layout(pad=0.4)
    return _save(figure, path)


def plot_all(out: Path, plane: str, coefficients: pd.DataFrame, oge: pd.DataFrame,
             refined: pd.DataFrame, ige: pd.DataFrame, sweep: pd.DataFrame,
             ladder: list, config: dict, args) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_STYLE)

    plant_dt = config["environment_params"].get("dt", 0.01)
    span = config["vehicle_params"]["wing"]["span"]
    written = [
        figure_rollout(out / f"{plane}_fig1_rollout", oge, refined, plant_dt,
                       plant_dt / args.refine),
        figure_aero(out / f"{plane}_fig2_aero", coefficients, out, plane),
        figure_ground_effect(out / f"{plane}_fig3_ground_effect", sweep, ige, span,
                             args.ige_h_over_b),
    ]
    if ladder:
        written.append(figure_time_step(out / f"{plane}_fig4_time_step", ladder, plant_dt))
    if args.diagnostic_plots:
        written.append(figure_all_states(out / f"{plane}_diag_states_oge", plane, oge,
                                         "states out of ground effect"))
        written.append(figure_all_states(out / f"{plane}_diag_states_ige", plane, ige,
                                         "states in ground effect"))
    return written


# Smallest y range a diagnostic state panel is allowed, per unit, so that a channel both
# simulators hold at zero reads as a flat line instead of autoscaling its own machine noise.
AXIS_FLOOR = {"m": 2.0, "m/s": 1.0, "deg": 2.0, "deg/s": 2.0}


def figure_all_states(path: Path, plane: str, frame: pd.DataFrame, title: str) -> Path:
    """Every channel, for diagnosis rather than for a paper: --diagnostic-plots.

    This is the figure that shows what an unexpected number in the printed table came from, and
    it is how the lateral departure on the rotating earth was found in the first place.
    """
    import matplotlib.pyplot as plt
    horizon = trust_horizon(frame)
    steep = frame[frame["theta_falcons_deg"].abs() > 89.0]
    gimbal = float(steep["t"].iloc[0]) if len(steep) else np.nan
    figure, grid = plt.subplots(4, 4, figsize=(13, 9))
    figure.suptitle(f"{plane}: {title}")
    axes = grid.ravel()
    for axis, (label, stem, unit, is_angle) in zip(axes, CHANNELS):
        falcons, jsbsim = channel_columns(stem, unit)
        axis.plot(frame["t"], frame[falcons], label="FALCON-S", **FALCONS_STYLE)
        axis.plot(frame["t"], frame[jsbsim], label="JSBSim", **JSBSIM_STYLE)
        difference = frame[falcons] - frame[jsbsim]
        if is_angle:
            difference = wrap_deg(difference)
        axis.set_title(f"max |diff| {np.abs(difference).max():.2e} {unit}")
        low = min(frame[falcons].min(), frame[jsbsim].min())
        high = max(frame[falcons].max(), frame[jsbsim].max())
        floor = AXIS_FLOOR.get(unit, 0.0)
        if high - low < floor:
            middle = 0.5 * (high + low)
            axis.set_ylim(middle - floor / 2, middle + floor / 2)
        if np.isfinite(horizon):
            axis.axvline(horizon, color="0.55", ls=":", lw=1.0)
        if label in ("phi", "psi") and np.isfinite(gimbal):
            axis.axvline(gimbal, color="tab:red", ls="-.", lw=1.0)
        axis.set(xlabel="t [s]", ylabel=f"{label} [{unit}]")
    trajectory = axes[len(CHANNELS)]
    trajectory.plot(frame["track_falcons"], frame["h_falcons"], **FALCONS_STYLE)
    trajectory.plot(frame["track_jsbsim"], frame["h_jsbsim"], **JSBSIM_STYLE)
    trajectory.set(xlabel="ground track [m]", ylabel="h [m]")
    trajectory.set_title("vertical profile")
    for axis in axes[len(CHANNELS) + 1:]:
        axis.axis("off")
    axes[0].legend(loc="best")
    figure.tight_layout()
    return _save(figure, path)


if __name__ == "__main__":
    main()
