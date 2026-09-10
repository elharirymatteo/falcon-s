"""What the aerodynamic model is handed and what it returns, shared by every backend.

The CPU, warp, torch and MPPI plants each grew their own way of feeding the coefficient model,
and the orderings did not agree: this repo has always packed control deflections as
[elevator, aileron, rudder], while the OpenVSP derivative model was specified against
[aileron, elevator, rudder]. A positional 3-vector cannot tell those apart, and swapping
elevator with aileron produces a plant that still flies -- badly, and for reasons that look
aerodynamic rather than clerical. So the packing lives here, once, named and tested, and the
model itself only ever sees fields.

Units are the other half of the contract. Actuators run in DEGREES (see
`ActuatorLimits.scale_from_normalized`, driven by the min/max_deflection in each airframe JSON)
and the derivative set is per RADIAN. That conversion happens exactly here, at the boundary, so
no model or kernel carries a deg2rad of its own.
"""
from typing import Any, Dict, NamedTuple

import numpy as np

# Positional order of the aero action vector, and so of every deflection array in the codebase.
# Airframe JSONs must list their `aero_surfaces` to match; `check_surface_order` enforces it
# rather than trusting dict ordering.
SURFACES = ("elevator", "aileron", "rudder")
INDEX = {name: i for i, name in enumerate(SURFACES)}


class AeroInputs(NamedTuple):
    """One flight condition, in SI and radians, as the coefficient model wants it.

    Geometry (span, MAC) is deliberately absent: it is per-airframe, not per-step, so it belongs
    with the parameters. `h` is height above ground, positive up -- callers convert from NED
    themselves, since only they know the state layout.
    """
    alpha: float        # angle of attack [rad]
    beta: float         # sideslip [rad]
    v: float            # airspeed [m/s]
    elevator: float     # deflection [rad]
    aileron: float      # deflection [rad]
    rudder: float       # deflection [rad]
    p: float            # roll rate [rad/s]
    q: float            # pitch rate [rad/s]
    r: float            # yaw rate [rad/s]
    h: float            # height above ground [m], positive up


class AeroCoefs(NamedTuple):
    """Force and moment coefficients in the model's own axes.

    CD/CS/CL are wind-axis force coefficients (drag, side force, lift); Cl/Cm/Cn are body-axis
    moment coefficients (roll, pitch, yaw). Named to match the OpenVSP columns they come from,
    with the repo's `CY` spelled `CS` as the derivative CSVs spell it.
    """
    CD: float
    CS: float
    CL: float
    Cl: float
    Cm: float
    Cn: float

    def as_array(self) -> np.ndarray:
        """[C_D, C_S, C_L, C_l, C_m, C_n] -- the vector form the model was specified as."""
        return np.array(self, dtype=np.float64)


def deflections_to_radians(deflections_deg) -> Dict[str, float]:
    """A positional deflection vector in degrees -> named deflections in radians.

    This is the only place the repo's [elevator, aileron, rudder] packing is interpreted.
    """
    if len(deflections_deg) != len(SURFACES):
        raise ValueError(
            f"expected {len(SURFACES)} deflections ordered {SURFACES}, got {len(deflections_deg)}"
        )
    rad = np.radians(np.asarray(deflections_deg, dtype=np.float64))
    return {name: float(rad[i]) for name, i in INDEX.items()}


def aero_inputs(alpha, beta, v, deflections_deg, ang_vel, h) -> AeroInputs:
    """Assemble `AeroInputs` from the pieces a plant has to hand.

    `alpha`, `beta` in radians and `ang_vel` in rad/s, as every plant already computes them;
    `deflections_deg` positional per `SURFACES`, in degrees, as the actuators emit them.
    """
    d = deflections_to_radians(deflections_deg)
    return AeroInputs(
        alpha=float(alpha), beta=float(beta), v=float(v),
        elevator=d["elevator"], aileron=d["aileron"], rudder=d["rudder"],
        p=float(ang_vel[0]), q=float(ang_vel[1]), r=float(ang_vel[2]),
        h=float(h),
    )


def check_surface_order(config: Dict[str, Any], name: str = "config") -> None:
    """Refuse an airframe whose control surfaces are not in `SURFACES` order.

    The action vector's meaning comes from this ordering, and nothing downstream re-checks it: a
    JSON listing ailerons before the elevator would drive the elevator with the aileron command
    and never say so. Matched on each surface's declared `surface_type`, not on the JSON key,
    so a renamed key ("ailerons" vs "aileron") is fine but a reordered one is not.
    """
    surfaces = config["vehicle_params"]["actuator_system"]["aero_surfaces"]
    declared = tuple(spec.get("surface_type") for spec in surfaces.values())
    if declared != SURFACES:
        raise ValueError(
            f"{name}: aero_surfaces are ordered {declared} but the action vector is packed "
            f"{SURFACES}; every deflection would be applied to the wrong surface. Reorder the "
            f"aero_surfaces block in the airframe JSON."
        )
