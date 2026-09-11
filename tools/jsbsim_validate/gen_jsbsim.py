#!/usr/bin/env python3
"""Write a JSBSim aircraft that carries a FALCON-S airframe's mass, inertia, geometry and
OpenVSP derivative aerodynamics, so the two simulators can fly the same aeroplane.

The aerodynamic model is linear in six deltas about one measured operating point, which maps
onto JSBSim arithmetic directly -- `<sum>` of `<product>` terms -- with no tables and no
interpolation anywhere:

    C_D = CD_Total + CD_Alpha*da + k_ind*(CL_Alpha*da)^2 + CD_elevator*de + CD_q*q_hat
    C_S = CS_Beta*beta  + CS_aileron*da_a  + CS_rudder*dr  + CS_p*p_hat + CS_r*r_hat
    C_L = CL_Total  + CL_Alpha*da  + CL_elevator*de  + CL_q*q_hat
    C_l = CMl_Beta*beta + CMl_aileron*da_a + CMl_rudder*dr + CMl_p*p_hat + CMl_r*r_hat
    C_m = CMm_Total + CMm_Alpha*da + CMm_elevator*de + CMm_q*q_hat
    C_n = CMn_Beta*beta + CMn_aileron*da_a + CMn_rudder*dr + CMn_p*p_hat + CMn_r*r_hat

    da = alpha - alpha_run,  de = delta_e - de_run      (the linearisation point)
    p_hat = p*b/(2V),  q_hat = q*c/(2V),  r_hat = r*b/(2V)

No interpolation means no interpolation error, so a coefficient sweep compares the two
implementations of the model rather than the accuracy of an export.

Coefficient values come from the same `DerivativeAeroParameters` the plant loads, so the XML
cannot drift from the model it represents.

Deliberately left out:

  * ground effect. JSBSim is the out-of-ground-effect reference that FALCON-S's measured
    ground-effect sweep is judged against, so giving JSBSim a ground-effect model of its own
    would defeat the comparison. The free-air set is what is written here.
  * actuator and engine dynamics. The validation holds every control at a commanded value and
    the throttle shut, and JSBSim's <flight_control> here is a bare command-to-radians gain.

Rate damping is NOT left out: CMl_p, CMm_q, CMn_r, CL_q, CD_q, CS_p and CS_r are members of the
measured set and every FALCON-S backend applies them, so JSBSim must too or the two aeroplanes
differ in pitch and roll damping.
"""

import argparse
from pathlib import Path

import numpy as np

from falcons.aircraft.config import AircraftConfig
from falcons.aircraft.derivatives import IDX, DerivativeAeroParameters
from falcons.sim.aero_contract import SURFACES, check_surface_order

# 1 slug*ft^2 = 14.5939029372 kg * 0.3048^2 m^2. Inertia is written to the XML already in
# slug*ft^2 because JSBSim's own KG*M2 conversion carries a ~9e-5 relative error, which lands
# straight on the angular rates this tool is trying to measure.
KGM2_PER_SLUGFT2 = 14.5939029372 * 0.3048**2

# JSBSim property carrying each deflection, in RADIANS. The <flight_control> block below builds
# these from the normalised commands, matching ActuatorLimits.scale_from_normalized.
CONTROL_PROPERTY = {"elevator": "fcs/elevator-rad",
                    "aileron": "fcs/aileron-rad",
                    "rudder": "fcs/rudder-rad"}
COMMAND_PROPERTY = {"elevator": "fcs/elevator-cmd-norm",
                    "aileron": "fcs/aileron-cmd-norm",
                    "rudder": "fcs/rudder-cmd-norm"}

# JSBSim axis -> the reference length its coefficient is non-dimensionalised on. FALCON-S uses
# span for roll and yaw and the mean aerodynamic chord for pitch, which is JSBSim's convention
# too, so the coefficients transfer unscaled.
AXIS_LENGTH = {"DRAG": None, "SIDE": None, "LIFT": None,
               "ROLL": "metrics/bw-ft", "PITCH": "metrics/cbarw-ft", "YAW": "metrics/bw-ft"}

# The six coefficients, each as (JSBSim axis, constant channel or None, [(channel, driver)...]).
# Written out rather than derived so that this file states the model it is exporting, and a
# reader can check it against the docstring above line by line.
COEFFICIENTS = [
    ("CD", "DRAG", "CD_Total", [("CD_Alpha", "da"), ("CD_elevator", "de"), ("CD_q", "q_hat")]),
    ("CS", "SIDE", None, [("CS_Beta", "beta"), ("CS_aileron", "aileron"),
                          ("CS_rudder", "rudder"), ("CS_p", "p_hat"), ("CS_r", "r_hat")]),
    ("CL", "LIFT", "CL_Total", [("CL_Alpha", "da"), ("CL_elevator", "de"), ("CL_q", "q_hat")]),
    ("Cl", "ROLL", None, [("CMl_Beta", "beta"), ("CMl_aileron", "aileron"),
                          ("CMl_rudder", "rudder"), ("CMl_p", "p_hat"), ("CMl_r", "r_hat")]),
    ("Cm", "PITCH", "CMm_Total", [("CMm_Alpha", "da"), ("CMm_elevator", "de"),
                                  ("CMm_q", "q_hat")]),
    ("Cn", "YAW", None, [("CMn_Beta", "beta"), ("CMn_aileron", "aileron"),
                         ("CMn_rudder", "rudder"), ("CMn_p", "p_hat"), ("CMn_r", "r_hat")]),
]


def num(x: float) -> str:
    """Enough digits to round-trip a float64 exactly, so the XML is not a second source of error."""
    return repr(float(x))


def drivers(params: DerivativeAeroParameters) -> dict:
    """The six deltas and three non-dimensional rates, as JSBSim expression fragments.

    `bi2vel` and `ci2vel` are JSBSim's own b/(2V) and c/(2V), so the non-dimensionalisation is
    JSBSim's rather than a reimplementation -- including its own low-airspeed guard, which is not
    FALCON-S's 0.1 m/s floor. The two only disagree below a few m/s, which is outside anything
    this validation flies, but it is the one place the transcription is not literal.
    """
    def delta(prop, ref):
        if ref == 0.0:
            return f"<property>{prop}</property>"
        return (f"<difference><property>{prop}</property>"
                f"<value>{num(ref)}</value></difference>")

    def rate(prop, i2vel):
        return (f"<product><property>{prop}</property>"
                f"<property>{i2vel}</property></product>")

    return {
        "da": delta("aero/alpha-rad", params.alpha_run),
        "de": delta(CONTROL_PROPERTY["elevator"], params.de_run),
        "beta": "<property>aero/beta-rad</property>",
        "aileron": f"<property>{CONTROL_PROPERTY['aileron']}</property>",
        "rudder": f"<property>{CONTROL_PROPERTY['rudder']}</property>",
        "p_hat": rate("velocities/p-aero-rad_sec", "aero/bi2vel"),
        "q_hat": rate("velocities/q-aero-rad_sec", "aero/ci2vel"),
        "r_hat": rate("velocities/r-aero-rad_sec", "aero/bi2vel"),
    }


def coefficient_xml(name, constant, terms, params, drv, indent="   ") -> str:
    """One coefficient as a <sum> of <product> terms."""
    free = params.free
    parts = []
    if constant is not None:
        parts.append(f"<value>{num(free[IDX[constant]])}</value>")
    for channel, driver in terms:
        gain = free[IDX[channel]]
        parts.append(f"<!-- {channel} -->\n{indent}   "
                     f"<product><value>{num(gain)}</value>{drv[driver]}</product>")
    if name == "CD":
        # The induced term, k_ind * (CL_Alpha * da)^2. k_ind is precomputed at load time from the
        # free-air set -- CD_Alpha / (2 * CL_Total * CL_Alpha) -- exactly as the plant does it.
        # This is the only non-linear term in the whole model.
        k, cla = params.k_ind_free, free[IDX["CL_Alpha"]]
        parts.append(
            f"<!-- induced: k_ind * (CL_Alpha * da)^2 -->\n{indent}   "
            f"<product><value>{num(k)}</value>"
            f"<pow><product><value>{num(cla)}</value>{drv['da']}</product>"
            f"<value>2.0</value></pow></product>")
    body = "\n".join(f"{indent}   {p}" for p in parts)
    return (f'{indent}<function name="aero/coeff/{name}">\n'
            f'{indent} <description>{name} from the OpenVSP derivative set</description>\n'
            f'{indent} <sum>\n{body}\n{indent} </sum>\n'
            f'{indent}</function>')


def axis_xml(name, axis) -> str:
    terms = ["aero/qbar-psf", "metrics/Sw-sqft"]
    if AXIS_LENGTH[axis]:
        terms.append(AXIS_LENGTH[axis])
    terms.append(f"aero/coeff/{name}")
    product = "\n".join(f"     <property>{t}</property>" for t in terms)
    return (f'  <axis name="{axis}">\n'
            f'   <function name="aero/{axis.lower()}">\n'
            f'    <product>\n{product}\n    </product>\n'
            f'   </function>\n'
            f'  </axis>')


def control_limits(config: dict) -> dict:
    """Deflection limit per surface, in RADIANS, asserted symmetric.

    ServoActuator scales a normalised command to [min_deflection, max_deflection], so a symmetric
    limit makes a zero command exactly zero and lets JSBSim reproduce the mapping with one gain.
    The JSON key is the airframe's own name for the surface ("ailerons"); `surface_type` is what
    identifies it, which is what `check_surface_order` matches on.
    """
    check_surface_order(config, name="gen_jsbsim")
    surfaces = config["vehicle_params"]["actuator_system"]["aero_surfaces"]
    by_type = {s["surface_type"]: s for s in surfaces.values()}
    limits = {}
    for surface in SURFACES:
        lo, hi = by_type[surface]["min_deflection"], by_type[surface]["max_deflection"]
        if abs(lo + hi) > 1e-12:
            raise SystemExit(f"{surface} limits {lo}..{hi} are not symmetric")
        limits[surface] = float(np.radians(hi))
    return limits


def build(plane: str) -> str:
    config = AircraftConfig(plane).load()
    vehicle = config["vehicle_params"]
    wing = vehicle["wing"]
    params = DerivativeAeroParameters.from_config(config)
    limits = control_limits(config)
    drv = drivers(params)

    functions = [coefficient_xml(n, c, t, params, drv) for n, _, c, t in
                 [(n, a, c, t) for n, a, c, t in COEFFICIENTS]]
    axes = [axis_xml(n, a) for n, a, _, _ in COEFFICIENTS]

    J = np.array(vehicle["inertia_matrix"], dtype=float) / KGM2_PER_SLUGFT2
    gains = "\n".join(
        f'    <pure_gain name="{CONTROL_PROPERTY[s]}">\n'
        f'     <input>{COMMAND_PROPERTY[s]}</input>\n'
        f'     <gain>{num(limits[s])}</gain>\n'
        f'    </pure_gain>'
        for s in SURFACES)

    run_point = (f"alpha = {np.degrees(params.alpha_run):.4f} deg, "
                 f"elevator = {np.degrees(params.de_run):.4f} deg, "
                 f"V = {params.v_ref} m/s")

    return f"""<?xml version="1.0"?>
<!-- Generated by tools/jsbsim_validate/gen_jsbsim.py from FALCON-S airframe {plane}.
     OpenVSP linear derivative set about {run_point}.
     Out of ground effect, no engine. Do not hand-edit; regenerate. -->
<fdm_config name="{plane}_falcons" version="2.0" release="ALPHA">

 <fileheader>
  <author>tools/jsbsim_validate/gen_jsbsim.py</author>
  <description>FALCON-S {plane}, out of ground effect</description>
 </fileheader>

 <metrics>
  <wingarea unit="M2">{wing['area']}</wingarea>
  <wingspan unit="M">{wing['span']}</wingspan>
  <chord unit="M">{wing['mac']}</chord>
  <!-- Aerodynamic reference point on the CG: FALCON-S applies aerodynamic forces at the CG,
       so any offset here would add a moment arm it does not have. -->
  <location name="AERORP" unit="M"><x>0.0</x><y>0.0</y><z>0.0</z></location>
 </metrics>

 <mass_balance>
  <!-- slug*ft^2, converted here rather than by JSBSim; see KGM2_PER_SLUGFT2 in the generator.
       JSBSim's default cross-product convention puts +ixz at J[0][2], which is the FALCON-S
       inertia matrix entry, so these transfer sign-for-sign. -->
  <ixx unit="SLUG*FT2">{J[0][0]:.10g}</ixx>
  <iyy unit="SLUG*FT2">{J[1][1]:.10g}</iyy>
  <izz unit="SLUG*FT2">{J[2][2]:.10g}</izz>
  <ixy unit="SLUG*FT2">{J[0][1]:.10g}</ixy>
  <ixz unit="SLUG*FT2">{J[0][2]:.10g}</ixz>
  <iyz unit="SLUG*FT2">{J[1][2]:.10g}</iyz>
  <emptywt unit="KG">{vehicle['mass']}</emptywt>
  <location name="CG" unit="M"><x>0.0</x><y>0.0</y><z>0.0</z></location>
 </mass_balance>

 <ground_reactions/>
 <propulsion/>

 <flight_control name="command to radians">
  <channel name="surfaces">
{gains}
  </channel>
 </flight_control>

 <aerodynamics>

{chr(10).join(functions)}

{chr(10).join(axes)}

 </aerodynamics>

</fdm_config>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plane", default="Navion", help="FALCON-S airframe name")
    parser.add_argument("--out", default=Path(__file__).parent / "aircraft", type=Path,
                        help="JSBSim aircraft directory to write into")
    args = parser.parse_args()

    xml = build(args.plane)
    directory = args.out / f"{args.plane}_falcons"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{args.plane}_falcons.xml"
    path.write_text(xml)
    print(f"wrote {path} ({path.stat().st_size / 1024:.1f} kB)")


if __name__ == "__main__":
    main()
