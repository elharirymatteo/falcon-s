#!/usr/bin/env python3
"""Write a JSBSim aircraft that carries a FALCON-S airframe's mass, inertia, geometry and
polynomial aerodynamics, so the two simulators can fly the same aeroplane.

Every FALCON-S coefficient is a cubic in exactly two variables — an angle and one control
deflection — which is what makes this a handful of 2-D JSBSim tables rather than an
interpolation problem:

    CD, CL, CMy   alpha x elevator
    CY, CMz       beta  x rudder
    CMx           beta  x aileron

Table values come from the plant's own PolynomialAerodynamics, so the tables cannot drift
from the model they represent.

Deliberately left out:

  * ground effect. JSBSim is the out-of-ground-effect reference that FALCON-S's ground-effect
    model is measured against, so giving JSBSim a ground-effect model of its own would defeat
    the comparison.
  * rate damping (Clp, Cmq, Cnr). The CPU plant does not apply it either. Note that the Warp
    plant does, so this aircraft is not a reference for that plant.
  * actuator and engine dynamics. The validation holds every control at zero and the throttle
    shut, and JSBSim's <flight_control> here is a bare command-to-degrees gain.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from falcons.aircraft.config import AircraftConfig

# ─────────────────────────────────────────────────────────────────────────────────────────────
# NOT PORTED YET to the OpenVSP derivative aero model.
#
# This tool exported the plant's POLYNOMIAL coefficients as JSBSim 2-D tables (alpha x control),
# so JSBSim and FALCON-S could fly the same aeroplane. The polynomial model is gone: the plant is
# now a linear derivative set about one VSPAERO operating point plus a measured ground-effect
# sweep, which does not fit the 2-D table layout `check_layout` enforces. A linear set maps onto
# JSBSim <function> blocks instead, which is a rewrite of the emitter rather than a tweak.
#
# See plan.md, Phase 4. Until then this tool refuses to run rather than exporting tables that no
# longer describe the simulator.
# ─────────────────────────────────────────────────────────────────────────────────────────────
_NOT_PORTED = (
    "tools/jsbsim_validate is not ported to the OpenVSP derivative aero model. It exported the "
    "polynomial coefficients as JSBSim 2-D tables; the plant no longer has a polynomial. "
    "Porting it means emitting <function> blocks for a linear derivative set -- see plan.md "
    "Phase 4."
)

# 1 slug*ft^2 = 14.5939029372 kg * 0.3048^2 m^2. Inertia is written to the XML already in
# slug*ft^2 because JSBSim's own KG*M2 conversion carries a ~9e-5 relative error, which lands
# straight on the angular rates this tool is trying to measure.
KGM2_PER_SLUGFT2 = 14.5939029372 * 0.3048**2

# coefficient -> (angle variable, control variable, JSBSim axis). Asserted against the CSV below.
LAYOUT = {
    "CD": ("alpha", "delta_e", "DRAG"),
    "CY": ("beta", "delta_r", "SIDE"),
    "CL": ("alpha", "delta_e", "LIFT"),
    "CMx": ("beta", "delta_a", "ROLL"),
    "CMy": ("alpha", "delta_e", "PITCH"),
    "CMz": ("beta", "delta_r", "YAW"),
}

# JSBSim axis -> the reference length its coefficient is non-dimensionalised on. FALCON-S uses
# span for roll and yaw and the mean aerodynamic chord for pitch, which is JSBSim's convention
# too, so the coefficients transfer unscaled.
AXIS_LENGTH = {"DRAG": None, "SIDE": None, "LIFT": None,
               "ROLL": "metrics/bw-ft", "PITCH": "metrics/cbarw-ft", "YAW": "metrics/bw-ft"}

# aero_action index each control sits at, matching PolynomialAerodynamics.get_coefficient.
CONTROL_INDEX = {"delta_e": 0, "delta_a": 1, "delta_r": 2}
CONTROL_PROPERTY = {"delta_e": "fcs/elevator-deg", "delta_a": "fcs/aileron-deg",
                    "delta_r": "fcs/rudder-deg"}
SURFACE_ORDER = ["elevator", "ailerons", "rudder"]


def check_layout(poly: pd.DataFrame) -> None:
    """Fail loudly if a coefficient depends on anything but the two variables LAYOUT claims."""
    for coef, (angle, control, _) in LAYOUT.items():
        rows = poly[poly[coef].abs() > 0]
        used = [v for v in ("alpha", "beta", "delta_e", "delta_a", "delta_r")
                if (rows[v] > 0).any()]
        if sorted(used) != sorted([angle, control]):
            raise SystemExit(
                f"{coef} depends on {used}, not on {[angle, control]}. This airframe's "
                f"polynomial does not fit in 2-D tables; the generator needs extending.")


def control_limits(config: dict) -> dict:
    """Deflection limit per control, in degrees, asserted symmetric.

    ServoActuator scales a normalised command to [min_deflection, max_deflection], so a
    symmetric limit makes a zero command exactly zero degrees and lets JSBSim reproduce the
    mapping with a single gain.
    """
    surfaces = config["vehicle_params"]["actuator_system"]["aero_surfaces"]
    if list(surfaces) != SURFACE_ORDER:
        raise SystemExit(f"expected surfaces {SURFACE_ORDER}, found {list(surfaces)}; the "
                         "aero_action index mapping in this script would be wrong")
    limits = {}
    for control, name in zip(["delta_e", "delta_a", "delta_r"], SURFACE_ORDER):
        lo, hi = surfaces[name]["min_deflection"], surfaces[name]["max_deflection"]
        if abs(lo + hi) > 1e-12:
            raise SystemExit(f"{name} limits {lo}..{hi} are not symmetric")
        limits[control] = float(hi)
    return limits


def coefficient_table(aero, coef: str, angle: str, control: str,
                      angles_deg: np.ndarray, controls_deg: np.ndarray) -> np.ndarray:
    """Evaluate one coefficient over the grid using the plant's own polynomial code."""
    index = CONTROL_INDEX[control]
    table = np.zeros((len(angles_deg), len(controls_deg)))
    for i, angle_deg in enumerate(angles_deg):
        alpha = np.radians(angle_deg) if angle == "alpha" else 0.0
        beta = np.radians(angle_deg) if angle == "beta" else 0.0
        for j, deflection in enumerate(controls_deg):
            action = np.zeros(3)
            action[index] = deflection
            table[i, j] = aero.get_coefficient(coef, alpha, beta, action)
    return table


def table_xml(angle: str, control: str, angles_deg, controls_deg, table, indent: str) -> str:
    header = "".join(f"{c:>16.6g}" for c in controls_deg)
    rows = [f"{indent}      {a:>10.4f}" + "".join(f"{v:>16.8e}" for v in table[i])
            for i, a in enumerate(angles_deg)]
    return "\n".join([
        f"{indent}<table>",
        f"{indent}  <independentVar lookup=\"row\">aero/{angle}-deg</independentVar>",
        f"{indent}  <independentVar lookup=\"column\">{CONTROL_PROPERTY[control]}</independentVar>",
        f"{indent}  <tableData>",
        f"{indent}      {'':>10}" + header,
        *rows,
        f"{indent}  </tableData>",
        f"{indent}</table>",
    ])


def build(plane: str, alpha_range: float, beta_range: float,
          angle_step: float, control_step: float) -> str:
    config = AircraftConfig(plane).load()
    vehicle = config["vehicle_params"]
    wing = vehicle["wing"]
    poly = pd.read_csv(config["aero_params"]["poly_params_file"])
    check_layout(poly)
    limits = control_limits(config)
    aero = PolynomialAerodynamics(config["aero_params"], vehicle,
                                  config["environment_params"], poly)

    def grid(half_range: float, step: float) -> np.ndarray:
        n = int(round(half_range / step))
        return np.linspace(-half_range, half_range, 2 * n + 1)       # includes 0

    angle_grid = {"alpha": grid(alpha_range, angle_step), "beta": grid(beta_range, angle_step)}
    J = np.array(vehicle["inertia_matrix"], dtype=float) / KGM2_PER_SLUGFT2

    functions, axes = [], []
    for coef, (angle, control, axis) in LAYOUT.items():
        angles_deg = angle_grid[angle]
        m = int(round(limits[control] / control_step))
        controls_deg = np.linspace(-limits[control], limits[control], 2 * m + 1)
        table = coefficient_table(aero, coef, angle, control, angles_deg, controls_deg)
        functions.append(
            f'  <function name="aero/coeff/{coef}">\n'
            f'   <description>{coef} from {plane}_poly.csv, {angle} x {control}</description>\n'
            + table_xml(angle, control, angles_deg, controls_deg, table, "   ") + "\n"
            f'  </function>')
        terms = ["aero/qbar-psf", "metrics/Sw-sqft"]
        if AXIS_LENGTH[axis]:
            terms.append(AXIS_LENGTH[axis])
        terms.append(f"aero/coeff/{coef}")
        product = "\n".join(f"     <property>{t}</property>" for t in terms)
        axes.append(
            f'  <axis name="{axis}">\n'
            f'   <function name="aero/{axis.lower()}">\n'
            f'    <product>\n{product}\n    </product>\n'
            f'   </function>\n'
            f'  </axis>')

    gains = "\n".join(
        f'    <pure_gain name="{CONTROL_PROPERTY[c]}">\n'
        f'     <input>fcs/{n}-cmd-norm</input>\n'
        f'     <gain>{limits[c]}</gain>\n'
        f'    </pure_gain>'
        for c, n in [("delta_e", "elevator"), ("delta_a", "aileron"), ("delta_r", "rudder")])

    return f"""<?xml version="1.0"?>
<!-- Generated by tools/jsbsim_validate/gen_jsbsim.py from FALCON-S airframe {plane}.
     Out of ground effect, no rate damping, no engine. Do not hand-edit; regenerate. -->
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

 <flight_control name="command to degrees">
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
    raise SystemExit(_NOT_PORTED)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plane", default="Navion", help="FALCON-S airframe name")
    parser.add_argument("--out", default=Path(__file__).parent / "aircraft", type=Path,
                        help="JSBSim aircraft directory to write into")
    parser.add_argument("--alpha-range", type=float, default=180.0,
                        help="alpha table half-range, degrees. Full circle by default: with the "
                             "controls at zero these airframes have no pitch trim and tumble, and "
                             "a table that stopped at stall would clamp where the polynomial keeps "
                             "going, inventing a divergence that is the table's fault")
    parser.add_argument("--beta-range", type=float, default=90.0,
                        help="beta table half-range, degrees. asin(v/Va) cannot exceed 90")
    parser.add_argument("--angle-step", type=float, default=1.0,
                        help="alpha and beta table step, degrees. The only axis that interpolates "
                             "in this validation, so check 1 of validate.py measures what it costs")
    parser.add_argument("--control-step", type=float, default=5.0,
                        help="deflection table step, degrees. Coarse on purpose: the controls sit "
                             "at zero throughout, and zero is a grid node, so this axis "
                             "contributes no interpolation error here")
    args = parser.parse_args()

    xml = build(args.plane, args.alpha_range, args.beta_range,
                args.angle_step, args.control_step)
    directory = args.out / f"{args.plane}_falcons"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{args.plane}_falcons.xml"
    path.write_text(xml)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
