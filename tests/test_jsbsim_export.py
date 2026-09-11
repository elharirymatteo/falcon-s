"""The JSBSim export must be the same aeroplane.

`tools/jsbsim_validate` exists to cross-check FALCON-S against an independent flight dynamics
model, and that argument only holds if the XML it writes carries the airframe faithfully. The
validation run itself needs JSBSim installed and takes minutes; this does not. It parses the
emitted `<function>` blocks and evaluates them against the same inputs the plant sees, which
catches the failure mode that actually matters here -- a dropped term, a flipped sign, a
degree/radian slip in the emitter -- at a cost the suite can afford on every commit.

What it cannot catch is a disagreement about what JSBSim does with a coefficient once it has
one: the axis conventions, the wind-to-body rotation, the reference lengths. That is check 2 of
`validate.py` and it needs the real JSBSim.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from conftest import ONLINE_PLANES
from falcons.aircraft.config import AircraftConfig
from falcons.aircraft.params import DerivativeAeroParameters
from falcons.sim.aero_contract import aero_inputs
from falcons.sim.cpu.physics.aerodynamics import DerivativeAerodynamics

# `tools/` is not a package and is not installed -- it is scripts that import falcons, not the
# other way round. Reaching it by path keeps it that way while still letting the emitter be
# tested; importing `gen_jsbsim` pulls in no JSBSim, only the XML it writes.
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "jsbsim_validate"))
import gen_jsbsim  # noqa: E402

pytestmark = pytest.mark.skipif(not ONLINE_PLANES, reason="no airframe has usable derivative CSVs")


def evaluate(node, props):
    """JSBSim's <function> arithmetic, enough of it to evaluate what gen_jsbsim emits.

    Written out rather than imported so that this is an INDEPENDENT reading of the XML. Sharing
    an evaluator with the emitter would make the test agree with itself.
    """
    tag = node.tag
    if tag == "value":
        return float(node.text)
    if tag == "property":
        return props[node.text.strip()]
    kids = [c for c in node if c.tag is not ET.Comment]
    if tag == "sum":
        return sum(evaluate(c, props) for c in kids)
    if tag == "product":
        out = 1.0
        for c in kids:
            out *= evaluate(c, props)
        return out
    if tag == "difference":
        out = evaluate(kids[0], props)
        for c in kids[1:]:
            out -= evaluate(c, props)
        return out
    if tag == "pow":
        return evaluate(kids[0], props) ** evaluate(kids[1], props)
    raise AssertionError(f"unhandled JSBSim element <{tag}> -- the evaluator needs extending")


def properties(u, span, mac):
    """The JSBSim properties the emitted blocks read, from one FALCON-S flight condition.

    `bi2vel`/`ci2vel` are JSBSim's b/(2V) and c/(2V). Cases here stay well above the plant's
    0.1 m/s airspeed floor so that the two definitions cannot differ -- the floor is the one
    place the transcription is deliberately not literal, and it is documented as such in
    `gen_jsbsim.drivers`.
    """
    return {
        "aero/alpha-rad": u.alpha,
        "aero/beta-rad": u.beta,
        "fcs/elevator-rad": u.elevator,
        "fcs/aileron-rad": u.aileron,
        "fcs/rudder-rad": u.rudder,
        "velocities/p-aero-rad_sec": u.p,
        "velocities/q-aero-rad_sec": u.q,
        "velocities/r-aero-rad_sec": u.r,
        "aero/bi2vel": span / (2.0 * u.v),
        "aero/ci2vel": mac / (2.0 * u.v),
    }


# Deliberately asymmetric and all-channels-live, so no term can cancel out of the comparison.
# Heights are far above every airframe's measured sweep: the XML carries the FREE-AIR set by
# design (JSBSim is the out-of-ground-effect reference), so the plant must be read with ground
# effect off to compare like with like.
CASES = [
    dict(alpha=0.07, beta=0.03, v=25.0, elevator=-0.08, aileron=0.05, rudder=-0.04,
         p=0.3, q=-0.2, r=0.15),
    dict(alpha=-0.04, beta=-0.06, v=40.0, elevator=0.12, aileron=-0.09, rudder=0.07,
         p=-0.6, q=0.45, r=-0.25),
    dict(alpha=0.0, beta=0.0, v=15.0, elevator=0.0, aileron=0.0, rudder=0.0,
         p=0.0, q=0.0, r=0.0),
]


def build(ac):
    cfg = AircraftConfig(ac).load()
    wing = cfg["vehicle_params"]["wing"]
    span, mac = float(wing["span"]), float(wing["mac"])
    plant = DerivativeAerodynamics(DerivativeAeroParameters.from_config(cfg),
                                   span=span, mac=mac, ge_enable=False)
    root = ET.fromstring(gen_jsbsim.build(ac))
    blocks = {f.get("name").rsplit("/", 1)[1]: f.find("sum")
              for f in root.iter("function") if f.get("name").startswith("aero/coeff/")}
    return plant, blocks, span, mac


@pytest.mark.parametrize("ac", ONLINE_PLANES)
def test_every_coefficient_is_exported(ac):
    _, blocks, _, _ = build(ac)
    assert set(blocks) == {"CD", "CS", "CL", "Cl", "Cm", "Cn"}


@pytest.mark.parametrize("ac", ONLINE_PLANES)
@pytest.mark.parametrize("case", range(len(CASES)))
def test_exported_xml_reproduces_the_plant(ac, case):
    """The whole point of the tool: same inputs, same coefficients, to round-off."""
    plant, blocks, span, mac = build(ac)
    u = aero_inputs(CASES[case]["alpha"], CASES[case]["beta"], CASES[case]["v"],
                    [CASES[case]["elevator"], CASES[case]["aileron"], CASES[case]["rudder"]],
                    [CASES[case]["p"], CASES[case]["q"], CASES[case]["r"]], h=1000.0)
    # aero_inputs converts degrees to radians; the case values are already radians, so feed the
    # deflections straight through instead.
    u = u._replace(elevator=CASES[case]["elevator"], aileron=CASES[case]["aileron"],
                   rudder=CASES[case]["rudder"])
    props = properties(u, span, mac)
    ref = plant.coefficients(u)
    for name, block in blocks.items():
        got = evaluate(block, props)
        assert got == pytest.approx(getattr(ref, name), rel=1e-12, abs=1e-14), (
            f"{ac} {name}: JSBSim export {got!r} != plant {getattr(ref, name)!r}")


@pytest.mark.parametrize("ac", ONLINE_PLANES)
def test_rate_damping_reaches_the_export(ac):
    """Rate damping used to be excluded from this export, because the polynomial model had none.
    The measured set has CMl_p, CMm_q, CMn_r, CL_q, CD_q, CS_p and CS_r and every backend applies
    them, so an export without them is a different aeroplane -- and one that would look correct
    in any check flown at zero rates."""
    plant, blocks, span, mac = build(ac)
    still = aero_inputs(0.05, 0.02, 30.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], h=1000.0)
    rolling = still._replace(p=0.5, q=-0.4, r=0.3)
    for name, block in blocks.items():
        a = evaluate(block, properties(still, span, mac))
        b = evaluate(block, properties(rolling, span, mac))
        assert a != b, f"{ac} {name} does not respond to body rates in the JSBSim export"


@pytest.mark.parametrize("ac", ONLINE_PLANES)
def test_the_xml_is_well_formed_and_carries_the_airframe(ac):
    root = ET.fromstring(gen_jsbsim.build(ac))
    cfg = AircraftConfig(ac).load()
    wing = cfg["vehicle_params"]["wing"]
    metrics = root.find("metrics")
    assert float(metrics.find("wingspan").text) == wing["span"]
    assert float(metrics.find("wingarea").text) == wing["area"]
    assert float(metrics.find("chord").text) == wing["mac"]
    assert float(root.find("mass_balance/emptywt").text) == cfg["vehicle_params"]["mass"]
    # Inertia goes out in slug*ft^2, converted here rather than by JSBSim; check the round trip.
    J = np.array(cfg["vehicle_params"]["inertia_matrix"], dtype=float)
    ixx = float(root.find("mass_balance/ixx").text) * gen_jsbsim.KGM2_PER_SLUGFT2
    ixz = float(root.find("mass_balance/ixz").text) * gen_jsbsim.KGM2_PER_SLUGFT2
    assert ixx == pytest.approx(J[0][0], rel=1e-9)
    assert ixz == pytest.approx(J[0][2], rel=1e-9)
