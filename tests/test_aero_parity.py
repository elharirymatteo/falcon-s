"""Every backend must compute the same coefficients.

The aerodynamic model exists four times -- numpy (the CPU reference plant), warp, torch, and
through warp again in the MPPI rollout kernels. Four transcriptions of one equation set is four
chances to drop a term, and nothing else in the suite compares them: `test_parity.py` covers only
the Dryden wind model, and notes there that the angular gusts are not even fed to the aero model.

The CPU implementation is the reference. Warp runs in float32, so it is compared at float32
tolerance; torch is driven in float64 where it can be.
"""
import numpy as np
import pytest

from falcons.aircraft.config import AircraftConfig
from falcons.aircraft.params import DerivativeAeroParameters
from falcons.sim.aero_contract import AeroInputs
from falcons.sim.cpu.physics.aerodynamics import DerivativeAerodynamics

AC = "Volantex_Ranger"
pytestmark = pytest.mark.skipif(not AircraftConfig(AC).has_derivatives,
                                reason=f"{AC}: no OpenVSP derivative CSVs")


def build_cpu(ge_enable=True):
    cfg = AircraftConfig(AC).load()
    wing = cfg["vehicle_params"]["wing"]
    return DerivativeAerodynamics(DerivativeAeroParameters.from_config(cfg),
                                  span=wing["span"], mac=wing["mac"], ge_enable=ge_enable)


# A spread of conditions: in deep ground effect, mid-sweep, far above the table, at the run point,
# and with every control and rate channel excited so no term can be silently dropped.
CASES = [
    AeroInputs(alpha=0.0683, beta=0.0, v=10.875, elevator=-0.0776, aileron=0.0, rudder=0.0,
               p=0.0, q=0.0, r=0.0, h=100.0),                       # the run point, out of GE
    AeroInputs(alpha=0.05, beta=0.02, v=15.0, elevator=0.1, aileron=-0.05, rudder=0.03,
               p=0.4, q=-0.3, r=0.2, h=0.35),                        # deep GE, all channels live
    AeroInputs(alpha=-0.03, beta=-0.04, v=22.0, elevator=-0.2, aileron=0.1, rudder=-0.08,
               p=-0.5, q=0.6, r=-0.4, h=1.0),                        # mid-sweep
    AeroInputs(alpha=0.12, beta=0.01, v=8.0, elevator=0.3, aileron=0.02, rudder=0.01,
               p=0.1, q=0.2, r=0.05, h=3.24),                        # exactly the table top
    AeroInputs(alpha=0.02, beta=0.0, v=0.05, elevator=0.0, aileron=0.0, rudder=0.0,
               p=1.0, q=1.0, r=1.0, h=0.1),                          # below the V floor and the GE floor
]


@pytest.mark.parametrize("ge", [True, False])
@pytest.mark.parametrize("case", range(len(CASES)))
def test_cpu_model_is_finite_and_ordered(case, ge):
    """Guards the reference itself: no NaN from the V floor or the table clamps."""
    c = build_cpu(ge_enable=ge).coefficients(CASES[case])
    assert np.all(np.isfinite(c.as_array()))


def test_free_air_is_reached_above_the_table():
    """The anchoring claim, end to end: at and above the sweep's top the in-ground-effect model
    must return exactly what the free-air model returns."""
    ige, oge = build_cpu(True), build_cpu(False)
    top = CASES[3]._replace(h=3.24)                  # h/c = 20.68, the top row
    np.testing.assert_allclose(ige.coefficients(top).as_array(),
                               oge.coefficients(top).as_array(), rtol=0, atol=1e-12)
    far = CASES[3]._replace(h=100.0)                 # far above: clamped to the same row
    np.testing.assert_allclose(ige.coefficients(far).as_array(),
                               oge.coefficients(far).as_array(), rtol=0, atol=1e-12)


def test_ground_effect_is_off_below_the_table_floor_but_held():
    """Below the sweep the increment freezes rather than extrapolating."""
    m = build_cpu(True)
    at_floor = CASES[1]._replace(h=m.p.hc_floor * m.mac)
    below = CASES[1]._replace(h=0.0)
    np.testing.assert_allclose(m.coefficients(below).as_array(),
                               m.coefficients(at_floor).as_array(), rtol=0, atol=1e-12)


def test_ground_effect_raises_lift_near_the_ground():
    m, f = build_cpu(True), build_cpu(False)
    u = CASES[1]
    assert m.coefficients(u).CL > f.coefficients(u).CL


# ─── warp

@pytest.mark.cuda
@pytest.mark.parametrize("ge", [True, False])
def test_warp_matches_the_cpu_reference(ge):
    import warp as wp

    from falcons.aircraft.params import AerodynamicsParametersStruct
    from falcons.sim.warp.aerodynamics import compute_all_coeffs

    wp.init()
    cfg = AircraftConfig(AC).load()
    wing = cfg["vehicle_params"]["wing"]
    params = DerivativeAeroParameters.from_config(cfg)
    AP = params.as_warp_struct()
    span, mac = float(wing["span"]), float(wing["mac"])

    n = len(CASES)
    fields = ["alpha", "beta", "v", "elevator", "aileron", "rudder", "p", "q", "r", "h"]
    ins = {f: wp.array(np.array([getattr(c, f) for c in CASES], dtype=np.float32),
                       dtype=wp.float32, device="cuda") for f in fields}
    out = wp.zeros((n, 6), dtype=wp.float32, device="cuda")

    @wp.kernel
    def run(AP: AerodynamicsParametersStruct,
            span: wp.float32, mac: wp.float32,
            alpha: wp.array(dtype=wp.float32), beta: wp.array(dtype=wp.float32),
            v: wp.array(dtype=wp.float32), elevator: wp.array(dtype=wp.float32),
            aileron: wp.array(dtype=wp.float32), rudder: wp.array(dtype=wp.float32),
            p: wp.array(dtype=wp.float32), q: wp.array(dtype=wp.float32),
            r: wp.array(dtype=wp.float32), h: wp.array(dtype=wp.float32),
            ge_enable: bool, out: wp.array2d(dtype=wp.float32)):
        i = wp.tid()
        C_D, C_S, C_L, Cl, Cm, Cn, _cdf, _clf = compute_all_coeffs(
            AP, span, mac, alpha[i], beta[i], v[i],
            elevator[i], aileron[i], rudder[i], p[i], q[i], r[i], h[i], ge_enable)
        out[i, 0] = C_D
        out[i, 1] = C_S
        out[i, 2] = C_L
        out[i, 3] = Cl
        out[i, 4] = Cm
        out[i, 5] = Cn

    wp.launch(run, dim=n,
              inputs=[AP, span, mac] + [ins[f] for f in fields] + [ge, out],
              device="cuda")
    wp.synchronize()

    ref = np.array([build_cpu(ge_enable=ge).coefficients(c).as_array() for c in CASES])
    np.testing.assert_allclose(out.numpy(), ref, rtol=2e-5, atol=2e-7)


# ─── torch

def test_torch_matches_the_cpu_reference():
    import torch

    from falcons.sim.torch.altitude import torch_coefficients

    cfg = AircraftConfig(AC).load()
    wing = cfg["vehicle_params"]["wing"]
    params = DerivativeAeroParameters.from_config(cfg)

    for ge in (True, False):
        cols = {f: torch.tensor([getattr(c, f) for c in CASES], dtype=torch.float64)
                for f in AeroInputs._fields}
        got = torch_coefficients(params, float(wing["span"]), float(wing["mac"]),
                                 ge_enable=ge, device="cpu", dtype=torch.float64, **cols)
        ref = np.array([build_cpu(ge_enable=ge).coefficients(c).as_array() for c in CASES])
        np.testing.assert_allclose(np.stack([g.numpy() for g in got], axis=-1), ref,
                                   rtol=1e-11, atol=1e-13)
