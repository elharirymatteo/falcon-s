"""Every backend must compute the same coefficients.

The aerodynamic model exists four times -- numpy (the CPU reference plant), warp, torch, and
through warp again in the MPPI rollout kernels. Four transcriptions of one equation set is four
chances to drop a term, and nothing else in the suite compares them: `test_parity.py` covers only
the Dryden wind model, and notes there that the angular gusts are not even fed to the aero model.

The CPU implementation is the reference. Warp runs in float32, so it is compared at float32
tolerance; torch is driven in float64 where it can be.

Every airframe with usable data is exercised, not just one. The rate terms non-dimensionalise by
span on p/r and by MAC on q, and with a single airframe a span/MAC mix-up is a constant factor
that agrees across all four backends and so goes unseen. The airframes here differ by 3x in span
(Volantex 1.62 m, Airship_V7 5.07 m) and 4.6x in MAC, which separates the two.
"""
import numpy as np
import pytest

from conftest import ONLINE_PLANES
from falcons.aircraft.config import AircraftConfig
from falcons.aircraft.params import DerivativeAeroParameters
from falcons.sim.aero_contract import AeroInputs
from falcons.sim.cpu.physics.aerodynamics import DerivativeAerodynamics

pytestmark = pytest.mark.skipif(not ONLINE_PLANES,
                                reason="no airframe has usable OpenVSP derivative CSVs")


def build_cpu(ac, ge_enable=True):
    cfg = AircraftConfig(ac).load()
    wing = cfg["vehicle_params"]["wing"]
    return DerivativeAerodynamics(DerivativeAeroParameters.from_config(cfg),
                                  span=wing["span"], mac=wing["mac"], ge_enable=ge_enable)


def geometry(ac):
    wing = AircraftConfig(ac).load()["vehicle_params"]["wing"]
    return float(wing["span"]), float(wing["mac"])


# A spread of conditions: in deep ground effect, mid-sweep, far above the table, at the run point,
# and with every control and rate channel excited so no term can be silently dropped.
#
# Heights are h/b, not metres, because the sweep is measured on an h/b grid running 0.20 -> 2.00.
# A fixed metre height would sit deep in ground effect for Volantex and out of it for Airship_V7,
# so the cases would stop meaning what their names say as soon as the airframe changed.
#
# (alpha, beta, v, elevator, aileron, rudder, p, q, r, h_over_b, label)
RAW_CASES = [
    (None, 0.0, None, None, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, "run point, far out of GE"),
    (0.05, 0.02, 15.0, 0.1, -0.05, 0.03, 0.4, -0.3, 0.2, 0.216, "deep GE, all channels live"),
    (-0.03, -0.04, 22.0, -0.2, 0.1, -0.08, -0.5, 0.6, -0.4, 0.617, "mid-sweep"),
    (0.12, 0.01, 8.0, 0.3, 0.02, 0.01, 0.1, 0.2, 0.05, 2.0, "exactly the table top"),
    (0.02, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0617, "below the V floor and the GE floor"),
]
IDS = [c[-1] for c in RAW_CASES]


def cases_for(ac):
    """The case table resolved against one airframe: heights scaled by its span, and the `None`
    fields of case 0 filled from its own measured operating point."""
    span, _ = geometry(ac)
    p = DerivativeAeroParameters.from_config(AircraftConfig(ac).load())
    out = []
    for alpha, beta, v, elev, ail, rud, pp, q, r, hb, _ in RAW_CASES:
        out.append(AeroInputs(
            alpha=p.alpha_run if alpha is None else alpha, beta=beta,
            v=p.v_ref if v is None else v,
            elevator=p.de_run if elev is None else elev, aileron=ail, rudder=rud,
            p=pp, q=q, r=r, h=hb * span))
    return out


@pytest.mark.parametrize("ac", ONLINE_PLANES)
@pytest.mark.parametrize("ge", [True, False])
@pytest.mark.parametrize("case", range(len(RAW_CASES)), ids=IDS)
def test_cpu_model_is_finite_and_ordered(ac, case, ge):
    """Guards the reference itself: no NaN from the V floor or the table clamps."""
    c = build_cpu(ac, ge_enable=ge).coefficients(cases_for(ac)[case])
    assert np.all(np.isfinite(c.as_array()))


@pytest.mark.parametrize("ac", ONLINE_PLANES)
def test_free_air_is_reached_above_the_table(ac):
    """The anchoring claim, end to end: at and above the sweep's top the in-ground-effect model
    must return exactly what the free-air model returns."""
    ige, oge = build_cpu(ac, True), build_cpu(ac, False)
    span, mac = geometry(ac)
    base = cases_for(ac)[3]
    top = base._replace(h=ige.p.hc[-1] * mac)        # the top row itself
    np.testing.assert_allclose(ige.coefficients(top).as_array(),
                               oge.coefficients(top).as_array(), rtol=0, atol=1e-12)
    far = base._replace(h=100.0 * span)              # far above: clamped to the same row
    np.testing.assert_allclose(ige.coefficients(far).as_array(),
                               oge.coefficients(far).as_array(), rtol=0, atol=1e-12)


@pytest.mark.parametrize("ac", ONLINE_PLANES)
def test_ground_effect_is_off_below_the_table_floor_but_held(ac):
    """Below the sweep the increment freezes rather than extrapolating."""
    m = build_cpu(ac, True)
    base = cases_for(ac)[1]
    at_floor = base._replace(h=m.p.hc_floor * m.mac)
    below = base._replace(h=0.0)
    np.testing.assert_allclose(m.coefficients(below).as_array(),
                               m.coefficients(at_floor).as_array(), rtol=0, atol=1e-12)


@pytest.mark.parametrize("ac", ONLINE_PLANES)
def test_ground_effect_raises_lift_near_the_ground(ac):
    m, f = build_cpu(ac, True), build_cpu(ac, False)
    u = cases_for(ac)[1]
    assert m.coefficients(u).CL > f.coefficients(u).CL


# ─── warp

@pytest.mark.cuda
@pytest.mark.parametrize("ac", ONLINE_PLANES)
@pytest.mark.parametrize("ge", [True, False])
def test_warp_matches_the_cpu_reference(ac, ge):
    import warp as wp

    from falcons.aircraft.params import AerodynamicsParametersStruct
    from falcons.sim.warp.aerodynamics import compute_all_coeffs

    wp.init()
    cfg = AircraftConfig(ac).load()
    params = DerivativeAeroParameters.from_config(cfg)
    AP = params.as_warp_struct()
    span, mac = geometry(ac)
    cases = cases_for(ac)

    n = len(cases)
    fields = ["alpha", "beta", "v", "elevator", "aileron", "rudder", "p", "q", "r", "h"]
    ins = {f: wp.array(np.array([getattr(c, f) for c in cases], dtype=np.float32),
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

    ref = np.array([build_cpu(ac, ge_enable=ge).coefficients(c).as_array() for c in cases])
    np.testing.assert_allclose(out.numpy(), ref, rtol=2e-5, atol=2e-7)


# ─── torch

@pytest.mark.parametrize("ac", ONLINE_PLANES)
def test_torch_matches_the_cpu_reference(ac):
    import torch

    from falcons.sim.torch.altitude import torch_coefficients

    cfg = AircraftConfig(ac).load()
    params = DerivativeAeroParameters.from_config(cfg)
    span, mac = geometry(ac)
    cases = cases_for(ac)

    for ge in (True, False):
        cols = {f: torch.tensor([getattr(c, f) for c in cases], dtype=torch.float64)
                for f in AeroInputs._fields}
        got = torch_coefficients(params, span, mac,
                                 ge_enable=ge, device="cpu", dtype=torch.float64, **cols)
        ref = np.array([build_cpu(ac, ge_enable=ge).coefficients(c).as_array() for c in cases])
        np.testing.assert_allclose(np.stack([g.numpy() for g in got], axis=-1), ref,
                                   rtol=1e-11, atol=1e-13)
