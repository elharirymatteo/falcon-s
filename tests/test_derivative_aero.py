"""The OpenVSP derivative set as loaded: the operating point, the channel layout, and the
ground-effect anchoring that lets a clamped interpolation stand in for "free air above the table".

The aerodynamic model that consumes this arrives in Phase 3; these are the loader's invariants.
"""
import numpy as np
import pytest

from falcons.aircraft.config import AircraftConfig
from falcons.aircraft.params import CHANNELS, IDX, DerivativeAeroParameters
from conftest import ONLINE_PLANES as ONLINE

pytestmark = pytest.mark.skipif(not ONLINE, reason="no airframe has usable derivative CSVs yet")


def load(ac):
    return DerivativeAeroParameters.from_config(AircraftConfig(ac).load())


@pytest.fixture(params=ONLINE)
def ap(request):
    return load(request.param)


def test_channel_layout_is_unique_and_complete(ap):
    assert len(CHANNELS) == len(set(CHANNELS)) == 27
    assert ap.free.shape == (len(CHANNELS),)
    assert ap.ge_increment.shape == (len(ap.hc), len(CHANNELS))
    assert IDX["CD_Total"] == 0                      # layout is load-bearing for the warp port


def test_operating_point_is_radians(ap):
    """The CSV states the run point in degrees; nothing downstream should ever see degrees."""
    assert abs(ap.alpha_run) < np.pi / 2
    assert abs(ap.de_run) < np.pi / 2


def test_height_grid_is_strictly_increasing(ap):
    assert np.all(np.diff(ap.hc) > 0)
    assert ap.hc_floor == ap.hc[0]


# ─── the anchoring property. Ground effect is stored as an increment measured from the TOP of the
#     sweep, so `free + increment` equals the free-air set exactly there. That is what makes a
#     plain clamped interpolation correct above the table instead of needing a separate branch --
#     and what removes the step the raw table has at the hand-off.

def test_increment_vanishes_at_the_top_of_the_sweep(ap):
    assert np.array_equal(ap.ge_increment[-1], np.zeros(len(CHANNELS)))


def test_induced_drag_factor_is_continuous_with_free_air(ap):
    """k_ind is rebuilt from the ANCHORED coefficients, so its top row must be the free-air value
    exactly. Deriving it from the raw table instead would leave a step here."""
    assert ap.k_ind[-1] == ap.k_ind_free


def test_ground_effect_increases_lift_and_is_monotone(ap):
    """Physics sanity: lift rises as the ground is approached, monotonically."""
    cl = ap.ge_increment[:, IDX["CL_Total"]]
    assert cl[0] > 0                                  # deepest measured height gains lift
    assert np.all(np.diff(cl) < 0)                    # decaying to 0 as height grows


def test_induced_drag_factor_is_positive_everywhere(ap):
    assert np.all(ap.k_ind > 0) and ap.k_ind_free > 0


def test_reference_geometry_matches_the_json(ap, request):
    """AircraftConfig.load enforces this; asserted here so the two never drift apart in silence."""
    ac = request.node.callspec.params["ap"]
    wing = AircraftConfig(ac).load()["vehicle_params"]["wing"]
    assert (ap.span_ref, ap.area_ref, ap.mac_ref) == (wing["span"], wing["area"], wing["mac"])


def test_singular_induced_drag_factor_raises(tmp_path):
    """A set measured at zero lift has no k_ind. Fail at load, not as a NaN in flight."""
    src = AircraftConfig(ONLINE[0])
    rows = src.derivatives_path.read_text().replace("\r\n", "\n").splitlines()
    patched = [rows[0]] + [
        "CL_Total,0.0" if r.startswith("CL_Total,") else r for r in rows[1:]
    ]
    p = tmp_path / "derivatives.csv"
    p.write_text("\n".join(patched) + "\n")
    with pytest.raises(ValueError, match="singular"):
        DerivativeAeroParameters(derivatives_file=str(p),
                                 ge_derivatives_file=str(src.ge_derivatives_path))


def test_a_missing_coefficient_raises(tmp_path):
    src = AircraftConfig(ONLINE[0])
    rows = src.derivatives_path.read_text().replace("\r\n", "\n").splitlines()
    p = tmp_path / "derivatives.csv"
    p.write_text("\n".join([rows[0]] + [r for r in rows[1:] if not r.startswith("CMm_q,")]) + "\n")
    with pytest.raises(ValueError, match="CMm_q"):
        DerivativeAeroParameters(derivatives_file=str(p),
                                 ge_derivatives_file=str(src.ge_derivatives_path))


def test_the_cpu_path_loads_without_warp():
    """The derivative set and the CPU reference plant must import with no GPU stack present.

    `tools/jsbsim_validate` runs the CPU plant against JSBSim in a venv that has neither warp nor
    torch, and that independence is most of what the cross-check is worth -- a reference sharing a
    GPU stack with the thing it checks is a weaker reference. It is also why
    `falcons.aircraft.derivatives` is separate from `falcons.aircraft.params`: params.py defines
    the warp structs and imports warp at module scope.

    Run in a subprocess with `warp` poisoned, because warp is already imported in this one.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import sys
        class Blocked:
            def find_module(self, name, path=None):
                if name == "warp" or name.startswith("warp."):
                    raise ImportError("warp is blocked for this test")
        sys.meta_path.insert(0, Blocked())
        import falcons.aircraft.derivatives
        import falcons.sim.cpu.physics.aerodynamics
        import falcons.sim.cpu.aircraft
        assert "warp" not in sys.modules, "something pulled warp in anyway"
        print("OK")
    """)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0 and "OK" in r.stdout, (
        f"the CPU path cannot import without warp:\n{r.stdout}\n{r.stderr}")
