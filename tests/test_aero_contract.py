"""The aero input/output contract: the deflection packing, the degree/radian boundary, and the
guard that keeps an airframe JSON from silently reordering the action vector.

These are the cheap tests that would have caught the ordering conflict the contract exists to
resolve -- the repo packs [elevator, aileron, rudder], the OpenVSP model was specified against
[aileron, elevator, rudder].
"""
import json

import numpy as np
import pytest

from falcons.aircraft.config import PLANES, AircraftConfig
from falcons.sim.aero_contract import (
    INDEX,
    SURFACES,
    AeroCoefs,
    AeroInputs,
    aero_inputs,
    check_surface_order,
    deflections_to_radians,
)


def test_the_packing_is_elevator_aileron_rudder():
    """Load-bearing: every deflection array in the repo is in this order."""
    assert SURFACES == ("elevator", "aileron", "rudder")
    assert INDEX == {"elevator": 0, "aileron": 1, "rudder": 2}


def test_deflections_are_unpacked_by_name_not_position():
    """A vector with only the aileron set must land on the aileron and nowhere else."""
    d = deflections_to_radians([0.0, 10.0, 0.0])
    assert d["aileron"] == pytest.approx(np.radians(10.0))
    assert d["elevator"] == 0.0 and d["rudder"] == 0.0


def test_each_slot_reaches_its_own_surface():
    for name, i in INDEX.items():
        v = [0.0, 0.0, 0.0]
        v[i] = 7.0
        d = deflections_to_radians(v)
        assert d[name] == pytest.approx(np.radians(7.0))
        assert sum(abs(x) for k, x in d.items() if k != name) == 0.0


def test_degrees_become_radians_at_the_boundary():
    """Actuators emit degrees, the derivative set is per radian. 20 deg is a real elevator limit,
    so getting this wrong is a factor of 57 on the largest control input the plant sees."""
    d = deflections_to_radians([20.0, -15.0, 15.0])
    assert d["elevator"] == pytest.approx(0.3490658, abs=1e-7)
    assert d["aileron"] == pytest.approx(-0.2617994, abs=1e-7)
    assert d["rudder"] == pytest.approx(0.2617994, abs=1e-7)


def test_a_wrong_length_deflection_vector_raises():
    with pytest.raises(ValueError, match="expected 3 deflections"):
        deflections_to_radians([0.0, 0.0])


def test_aero_inputs_keeps_angles_and_rates_in_radians():
    """alpha/beta/rates arrive already in radians and must pass through untouched -- only the
    deflections get converted."""
    a = aero_inputs(alpha=0.05, beta=-0.01, v=15.0,
                    deflections_deg=[2.0, 0.0, 0.0], ang_vel=[0.1, 0.2, 0.3], h=3.0)
    assert (a.alpha, a.beta, a.v) == (0.05, -0.01, 15.0)
    assert (a.p, a.q, a.r) == (0.1, 0.2, 0.3)
    assert a.elevator == pytest.approx(np.radians(2.0))
    assert a.h == 3.0


def test_aero_inputs_carries_no_geometry():
    """Span and MAC are per-airframe, not per-step: they belong with the parameters."""
    assert not {"span", "b_w", "mac", "c_bar"} & set(AeroInputs._fields)


def test_coefs_vector_order_matches_the_specified_model():
    c = AeroCoefs(CD=1.0, CS=2.0, CL=3.0, Cl=4.0, Cm=5.0, Cn=6.0)
    assert np.array_equal(c.as_array(), np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert AeroCoefs._fields == ("CD", "CS", "CL", "Cl", "Cm", "Cn")


# ─── the surface-order guard

@pytest.mark.parametrize("ac", PLANES)
def test_every_shipped_airframe_is_in_contract_order(ac):
    check_surface_order(AircraftConfig(ac).load(), ac)


def test_a_reordered_airframe_raises(tmp_path):
    """Reordering the JSON block would drive the elevator with the aileron command."""
    cfg = AircraftConfig("Volantex_Ranger").load()
    surfaces = cfg["vehicle_params"]["actuator_system"]["aero_surfaces"]
    swapped = {k: surfaces[k] for k in ("ailerons", "elevator", "rudder")}
    cfg["vehicle_params"]["actuator_system"]["aero_surfaces"] = swapped
    with pytest.raises(ValueError, match="would be applied to the wrong surface"):
        check_surface_order(cfg, "Volantex_Ranger")


def test_the_guard_reads_surface_type_not_the_json_key():
    """A renamed key is harmless; the declared surface_type is what carries the meaning."""
    cfg = AircraftConfig("Volantex_Ranger").load()
    surfaces = cfg["vehicle_params"]["actuator_system"]["aero_surfaces"]
    renamed = {("aileron" if k == "ailerons" else k): v for k, v in surfaces.items()}
    cfg["vehicle_params"]["actuator_system"]["aero_surfaces"] = renamed
    check_surface_order(cfg, "Volantex_Ranger")
