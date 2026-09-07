"""Piston-engine trim thrust regression (Cirrus_SR22, Navion).

Both are single-motor piston GA aircraft whose JSON `motor_propeller_data` originally
omitted k_m/C_p/Sp, so `PropulsionParameters.from_config` fell back to the airship
copy-paste default k_m=37.42. Their trim speeds (94/75 m/s) exceed 37.42, so the Beard
slipstream Vd = Va + thr*(k_m - Va) fell BELOW Va at full throttle: the prop windmilled
as a brake (Cirrus -235 N, Navion -155 N) and neither could hold level flight in the
warp sim.

Fitted k_m/C_p/Sp from real prop geometry (78" Hartzell / 84" 2-blade) and rated power
(310 hp / 240 hp) via momentum-disk static thrust + power-available cruise thrust.

Pure-config test (no CUDA): reproduces the sim's exact per-engine formula against the
loaded PropulsionParametersStruct. Guards against the shared 37.42 default returning and
the negative-thrust brake regressing.
"""
import numpy as np
import pytest

from falcons.aircraft.params import load_params

rho = 1.225  # sea-level density used by the warp thrust kernel at the low sim altitudes

# (aircraft, trim Va [m/s] from default_initial_state, cruise drag [N] from poly aero,
#  n_model_thrusters) -- single centreline motor is split across 2 model thrusters,
#  each carrying half the disc area (PP.Sp is already the halved per-thruster area).
CASES = [
    ("Cirrus_SR22", 94.2, 856.0),
    ("Navion", 75.0, 951.0),
]


def _thrust(pp, Va, throttle):
    """Total thrust = 2 model thrusters, matching compute_thrust_forces_and_moments."""
    Vd = Va + throttle * (pp.k_m - Va)
    return 2.0 * 0.5 * rho * pp.Sp * pp.C_p * Vd * (Vd - Va)


@pytest.mark.parametrize("ac,Va,drag", CASES)
def test_km_not_the_shared_airship_default(ac, Va, drag):
    pp = load_params(ac)["vehicle_params"].PP
    assert abs(pp.k_m - 37.42) > 1.0, (
        f"{ac} k_m reverted to the shared airship default (37.42); at trim {Va} m/s "
        "the prop windmills as a brake and the plane cannot fly")


@pytest.mark.parametrize("ac,Va,drag", CASES)
def test_full_throttle_thrust_is_positive_at_trim(ac, Va, drag):
    """The core bug: full-throttle thrust must be positive (was negative) and beat drag."""
    pp = load_params(ac)["vehicle_params"].PP
    assert pp.k_m > Va, f"{ac} k_m ({pp.k_m}) <= trim Va ({Va}): prop brakes at cruise"
    T_full = _thrust(pp, Va, 1.0)
    assert T_full > drag, (
        f"{ac} full-throttle thrust {T_full:.0f} N does not exceed cruise drag "
        f"{drag:.0f} N -- no climb margin")


@pytest.mark.parametrize("ac,Va,drag", CASES)
def test_cruise_trim_throttle_in_range(ac, Va, drag):
    """A throttle in (0,1) must hold level cruise (thrust == drag), i.e. real trim exists."""
    pp = load_params(ac)["vehicle_params"].PP
    thr = np.linspace(0.0, 1.0, 101)
    T = np.array([_thrust(pp, Va, t) for t in thr])
    hold = thr[np.argmin(np.abs(T - drag))]
    assert 0.05 < hold < 0.95, (
        f"{ac} cruise trim throttle {hold:.2f} not in (0.05,0.95) -- mis-fit thrust scale")


def test_v7_km_not_the_shared_default():
    """k_m must be V7-specific, not the copy-paste 37.42 shared across the fleet."""
    from falcons.aircraft.params import load_params
    pp = load_params("Airship_V7")["vehicle_params"].PP
    assert abs(pp.k_m - 37.42) > 1.0, (
        f"V7 k_m reverted to the shared copy-paste default ({pp.k_m}); "
        "28 m/s cruise is unreachable at this value")
