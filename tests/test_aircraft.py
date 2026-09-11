import json
from pathlib import Path
import numpy as np
import pytest
from falcons.aircraft.config import PLANES, AircraftConfig, read_derivatives
from falcons.aircraft.params import load_params
from conftest import OFFLINE_REASON, requires_derivatives

GOLD = Path(__file__).parent / "golden" / "configs"


def _todict(o):
    if hasattr(o, "__dict__"): return {k: _todict(v) for k, v in vars(o).items()}
    if isinstance(o, (list, tuple)): return [_todict(x) for x in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, dict): return {k: _todict(v) for k, v in o.items()}
    return o if isinstance(o, (int, float, str, bool, type(None))) else str(o)


def test_planes_are_the_five():
    assert PLANES == ["Airship_V7", "Airship_A0S", "Volantex_Ranger", "Navion"]


@pytest.mark.parametrize("ac", PLANES)
def test_files_present(ac):
    assert AircraftConfig(ac).json_path.exists()


def test_lqr_gains_are_gone_with_the_plant_they_were_fitted_to():
    """The shipped K_LQR tables linearised the POLYNOMIAL plant around trim. That plant no longer
    exists, so flying them would be flying gains designed for different aerodynamics. They were
    deleted rather than left in place to be picked up silently; the LQR controller is refactored
    against the derivative set in its own phase (plan.md Phase 6)."""
    for name in PLANES:
        c = AircraftConfig(name)
        assert c.lqr_gains_path is None and c.acquisition_gains_path is None


# The OpenVSP derivative data is being extracted per airframe and lands one airframe at a time.
# Tests keyed on it SKIP while it is absent rather than failing, so dropping the two CSVs into
# data/<plane>/ is the whole of bringing an airframe online -- no test edit required.
def _skip_without_derivatives(ac):
    return requires_derivatives(ac)


def _skip_without_csvs(ac):
    """Weaker than `_skip_without_derivatives`: the files merely have to be present. For checks
    that exist to DIAGNOSE a bad export, and so must still run when the export does not load."""
    return pytest.mark.skipif(not AircraftConfig(ac).has_derivatives,
                              reason=f"{ac}: no OpenVSP derivative CSVs")


@pytest.mark.parametrize("ac", [pytest.param(a, marks=_skip_without_derivatives(a)) for a in PLANES])
def test_derivative_files_present(ac):
    c = AircraftConfig(ac)
    assert c.derivatives_path.exists() and c.ge_derivatives_path.exists()


def test_every_airframe_is_online():
    """All four ship OpenVSP tables that actually build a model. This is the one test that FAILS
    rather than skips when an airframe's data is bad: everything else keyed on derivative data
    skips, so without this a broken export would read as a quiet green run.

    Two ways to go dark. The file is absent -- the extractor emits bare `derivatives.csv` and the
    airframe prefix has to be added on the way in. Or it loads and raises, which means the export
    is incomplete; the message says which row is missing."""
    broken = {n: r for n, r in OFFLINE_REASON.items() if r is not None}
    assert not broken, "; ".join(f"{n}: {r}" for n, r in broken.items())


# Absolute paths (machine-specific) and the bulk derivative tables are excluded from the archive
# comparison. The tables are the CSVs' own content -- duplicating 11x27 coefficients into a golden
# config would gain nothing that `test_derivative_aero.py` does not already check, and would go
# stale on every re-extraction. The scalars derived from the data (the operating point, the
# reference geometry, k_ind_free) ARE compared, because those are what a bad CSV would move.
ARCHIVE_EXCLUDES = ("derivatives_file", "ge_derivatives_file",
                    "free", "hc", "ge_increment", "k_ind")


@pytest.mark.parametrize("ac", [pytest.param(a, marks=_skip_without_derivatives(a)) for a in PLANES])
def test_params_match_archive(ac):
    gold = json.load(open(GOLD / f"{ac}.json"))
    got = _todict(load_params(ac))
    for d in (gold["params"], got):
        for k in ARCHIVE_EXCLUDES:
            d["aero_params"].pop(k, None)
    assert got == gold["params"]


# A config that omits a required key would inherit Airship_V7's value for it, so the loader
# rejects it. Motor topology stays optional by design.

def test_every_shipped_aircraft_is_complete():
    for name in PLANES:
        AircraftConfig(name).load()          # load() itself does not need the derivative CSVs


def _stage(name, tmp_path, mutate, with_derivatives=False):
    """A copy of one airframe's data directory with its JSON mutated. The derivative CSVs are
    copied only on request: without them `load` skips the drift check, which is what the
    unrelated raise-tests below want."""
    src = AircraftConfig(name)
    cfg = json.loads(src.json_path.read_text())
    mutate(cfg)
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / f"{name}.json").write_text(json.dumps(cfg))
    if with_derivatives:
        (d / f"{name}_derivatives.csv").write_text(src.derivatives_path.read_text())
        (d / f"{name}_ge_derivatives.csv").write_text(src.ge_derivatives_path.read_text())
    return AircraftConfig(name, data_dir=tmp_path)


def test_a_missing_wing_key_raises_rather_than_taking_v7_geometry(tmp_path):
    cfg = _stage("Navion", tmp_path, lambda c: c["vehicle_params"]["wing"].pop("span"))
    with pytest.raises(ValueError, match=r"wing\.span"):
        cfg.load()


def test_a_motor_missing_its_disc_area_raises(tmp_path):
    def drop(c):
        motors = c["vehicle_params"]["actuator_system"]["motors"]
        next(iter(motors.values()))["motor_propeller_data"].pop("Sp")
    with pytest.raises(ValueError, match="Sp"):
        _stage("Volantex_Ranger", tmp_path, drop).load()


# ─── the drift gate: the JSON is the reference, and the CSV records what the coefficients were
#     actually non-dimensionalised by. Disagreement is silent and small, so it raises.

def test_geometry_that_drifted_from_the_measured_reference_raises(tmp_path):
    def bend(c):
        c["vehicle_params"]["wing"]["mac"] = 0.157      # the pre-refactor rounded value
    with pytest.raises(ValueError, match=r"FC_Cref_"):
        _stage("Volantex_Ranger", tmp_path, bend, with_derivatives=True).load()


def test_every_duplicated_geometry_field_is_checked(tmp_path):
    """span/area/mac each have a CSV counterpart; none may be checked by accident."""
    for key, fc in [("span", "FC_Bref_"), ("area", "FC_Sref_"), ("mac", "FC_Cref_")]:
        cfg = _stage("Volantex_Ranger", tmp_path / key,
                     lambda c, k=key: c["vehicle_params"]["wing"].update({k: 1.2345}),
                     with_derivatives=True)
        with pytest.raises(ValueError, match=fc):
            cfg.load()


def test_an_airframe_without_derivative_data_still_loads(tmp_path):
    """The drift check cannot become a wall in front of airframes whose CSVs have not landed."""
    cfg = _stage("Navion", tmp_path, lambda c: None)
    assert not cfg.has_derivatives
    assert cfg.load()["aero_params"]["derivatives_file"].endswith("Navion_derivatives.csv")


def test_motor_topology_stays_optional():
    single = AircraftConfig("Volantex_Ranger").load()["vehicle_params"]["actuator_system"]["motors"]
    assert set(single) == {"centre_motor"}          # no left/right, and it still loads


@pytest.mark.parametrize("ac", [pytest.param(a, marks=_skip_without_csvs(a)) for a in PLANES])
def test_the_run_point_is_a_trim_point(ac):
    """Each derivative set is a linearisation about a condition the airframe can actually hold, so
    lift at that condition must equal weight.

    This is the cheapest cross-check there is on a fresh export: it ties mass, reference area,
    density, airspeed and CL_Total together in one number, and a table exported for a DIFFERENT
    aeroplane fails it by orders of magnitude rather than percent. One already did -- Navion
    briefly shipped Volantex's set and came out at L/W = 0.001.

    Use the run point's OWN density. `FC_Rho_` is not always 1.225: Navion's condition is 5000 ft
    (rho = 1.0555), and assuming sea level makes a trimmed set look 16% heavy.
    """
    d = read_derivatives(AircraftConfig(ac).derivatives_path)
    cfg = json.loads(AircraftConfig(ac).json_path.read_text())
    weight = cfg["vehicle_params"]["mass"] * 9.80665
    lift = 0.5 * d["FC_Rho_"] * d["FC_Vinf_"] ** 2 * d["FC_Sref_"] * d["CL_Total"]
    assert lift == pytest.approx(weight, rel=1e-3), (
        f"{ac}: the derivative set's run point is not a trim point -- lift {lift:.1f} N against "
        f"weight {weight:.1f} N (L/W = {lift / weight:.4f}). Either the table belongs to another "
        f"airframe, or the mass in the JSON is not the mass it was flown at.")


def test_rate_damping_is_not_a_config_knob_any_more():
    """Clp/Cmq/Cnr were JSON scalars because the polynomial model had no rate derivatives. The
    OpenVSP set measures CMl_p, CMm_q and CMn_r, so a JSON carrying the old keys would be
    double-counting -- they must be gone from every airframe, not merely ignored."""
    for name in PLANES:
        aero = AircraftConfig(name).load()["aero_params"]
        assert not {"Clp", "Cmq", "Cnr"} & set(aero), f"{name} still declares rate damping"
