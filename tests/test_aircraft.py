import json
from pathlib import Path
import numpy as np
import pytest
from falcons.aircraft.config import PLANES, AircraftConfig
from falcons.aircraft.params import load_params

GOLD = Path(__file__).parent / "golden" / "configs"


def _todict(o):
    if hasattr(o, "__dict__"): return {k: _todict(v) for k, v in vars(o).items()}
    if isinstance(o, (list, tuple)): return [_todict(x) for x in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, dict): return {k: _todict(v) for k, v in o.items()}
    return o if isinstance(o, (int, float, str, bool, type(None))) else str(o)


def test_planes_are_the_five():
    assert PLANES == ["Airship_V7", "Airship_A0S", "Volantex_Ranger", "Navion", "Cirrus_SR22"]


@pytest.mark.parametrize("ac", PLANES)
def test_files_present(ac):
    c = AircraftConfig(ac)
    assert c.json_path.exists() and c.poly_path.exists()
    if ac == "Airship_A0S":
        assert c.lqr_gains_path is None
    else:
        assert c.lqr_gains_path.exists()


@pytest.mark.parametrize("ac", PLANES)
def test_params_match_archive(ac):
    gold = json.load(open(GOLD / f"{ac}.json"))
    got = _todict(load_params(ac))
    # the only permitted difference is the poly-file path, which now lives inside the package
    for d in (gold["params"], got):
        d["aero_params"].pop("poly_params_file", None)
    assert got == gold["params"]


# A config that omits a required key would inherit Airship_V7's value for it, so the loader
# rejects it. Motor topology and the rate-damping derivatives stay optional by design.

def test_every_shipped_aircraft_is_complete():
    for name in PLANES:
        AircraftConfig(name).load()


def _stage(name, tmp_path, mutate):
    src = AircraftConfig(name)
    cfg = json.loads(src.json_path.read_text())
    mutate(cfg)
    d = tmp_path / name
    d.mkdir()
    (d / f"{name}.json").write_text(json.dumps(cfg))
    (d / f"{name}_poly.csv").write_text(src.poly_path.read_text())
    return AircraftConfig(name, data_dir=tmp_path)


def test_a_missing_wing_key_raises_rather_than_taking_v7_geometry(tmp_path):
    cfg = _stage("Cirrus_SR22", tmp_path, lambda c: c["vehicle_params"]["wing"].pop("span"))
    with pytest.raises(ValueError, match=r"wing\.span"):
        cfg.load()


def test_a_motor_missing_its_disc_area_raises(tmp_path):
    def drop(c):
        motors = c["vehicle_params"]["actuator_system"]["motors"]
        next(iter(motors.values()))["motor_propeller_data"].pop("Sp")
    with pytest.raises(ValueError, match="Sp"):
        _stage("Volantex_Ranger", tmp_path, drop).load()


def test_motor_topology_and_rate_damping_stay_optional():
    single = AircraftConfig("Volantex_Ranger").load()["vehicle_params"]["actuator_system"]["motors"]
    assert set(single) == {"centre_motor"}          # no left/right, and it still loads
    assert "Clp" not in AircraftConfig("Navion").load()["aero_params"]
