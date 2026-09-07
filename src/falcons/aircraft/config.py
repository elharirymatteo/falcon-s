"""Aircraft data files: one place that knows the layout data/<plane>/<plane>{.json,_poly.csv,_K_LQR.csv}."""
import json
from pathlib import Path

from falcons.paths import DATA_DIR

PLANES = ["Airship_V7", "Airship_A0S", "Volantex_Ranger", "Navion", "Cirrus_SR22"]

# Keys a config must carry, because every default in params.py is Airship_V7's real value: a
# missing one here silently flies V7's physics under another aeroplane's name. Deliberately
# excluded are keys whose absence is meaningful rather than accidental — motor topology (twin
# airframes name left/right, singles centre), the estimator block, and the rate-damping
# derivatives Clp/Cmq/Cnr, whose 0.0 default is neutral rather than borrowed.
REQUIRED = [
    "vehicle_params.mass",
    "vehicle_params.inertia_matrix",
    "vehicle_params.wing.span",
    "vehicle_params.wing.area",
    "vehicle_params.wing.mac",
    "vehicle_params.wing.aspect_ratio",
    "vehicle_params.wing.taper_ratio",
    "vehicle_params.wing.cg_offset_vector",
    "default_initial_state.position",
    "default_initial_state.linear_vel",
]
REQUIRED_PER_MOTOR = ["position", "motor_propeller_data.Sp", "motor_propeller_data.k_m"]


def _absent(node, dotted: str) -> bool:
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return True
        node = node[key]
    return False


def _reject_incomplete(cfg: dict, name: str) -> None:
    gaps = [k for k in REQUIRED if _absent(cfg, k)]
    motors = cfg.get("vehicle_params", {}).get("actuator_system", {}).get("motors", {})
    if not motors:
        gaps.append("vehicle_params.actuator_system.motors (at least one motor)")
    for motor, spec in motors.items():
        gaps += [f"...motors.{motor}.{k}" for k in REQUIRED_PER_MOTOR if _absent(spec, k)]
    if gaps:
        raise ValueError(
            f"{name}: aircraft config is missing required keys and would silently take "
            f"Airship_V7's values for them: {', '.join(sorted(gaps))}"
        )


class AircraftConfig:
    def __init__(self, name: str, data_dir: Path = DATA_DIR):
        if name not in PLANES:
            raise ValueError(f"unknown aircraft {name!r}; choose one of {PLANES}")
        self.name = name
        self.dir = Path(data_dir) / name
        self.json_path = self.dir / f"{name}.json"
        self.poly_path = self.dir / f"{name}_poly.csv"
        lqr = self.dir / f"{name}_K_LQR.csv"
        self.lqr_gains_path = lqr if lqr.exists() else None
        acq = self.dir / f"{name}_K_LQR_acquisition.csv"
        self.acquisition_gains_path = acq if acq.exists() else None

    def load(self) -> dict:
        """Raw JSON with the polynomial-aero file resolved to its packaged location."""
        cfg = json.loads(self.json_path.read_text())
        _reject_incomplete(cfg, self.name)
        cfg["aero_params"]["poly_params_file"] = str(self.poly_path)
        return cfg
