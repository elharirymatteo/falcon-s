"""Aircraft data files: one place that knows the layout
data/<plane>/<plane>{.json,_derivatives.csv,_ge_derivatives.csv,_K_LQR.csv}.

The OpenVSP extractor emits its two tables as bare `derivatives.csv`/`ge_derivatives.csv`; they
are renamed to carry the airframe prefix on the way in. That is not decoration -- an unprefixed
table dropped into the wrong directory is invisible, and one already was (Navion briefly carried
Volantex's set, caught only because the geometry cross-check below raised)."""
import csv
import json
from pathlib import Path

from falcons.paths import DATA_DIR

PLANES = ["Airship_V7", "Airship_A0S", "Volantex_Ranger", "Navion"]

# Keys a config must carry, because every default in params.py is Airship_V7's real value: a
# missing one here silently flies V7's physics under another aeroplane's name. Deliberately
# excluded are keys whose absence is meaningful rather than accidental — motor topology (twin
# airframes name left/right, singles centre) and the estimator block.
#
# alpha_max_deg is required: it is the hard incidence termination, and the derivative model has no
# stall of its own to fall back on, so a missing value would let a rollout run far outside the
# envelope its coefficients were fitted in.
REQUIRED = [
    "vehicle_params.mass",
    "vehicle_params.inertia_matrix",
    "vehicle_params.wing.span",
    "vehicle_params.wing.area",
    "vehicle_params.wing.mac",
    "vehicle_params.wing.aspect_ratio",
    "vehicle_params.wing.taper_ratio",
    "aero_params.alpha_max_deg",
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


# Geometry that appears BOTH in the aircraft JSON and in the VSPAERO run recorded by
# <plane>_derivatives.csv. The JSON is the reference; the CSV rows say what the coefficients were
# actually non-dimensionalised by. A disagreement means the config has drifted away from the
# geometry that generated the data, so every coefficient is being applied against the wrong
# reference area/length -- silently, and by a few tenths of a percent, which is exactly the size
# that never gets noticed. So it raises.
DUPLICATED_GEOMETRY = [
    ("vehicle_params.wing.span", "FC_Bref_"),
    ("vehicle_params.wing.area", "FC_Sref_"),
    ("vehicle_params.wing.mac", "FC_Cref_"),
]


def _at(node, dotted: str):
    for key in dotted.split("."):
        node = node[key]
    return node


def read_derivatives(path: Path) -> dict:
    """`name,value` rows -> dict. The file is one VSPAERO operating point: coefficients, stability
    derivatives, and the FC_* rows recording the flight condition they were measured at."""
    with open(path, newline="") as f:
        return {r["name"]: float(r["value"]) for r in csv.DictReader(f)}


def _reject_drift(cfg: dict, derivatives: dict, name: str) -> None:
    drift = []
    for dotted, fc in DUPLICATED_GEOMETRY:
        if fc not in derivatives:
            continue
        json_value, csv_value = float(_at(cfg, dotted)), derivatives[fc]
        if json_value != csv_value:
            drift.append(f"{dotted}={json_value!r} but {fc}={csv_value!r} in the CSV")
    if drift:
        raise ValueError(
            f"{name}: aircraft config has drifted from the geometry its aerodynamic data was "
            f"measured at, so the coefficients would be applied against the wrong reference. "
            f"Fix the JSON to match: {'; '.join(drift)}"
        )


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
        # OpenVSP derivative data. Paths are built unconditionally -- only `load` cares whether
        # they exist, so an airframe whose CSVs have not been extracted yet can still be named.
        self.derivatives_path = self.dir / f"{name}_derivatives.csv"
        self.ge_derivatives_path = self.dir / f"{name}_ge_derivatives.csv"
        # The LQR gains were fitted to the polynomial plant and were deleted with it, so these are
        # None for every airframe today. The attribute stays because the LQR controller is being
        # refactored next, not removed -- see plan.md Phase 6.
        lqr = self.dir / f"{name}_K_LQR.csv"
        self.lqr_gains_path = lqr if lqr.exists() else None
        acq = self.dir / f"{name}_K_LQR_acquisition.csv"
        self.acquisition_gains_path = acq if acq.exists() else None

    @property
    def has_derivatives(self) -> bool:
        """Whether this airframe's OpenVSP data has been extracted yet. Tests parametrised over
        PLANES skip on this rather than failing, so dropping the CSVs in is all it takes to
        bring an airframe online."""
        return self.derivatives_path.exists() and self.ge_derivatives_path.exists()

    def load(self) -> dict:
        """Raw JSON with the aero data files resolved to their packaged locations, checked against
        the geometry the derivative data was measured at."""
        cfg = json.loads(self.json_path.read_text())
        _reject_incomplete(cfg, self.name)
        if self.has_derivatives:
            _reject_drift(cfg, read_derivatives(self.derivatives_path), self.name)
        cfg["aero_params"]["derivatives_file"] = str(self.derivatives_path)
        cfg["aero_params"]["ge_derivatives_file"] = str(self.ge_derivatives_path)
        return cfg
