import shutil
from pathlib import Path

import pytest
import torch

from falcons.aircraft.config import PLANES, AircraftConfig
from falcons.aircraft.params import load_params
from falcons.paths import RESULTS_DIR

# Airframes whose OpenVSP derivative data has been extracted. The aerodynamic model is that data,
# so an airframe without it cannot be built at all -- there is no fallback model any more. Tests
# that need one of the others SKIP until its CSVs land, rather than failing.
#
# Reduced coverage is the honest cost of this: Airship_V7 is the DEFAULT airframe for the env,
# warp and MPPI tests, so most of them are dark until V7's CSVs arrive. See plan.md Phase 1.
# Every result golden frozen by the aero refactor carries this reason, so one grep finds them all
# when the archive is regenerated after retraining. See plan.md Phase 5B.
GOLDENS_PENDING = "aero model replaced; goldens pending retrain (plan.md Phase 5)"


def _why_offline(plane):
    """None if this airframe can be built, else why not. Absent CSVs and malformed ones are both
    "cannot fly", but they are reported apart: a missing file is expected while extraction is in
    progress, a file that fails to load is a defect in the export that someone must fix."""
    if not AircraftConfig(plane).has_derivatives:
        return "no OpenVSP derivative CSVs yet"
    try:
        load_params(plane)
    except Exception as e:                                  # noqa: BLE001 -- report, do not mask
        return f"derivative data does not load ({type(e).__name__}: {e})"
    return None


OFFLINE_REASON = {p: _why_offline(p) for p in PLANES}
ONLINE_PLANES = [p for p in PLANES if OFFLINE_REASON[p] is None]
OFFLINE_PLANES = [p for p in PLANES if p not in ONLINE_PLANES]


def requires_derivatives(*planes):
    """Skip mark for a test that needs these airframes' derivative data."""
    missing = [p for p in planes if p not in ONLINE_PLANES]
    return pytest.mark.skipif(
        bool(missing),
        reason="; ".join(f"{p}: {OFFLINE_REASON[p]}" for p in missing),
    )


def plane_params(planes=None):
    """`pytest.param` list over airframes, each skipped if its data is absent."""
    return [pytest.param(p, marks=requires_derivatives(p)) for p in (planes or PLANES)]


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="needs CUDA")
        for it in items:
            if "cuda" in it.keywords:
                it.add_marker(skip)


@pytest.fixture
def gate_copy():
    """Keep the table a slow gate regenerated, under results/_gate/ (gitignored), and hand it back
    so the copy happens BEFORE the assertion. A gate that re-flies for hours and then fails is
    worth nothing if its evidence went out with pytest's rotated tmp_path."""
    def keep(path, name=None):
        dst = RESULTS_DIR / "_gate"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, dst / (name or Path(path).name))
        return path
    return keep
