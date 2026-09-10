import shutil
from pathlib import Path

import pytest
import torch

from falcons.aircraft.config import PLANES, AircraftConfig
from falcons.paths import RESULTS_DIR

# Airframes whose OpenVSP derivative data has been extracted. The aerodynamic model is that data,
# so an airframe without it cannot be built at all -- there is no fallback model any more. Tests
# that need one of the others SKIP until its CSVs land, rather than failing.
#
# Reduced coverage is the honest cost of this: Airship_V7 is the DEFAULT airframe for the env,
# warp and MPPI tests, so most of them are dark until V7's CSVs arrive. See plan.md Phase 1.
ONLINE_PLANES = [p for p in PLANES if AircraftConfig(p).has_derivatives]
OFFLINE_PLANES = [p for p in PLANES if p not in ONLINE_PLANES]


def requires_derivatives(*planes):
    """Skip mark for a test that needs these airframes' derivative data."""
    missing = [p for p in planes if p not in ONLINE_PLANES]
    return pytest.mark.skipif(
        bool(missing),
        reason=f"no OpenVSP derivative CSVs yet for {', '.join(missing)} (plan.md Phase 1)",
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
