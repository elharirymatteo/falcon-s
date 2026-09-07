import shutil
from pathlib import Path

import pytest
import torch

from falcons.paths import RESULTS_DIR


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
