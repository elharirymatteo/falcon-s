"""Repository-relative default locations. Every runner takes these as arguments; nothing reads them
from module scope at import time, so an installed package and a checkout behave the same."""
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REPO_DIR = PKG_DIR.parent.parent
DATA_DIR = PKG_DIR / "aircraft" / "data"
CKPT_DIR = REPO_DIR / "checkpoints"
RESULTS_DIR = REPO_DIR / "results"
