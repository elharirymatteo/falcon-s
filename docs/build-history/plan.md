# falcon-s Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `/home/matteo/Projects/falcon-s`, a fresh repository that reproduces every table and figure of the FALCON-S paper from 39 shipped checkpoints, with the old code refactored into an installable `falcons` package (~55 source files), one CLI, and an executable reproducibility check.

**Architecture:** Bottom-up migration from `/home/matteo/Projects/WIG_Plane_RL_Control` (the archive, never modified): aircraft data → three sim backends (internals untouched) → envs → controllers + one checkpoint loader → five benchmark protocols → tables/figures → train/eval CLI → `reproduce`. Every step regenerates its result CSV on the cached rollouts and diffs it against goldens frozen in Task 0; a step that fails the diff is not committed.

**Tech Stack:** Python 3.10, torch 2.7.0 (cu126), warp-lang 1.8.1, numpy 2.2.6, scipy 1.15.3, pandas 2.2.3, matplotlib 3.10.3, pillow 11.2.1, pytest. NVIDIA GPU with a CUDA 12.6 driver.

**Spec:** `docs/superpowers/specs/2026-09-04-falcon-s-design.md` (copied into the new repo in Task 1).

## Global Constraints

- Source of truth for every table/figure: the CSVs under `results/`. `tables.tex` is generated, never hand-edited.
- Package name `falcons`; import root `src/falcons`; installed editable; no `PYTHONPATH=.` anywhere.
- Plane identifiers are exactly `Airship_V7`, `Airship_A0S`, `Volantex_Ranger`, `Navion`, `Cirrus_SR22`, defined once in `falcons.aircraft.config.PLANES`.
- Checkpoints are `checkpoints/{algo}_{task}_{plane}_s{seed}.pt`, algo ∈ {ppo,sac,td3}, task ∈ {altitude,attitude}. Seed 0 is `_s0`, never unsuffixed.
- The three sim backends (`sim/cpu`, `sim/torch`, `sim/warp`) stay independent code paths. `sim/cpu` internals are copied, not rewritten.
- No module-level `OUT = "..."` path globals. Every runner takes `results_dir` / `ckpt_dir` arguments with package-relative defaults from `falcons.paths`.
- No `MPPI_SEED` / `MPPI_REPS` environment variables; they are function parameters. The `TRAJ_*` / `LQR_GAINS_PATH` variables stay: they are `wp.constant`s baked at import and require the child process.
- Regression gate: learned + LQR rows bit-identical to golden; MPPI rows within the golden `*_std`; wind cells within the Task 0 floor. Tolerances live only in `falcons/benchmark/__init__.py`.
- Commit after every green gate. Commit messages: `<layer>: <what>` (e.g. `sim: warp backend`).
- Shell variables used throughout: `OLD=/home/matteo/Projects/WIG_Plane_RL_Control`, `NEW=/home/matteo/Projects/falcon-s`, `PY=$NEW/.venv/bin/python`. Old-repo commands use `$OLD/airship/bin/python` from `$OLD`.

---

### Task 0: Freeze goldens and audit determinism (old repo, read-only)

**Files:**
- Create: `/home/matteo/Projects/falcon-s-goldens/` (scratch, outside both repos)

**Interfaces:**
- Produces: `golden/{altitude,ge_trim,ge_energy,maneuvers,robustness}.csv`, `golden/mppi_variance.json`, `golden/throughput.json`, `golden/figures/*`, `golden/configs/<plane>.json` (old loader output), `golden/warp_step.npz` (one warp step from a fixed seed), `golden/attitude_obs.npy`, `golden/DETERMINISM.md`. Consumed by every later gate.

- [ ] **Step 1: Copy the result artefacts**

```bash
G=/home/matteo/Projects/falcon-s-goldens; OLD=/home/matteo/Projects/WIG_Plane_RL_Control
mkdir -p $G/golden/figures $G/golden/configs $G/traces
cp $OLD/trl_paper/altitude/results/table_altitude_protocol.csv $G/golden/altitude.csv
cp $OLD/trl_paper/altitude/results/table_ge_trim.csv          $G/golden/ge_trim.csv
cp $OLD/trl_paper/altitude/results/table_ge_energy.csv        $G/golden/ge_energy.csv
cp $OLD/trl_paper/maneuvers/results/table_maneuver_track.csv  $G/golden/maneuvers.csv
cp $OLD/trl_paper/maneuvers/results/table_robustness.csv      $G/golden/robustness.csv
cp $OLD/trl_paper/maneuvers/results/mppi_variance.json        $G/golden/
cp $OLD/trl_paper/results/throughput.json                     $G/golden/
cp $OLD/paper/revisions/results_v3_tables.tex                 $G/golden/tables.tex
for f in altitude_ppo_viz_Airship_V7 altitude_ppo_viz_Volantex_Ranger altitude_sac_viz_Airship_V7 \
         altitude_sac_viz_Volantex_Ranger altitude_td3_viz_Airship_V7 altitude_td3_viz_Volantex_Ranger; do
  cp $OLD/figures/*/$f.pdf $G/golden/figures/; done
cp $OLD/trl_paper/maneuvers/results/{fig_attitude_exec.pdf,fig_maneuver_facet_Airship_V7.png,fig_maneuver_render.png,fig_robustness_ppo.pdf} $G/golden/figures/
cp $OLD/trl_paper/altitude/results/fig_ge_energy.pdf $G/golden/figures/
cp -r $OLD/trl_paper/maneuvers/results/traces/. $G/traces/          # 39 MB rollout cache = replay fixture
ls $G/golden $G/golden/figures | wc -l    # expect 9 + 11
```

- [ ] **Step 2: Dump the old config loader output and one warp step, per plane**

```bash
cd $OLD && airship/bin/python - <<'PY'
import json, numpy as np, torch, warp as wp
from data.vehicle_config import create_vehicle_config_from_json
from utils.config import AircraftConfigManager
from envs.altitude import WarpAltitudeEnv
from envs.attitude import WarpAttitudeEnv
from train.configs import altitude_env_cfg, attitude_env_cfg, PLANE_CONFIGS
G="/home/matteo/Projects/falcon-s-goldens/golden"
def todict(o):
    if hasattr(o, "__dict__"): return {k: todict(v) for k, v in vars(o).items()}
    if isinstance(o, (list, tuple)): return [todict(x) for x in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, dict): return {k: todict(v) for k, v in o.items()}
    return o if isinstance(o, (int, float, str, bool, type(None))) else str(o)
for ac in ["Airship_V7","Airship_A0S","Volantex_Ranger","Navion","Cirrus_SR22"]:
    json.dump({"raw": AircraftConfigManager(ac).load_config(),
               "params": todict(create_vehicle_config_from_json(ac))},
              open(f"{G}/configs/{ac}.json","w"), indent=1, sort_keys=True)
wp.init(); torch.manual_seed(0)
env = WarpAltitudeEnv(8, "cuda", altitude_env_cfg("Airship_V7", PLANE_CONFIGS["Airship_V7"]["spawn"]))
obs0 = env.reset().clone(); act = torch.zeros(8, env.num_act, device="cuda")
obs1, rew, done, _ = env.step(act)
np.savez(f"{G}/warp_step.npz", obs0=obs0.cpu().numpy(), obs1=obs1.cpu().numpy(), rew=rew.cpu().numpy())
torch.manual_seed(0)
aenv = WarpAttitudeEnv(8, "cuda", attitude_env_cfg("Airship_V7"))
np.save(f"{G}/attitude_obs.npy", aenv.reset().cpu().numpy())
print("goldens written")
PY
```

If `env.step` returns a different tuple shape, read `envs/altitude.py:458-520` and adapt the unpacking; record what it returns in `DETERMINISM.md`.

- [ ] **Step 3: Determinism audit — run each cheap probe twice and diff**

```bash
cd $OLD && airship/bin/python - <<'PY'
import numpy as np, torch, warp as wp, json, os
from trl_paper.altitude.protocol import run_rl, aggregate
from trl_paper.maneuvers.harness import run_maneuver
from eval.attitude._policy import load_policy
G="/home/matteo/Projects/falcon-s-goldens/golden"; out=[]
wp.init()
a = aggregate(run_rl("Volantex_Ranger", "ppo")); b = aggregate(run_rl("Volantex_Ranger", "ppo"))
out.append(("altitude ppo Volantex (200 ep)", {k: abs(a[k]-b[k]) for k in a if isinstance(a[k], float)}))
def fly():
    fac = lambda env, p, h, v: load_policy("ppo", "Airship_V7", 15, 4, env=env, stream=(p,h,v)).actor_mean
    return run_maneuver("Airship_V7", "circle", controller_factory=fac, return_trace=True)["phi"]["rmse"]
out.append(("maneuver ppo V7 circle", abs(fly() - fly())))
open(f"{G}/DETERMINISM.md","w").write("# run-twice deltas on the archive\n\n" + "\n".join(f"- {n}: {d}" for n, d in out) + "\n")
print(open(f"{G}/DETERMINISM.md").read())
PY
```

Expected: both deltas `0.0` (warp rollouts with a fixed seed are bit-stable). **If either is non-zero, stop and report the floor before Task 1** — the tolerance table in Task 7 uses it. MPPI is known non-deterministic and is not probed here.

- [ ] **Step 4: Record source provenance**

```bash
cd $OLD && { echo "source_commit=$(git rev-parse HEAD)"; echo "source_remote=$(git remote get-url origin)"; echo "frozen=$(date -I)"; } > $G/golden/SOURCE.txt; cat $G/golden/SOURCE.txt
```

---

### Task 1: Scaffold the repository and the environment

**Files:**
- Create: `$NEW/pyproject.toml`, `$NEW/requirements.lock`, `$NEW/scripts/setup_venv.sh`, `$NEW/.gitignore`, `$NEW/src/falcons/__init__.py`, `$NEW/src/falcons/paths.py`, `$NEW/src/falcons/cli.py`, `$NEW/tests/conftest.py`, `$NEW/tests/test_scaffold.py`, `$NEW/docs/` (spec + this plan)

**Interfaces:**
- Produces: `falcons.paths.CKPT_DIR`, `falcons.paths.RESULTS_DIR`, `falcons.paths.DATA_DIR` (pathlib.Path); `falcons.cli.main()`; console script `falcons`.

- [ ] **Step 1: Create the tree and pyproject**

```bash
NEW=/home/matteo/Projects/falcon-s; mkdir -p $NEW/src/falcons $NEW/tests $NEW/scripts $NEW/checkpoints $NEW/results $NEW/docs
cp $OLD/docs/superpowers/specs/2026-09-04-falcon-s-design.md $NEW/docs/design.md
cp $OLD/docs/superpowers/plans/2026-09-04-falcon-s.md $NEW/docs/plan.md
cat > $NEW/pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "falcons"
version = "1.0.0"
description = "FALCON-S: fixed-wing flight-control simulation framework and controller benchmark near the ground"
requires-python = "==3.10.*"
dependencies = [
  "numpy==2.2.6",
  "scipy==1.15.3",
  "pandas==2.2.3",
  "matplotlib==3.10.3",
  "pillow==11.2.1",
  "torch==2.7.0",
  "warp-lang==1.8.1",
]

[project.optional-dependencies]
test = ["pytest==8.3.5"]

[project.scripts]
falcons = "falcons.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
falcons = ["aircraft/data/*/*.json", "aircraft/data/*/*.csv"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: full reproduction, minutes to hours"]
EOF
```

- [ ] **Step 2: Write `.gitignore`, `paths.py`, package init, CLI stub**

```bash
cat > $NEW/.gitignore <<'EOF'
__pycache__/
*.pyc
.venv/
*.egg-info/
results/traces/
results/_frames/
*.mp4
EOF
cat > $NEW/src/falcons/__init__.py <<'EOF'
"""FALCON-S: simulation framework and controller benchmark for fixed-wing flight near the ground."""
__version__ = "1.0.0"
EOF
cat > $NEW/src/falcons/paths.py <<'EOF'
"""Repository-relative default locations. Every runner takes these as arguments; nothing reads them
from module scope at import time, so an installed package and a checkout behave the same."""
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REPO_DIR = PKG_DIR.parent.parent
DATA_DIR = PKG_DIR / "aircraft" / "data"
CKPT_DIR = REPO_DIR / "checkpoints"
RESULTS_DIR = REPO_DIR / "results"
EOF
cat > $NEW/src/falcons/cli.py <<'EOF'
"""`falcons` command line: argparse glue only. Every subcommand calls one importable function."""
import argparse
import sys


def cmd_check(a):
    import torch, warp
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} | warp {warp.__version__}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="falcons")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="verify the install: versions, GPU, checkpoints, one step per backend")
    a = p.parse_args(argv)
    return {"check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
EOF
```

- [ ] **Step 3: Build the venv and freeze the lock**

```bash
cat > $NEW/scripts/setup_venv.sh <<'EOF'
#!/usr/bin/env bash
# From a clean clone to a green `falcons check`. Requires python3.10 and an NVIDIA driver for CUDA 12.6.
set -euo pipefail
cd "$(dirname "$0")/.."
python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e ".[test]" --no-deps
.venv/bin/falcons check
EOF
chmod +x $NEW/scripts/setup_venv.sh
cd $NEW && python3.10 -m venv .venv && .venv/bin/pip install --upgrade pip
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.0
.venv/bin/pip install warp-lang==1.8.1 numpy==2.2.6 scipy==1.15.3 pandas==2.2.3 matplotlib==3.10.3 pillow==11.2.1 pytest==8.3.5
{ echo "--extra-index-url https://download.pytorch.org/whl/cu126"; .venv/bin/pip freeze; } > requirements.lock
.venv/bin/pip install -e ".[test]" --no-deps
.venv/bin/falcons check     # expect: torch 2.7.0+cu126 cuda=True | warp 1.8.1
```

- [ ] **Step 4: Write the scaffold test and run it**

```bash
cat > $NEW/tests/conftest.py <<'EOF'
import pytest, torch

def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="needs CUDA")
        for it in items:
            if "cuda" in it.keywords:
                it.add_marker(skip)
EOF
cat > $NEW/tests/test_scaffold.py <<'EOF'
import subprocess, sys
from falcons import paths


def test_paths_resolve():
    assert paths.PKG_DIR.name == "falcons"
    assert (paths.REPO_DIR / "pyproject.toml").exists()


def test_cli_check_runs():
    assert subprocess.run([sys.executable, "-m", "falcons.cli", "check"]).returncode == 0
EOF
cd $NEW && .venv/bin/pytest -q      # expect 2 passed
```

- [ ] **Step 5: Verify the setup script from scratch, then commit**

```bash
cd $NEW && rm -rf .venv && bash scripts/setup_venv.sh   # must end with the versions line
git init -b main && git add -A && git commit -q -m "scaffold: package, lock, venv script, cli stub"
```

---

### Task 2: `aircraft/` — data, config loader, parameter structs

**Files:**
- Create: `src/falcons/aircraft/__init__.py`, `src/falcons/aircraft/config.py`, `src/falcons/aircraft/params.py`, `src/falcons/aircraft/data/<plane>/…`, `tests/test_aircraft.py`
- Source: `$OLD/utils/config.py`, `$OLD/data/vehicle_config.py`, `$OLD/data/aircraft/*`, `$OLD/trl_paper/altitude/gains/*`

**Interfaces:**
- Produces: `falcons.aircraft.config.PLANES: list[str]`; `AircraftConfig(name)` with `.name`, `.dir: Path`, `.json_path`, `.poly_path`, `.lqr_gains_path`, `.acquisition_gains_path` (Path or None), `.load() -> dict` (raw JSON with `aero_params.poly_params_file` rewritten to `.poly_path`); `falcons.aircraft.params.load_params(name) -> dict` with keys `vehicle_params, environment_params, control_limits, aero_params, default_initial_state` (identical objects to the old `create_vehicle_config_from_json`).

- [ ] **Step 1: Copy and rename the data**

```bash
mkdir -p $NEW/src/falcons/aircraft/data && cd $OLD/data/aircraft
for ac in Airship_V7 Airship_A0S Volantex_Ranger Navion Cirrus_SR22; do
  d=$NEW/src/falcons/aircraft/data/$ac; mkdir -p $d
  cp $ac/$(ls $ac | grep -i '\.json$' | head -1)                 $d/$ac.json
  cp $ac/$(ls $ac | grep -i 'K_LQR\.csv$' | head -1)              $d/${ac}_K_LQR.csv
done
# poly files: V7 and A0S carry an extra refit variant; the json names the one in use — copy that one
for ac in Airship_V7 Airship_A0S Volantex_Ranger Navion Cirrus_SR22; do
  d=$NEW/src/falcons/aircraft/data/$ac
  poly=$(python3 -c "import json,os;print(os.path.basename(json.load(open('$d/$ac.json'))['aero_params']['poly_params_file']))")
  cp $ac/$poly $d/${ac}_poly.csv
done
cp $OLD/trl_paper/altitude/gains/Airship_V7_acquisition_K_LQR.csv      $NEW/src/falcons/aircraft/data/Airship_V7/Airship_V7_K_LQR_acquisition.csv
cp $OLD/trl_paper/altitude/gains/Volantex_Ranger_acquisition_K_LQR.csv $NEW/src/falcons/aircraft/data/Volantex_Ranger/Volantex_Ranger_K_LQR_acquisition.csv
find $NEW/src/falcons/aircraft/data -type f | sort      # expect 17 files: 5 json, 5 poly, 5 K_LQR, 2 acquisition
```

If A0S's json points at `A0_V8_poly_params_refit_corrected.csv`, that is the one copied — the corrected refit is what the paper used.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_aircraft.py
import json
import numpy as np
import pytest
from falcons.aircraft.config import PLANES, AircraftConfig
from falcons.aircraft.params import load_params

GOLD = "/home/matteo/Projects/falcon-s-goldens/golden/configs"


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
    assert c.json_path.exists() and c.poly_path.exists() and c.lqr_gains_path.exists()


@pytest.mark.parametrize("ac", PLANES)
def test_params_match_archive(ac):
    gold = json.load(open(f"{GOLD}/{ac}.json"))
    got = _todict(load_params(ac))
    # the only permitted difference is the poly-file path, which now lives inside the package
    for d in (gold["params"], got):
        d["aero_params"].pop("poly_params_file", None)
    assert got == gold["params"]
```

Run: `cd $NEW && .venv/bin/pytest tests/test_aircraft.py -q` → FAIL `ModuleNotFoundError: falcons.aircraft`.

- [ ] **Step 3: Write `config.py`**

```python
# src/falcons/aircraft/config.py
"""Aircraft data files: one place that knows the layout data/<plane>/<plane>{.json,_poly.csv,_K_LQR.csv}."""
import json
from pathlib import Path

from falcons.paths import DATA_DIR

PLANES = ["Airship_V7", "Airship_A0S", "Volantex_Ranger", "Navion", "Cirrus_SR22"]


class AircraftConfig:
    def __init__(self, name: str, data_dir: Path = DATA_DIR):
        if name not in PLANES:
            raise ValueError(f"unknown aircraft {name!r}; choose one of {PLANES}")
        self.name = name
        self.dir = Path(data_dir) / name
        self.json_path = self.dir / f"{name}.json"
        self.poly_path = self.dir / f"{name}_poly.csv"
        self.lqr_gains_path = self.dir / f"{name}_K_LQR.csv"
        acq = self.dir / f"{name}_K_LQR_acquisition.csv"
        self.acquisition_gains_path = acq if acq.exists() else None

    def load(self) -> dict:
        """Raw JSON with the polynomial-aero file resolved to its packaged location."""
        cfg = json.loads(self.json_path.read_text())
        cfg["aero_params"]["poly_params_file"] = str(self.poly_path)
        return cfg
```

- [ ] **Step 4: Write `params.py` from `data/vehicle_config.py`**

```bash
cp $OLD/data/vehicle_config.py $NEW/src/falcons/aircraft/params.py
touch $NEW/src/falcons/aircraft/__init__.py
cd $NEW && python3 - <<'PY'
p="src/falcons/aircraft/params.py"; t=open(p).read()
t=t.replace("from utils.config import AircraftConfigManager","from falcons.aircraft.config import AircraftConfig")
old=t[t.index("def create_vehicle_config_from_json"):]
new='''def load_params(aircraft_name: str):
    """All parameter structs for one aircraft, built from its packaged JSON + polynomial-aero CSV."""
    cfg = AircraftConfig(aircraft_name)
    config = cfg.load()
    return {
        'vehicle_params': VehicleParameters.from_config(config),
        'environment_params': EnvironmentParameters.from_config(config),
        'control_limits': ControlLimits.from_config(config),
        'aero_params': AerodynamicsParameters.from_config(config, cfg),
        'default_initial_state': DefaultInitialState.from_config(config)
    }
'''
open(p,"w").write(t.replace(old,new))
PY
grep -n "config_manager\|AircraftConfigManager\|get_config_path\|get_lqr_gains_path\|aircraft_dir" src/falcons/aircraft/params.py
```

`AerodynamicsParameters.from_config(config, cfg)` receives the config object because the old code used the manager to locate the poly CSV. Open the function, replace any `config_manager.aircraft_dir / …` or `get_…path()` use with `cfg.poly_path`, and nothing else.

- [ ] **Step 5: Run the test, then commit**

Run: `cd $NEW && .venv/bin/pytest tests/test_aircraft.py -q` → 11 passed.

```bash
cd $NEW && git add -A && git commit -q -m "aircraft: data for five planes, config loader, parameter structs"
```

---

### Task 3: `sim/cpu` — the reference plant, copied verbatim

**Files:**
- Create: `src/falcons/sim/__init__.py`, `src/falcons/sim/cpu/**` (from `$OLD/backends/cpu/**`), `tests/test_sim_cpu.py`
- Source tests: `$OLD/tests/test_piston_thrust.py`, `$OLD/tests/test_v7_trim.py`

**Interfaces:**
- Produces: `falcons.sim.cpu.Aircraft` (was `backends.cpu.aircraft_sim.aircraft_model_sim.AircraftModel`), constructor `Aircraft(aircraft_name, save_history=False)`; everything under `sim/cpu/{physics,actuators,sensors,estimators}` keeps its old names.

- [ ] **Step 1: Copy and rewrite imports**

```bash
mkdir -p $NEW/src/falcons/sim && touch $NEW/src/falcons/sim/__init__.py
cp -r $OLD/backends/cpu $NEW/src/falcons/sim/cpu
cd $NEW/src/falcons/sim/cpu
mv aircraft_sim/aircraft_model_sim.py aircraft.py && rm -r aircraft_sim
mv base/aircraft_base.py base.py && rm -r base
find . -name "__pycache__" -prune -exec rm -rf {} +
grep -rl "backends\.cpu\|utils\.config\|data\.vehicle_config\|utils\.util" . | xargs sed -i \
  -e 's/backends\.cpu\.aircraft_sim\.aircraft_model_sim/falcons.sim.cpu.aircraft/g' \
  -e 's/backends\.cpu\.base\.aircraft_base/falcons.sim.cpu.base/g' \
  -e 's/backends\.cpu/falcons.sim.cpu/g' \
  -e 's/from utils\.config import AircraftConfigManager/from falcons.aircraft.config import AircraftConfig/' \
  -e 's/AircraftConfigManager(/AircraftConfig(/g' \
  -e 's/data\.vehicle_config/falcons.aircraft.params/g'
sed -i -e 's/^class AircraftModel(/class Aircraft(/' -e 's/\.load_config()/.load()/g' aircraft.py
grep -rn "backends\|utils\.\|AircraftModel\b" . ; echo "(expect no output)"
```

`aircraft.py` line 31–34 built `self.config_manager = AircraftConfigManager(name)` then `.load_config()`; after the sed it is `self.config_manager = AircraftConfig(name)` / `.load()`. Keep the attribute name `config_manager` — `controllers/lqr/control.py` reads `self.airship.config_manager` (fixed in Task 6).

- [ ] **Step 2: Port the two tests**

```bash
cp $OLD/tests/test_piston_thrust.py $NEW/tests/test_sim_cpu.py
sed -n '1,62p' $OLD/tests/test_v7_trim.py | grep -v "^import pytest" >> $NEW/tests/test_sim_cpu.py
cd $NEW && sed -i \
  -e 's/from data\.vehicle_config import create_vehicle_config_from_json/from falcons.aircraft.params import load_params/' \
  -e 's/create_vehicle_config_from_json(/load_params(/g' \
  -e 's/from backends\.cpu\.[a-z_.]*aircraft_model_sim import AircraftModel/from falcons.sim.cpu.aircraft import Aircraft/' \
  -e 's/AircraftModel(/Aircraft(/g' tests/test_sim_cpu.py
grep -n "^from\|^import" tests/test_sim_cpu.py
```

Open the file once: the two source tests each had their own imports and helpers; if `test_v7_trim.py` defined a helper with the same name as one in `test_piston_thrust.py`, keep the first and delete the duplicate.

- [ ] **Step 3: Run, then commit**

Run: `cd $NEW && .venv/bin/pytest tests/test_sim_cpu.py -q` → 9 passed (6 piston + 3 trim).

```bash
cd $NEW && git add -A && git commit -q -m "sim: cpu reference plant (verbatim), trim and thrust tests"
```

---

### Task 4: `sim/warp` and `sim/torch`

**Files:**
- Create: `src/falcons/sim/warp/*.py`, `src/falcons/sim/torch/{__init__,altitude,wind}.py`, `tests/test_parity.py`, `tests/test_sim_warp.py`
- Source: `$OLD/backends/warp/sim/*`, `$OLD/backends/warp/altitude/{obs,reward}.py`, `$OLD/backends/torch/{altitude,wind}.py`; tests `test_torch_dryden_parity`, `test_termination_kernel`, `test_obs_kernel`, `test_sensors_warp`, `test_attitude_propagation`

**Interfaces:**
- Produces: `falcons.sim.warp.Aircraft` (was `AircraftModelControl`; same constructor), `falcons.sim.warp.altitude_obs.compute_obs`, `falcons.sim.warp.altitude_reward.compute_reward`, `falcons.sim.warp.termination.check_termination_batch`, `falcons.sim.torch.altitude.AltitudeEnv` (was `TorchAltitudeEnv`), `falcons.sim.torch.wind.TorchDryden`, `falcons.sim.cpu.physics.dryden.DrydenTurbulenceModel`.

- [ ] **Step 1: Copy and rewrite**

```bash
cd $NEW/src/falcons/sim
cp -r $OLD/backends/warp/sim warp
cp $OLD/backends/warp/altitude/obs.py    warp/altitude_obs.py
cp $OLD/backends/warp/altitude/reward.py warp/altitude_reward.py
mkdir torch && touch torch/__init__.py && cp $OLD/backends/torch/{altitude,wind}.py torch/
find . -name "__pycache__" -prune -exec rm -rf {} +
grep -rl "backends\|utils\.config\|data\.vehicle_config" warp torch | xargs sed -i \
  -e 's/backends\.warp\.altitude\.obs/falcons.sim.warp.altitude_obs/g' \
  -e 's/backends\.warp\.altitude\.reward/falcons.sim.warp.altitude_reward/g' \
  -e 's/backends\.warp\.sim/falcons.sim.warp/g' \
  -e 's/backends\.torch/falcons.sim.torch/g' \
  -e 's/backends\.cpu/falcons.sim.cpu/g' \
  -e 's/from utils\.config import AircraftConfigManager/from falcons.aircraft.config import AircraftConfig/' \
  -e 's/AircraftConfigManager(\([^)]*\))\.load_config()/AircraftConfig(\1).load()/g' \
  -e 's/data\.vehicle_config/falcons.aircraft.params/g' \
  -e 's/create_vehicle_config_from_json/load_params/g'
sed -i 's/^class AircraftModelControl()/class Aircraft/' warp/aircraft.py
sed -i 's/^class TorchAltitudeEnv/class AltitudeEnv/' torch/altitude.py
grep -rn "AircraftModelControl\|TorchAltitudeEnv" warp torch     # rename any remaining internal references
cat > warp/__init__.py <<'EOF'
from falcons.sim.warp.aircraft import Aircraft  # noqa: F401
EOF
```

If `warp/__init__.py` already existed with content, append the import instead of overwriting.

- [ ] **Step 2: Port the tests**

```bash
cd $NEW/tests
cp $OLD/tests/test_torch_dryden_parity.py test_parity.py
cat $OLD/tests/test_termination_kernel.py $OLD/tests/test_obs_kernel.py $OLD/tests/test_sensors_warp.py $OLD/tests/test_attitude_propagation.py > test_sim_warp.py
sed -i \
  -e 's/backends\.cpu\.physics\.dryden/falcons.sim.cpu.physics.dryden/' \
  -e 's/backends\.torch\.wind/falcons.sim.torch.wind/' \
  -e 's/backends\.torch\.altitude/falcons.sim.torch.altitude/' \
  -e 's/TorchAltitudeEnv/AltitudeEnv/g' \
  -e 's/backends\.warp\.sim\.termination/falcons.sim.warp.termination/' \
  -e 's/backends\.warp\.sim\.aircraft import AircraftModelControl/falcons.sim.warp.aircraft import Aircraft/' \
  -e 's/AircraftModelControl(/Aircraft(/g' \
  -e 's/backends\.warp\.altitude\.obs/falcons.sim.warp.altitude_obs/' \
  -e 's/from data\.vehicle_config import create_vehicle_config_from_json/from falcons.aircraft.params import load_params/' \
  -e 's/create_vehicle_config_from_json(/load_params(/g' \
  -e '/plt\.show()/d' test_parity.py test_sim_warp.py
```

Open `test_sim_warp.py`: four files were concatenated, so deduplicate the import block at the top and delete any second `pytestmark = …` line.

- [ ] **Step 3: Add the bit-identity test against the Task 0 golden**

Append to `tests/test_parity.py`:

```python
import numpy as np, torch, warp as wp, pytest


@pytest.mark.cuda
def test_warp_altitude_step_matches_archive():
    """One reset + one zero-action step from a fixed seed must equal the archive's output exactly.
    Anything else means the copy changed the plant."""
    from falcons.envs.altitude import AltitudeEnv
    from falcons.envs.configs import altitude_env_cfg, PLANE_CONFIGS
    g = np.load("/home/matteo/Projects/falcon-s-goldens/golden/warp_step.npz")
    wp.init(); torch.manual_seed(0)
    env = AltitudeEnv(8, "cuda", altitude_env_cfg("Airship_V7", PLANE_CONFIGS["Airship_V7"]["spawn"]))
    obs0 = env.reset().clone()
    obs1, rew, done, _ = env.step(torch.zeros(8, env.num_act, device="cuda"))
    np.testing.assert_array_equal(obs0.cpu().numpy(), g["obs0"])
    np.testing.assert_array_equal(obs1.cpu().numpy(), g["obs1"])
    np.testing.assert_array_equal(rew.cpu().numpy(), g["rew"])
```

This test imports `falcons.envs`, which Task 5 creates; it is expected to fail with `ModuleNotFoundError` until then.

- [ ] **Step 4: Run what can run, then commit**

Run: `cd $NEW && .venv/bin/pytest tests/test_parity.py tests/test_sim_warp.py -q --deselect tests/test_parity.py::test_warp_altitude_step_matches_archive` → 7 + 4-file tests pass, 0 failed.

```bash
cd $NEW && git add -A && git commit -q -m "sim: warp and torch backends, parity tests"
```

---

### Task 5: `envs/` and env configs

**Files:**
- Create: `src/falcons/envs/{__init__,altitude,attitude,cpu,configs}.py`, `tests/test_envs.py`
- Source: `$OLD/envs/{altitude,attitude,cpu_env}.py`, `$OLD/train/configs.py`; tests `test_attitude_{env,obs,reward,schedule,curriculum}`, `test_heading_invariance_warp`

**Interfaces:**
- Produces: `falcons.envs.altitude.AltitudeEnv(num_envs, device, cfg)` (was `WarpAltitudeEnv`), `falcons.envs.attitude.AttitudeEnv(num_envs, device, cfg)` (was `WarpAttitudeEnv`; keeps `.set_curriculum(frac)`, `.phi_max_full`, `.hdot_max_full`, `.base_vel`, `._phi_tgt/_hdot_tgt/_va_tgt`, `._obs_launch()`, `._obs`, `.model`), `falcons.envs.cpu.CpuEnv` (was `CoreAircraftEnv`, no longer a `gym.Env` subclass), `falcons.envs.configs.{PLANE_CONFIGS, TASK_DEFAULTS, RAMP_CONFIGS, ATTITUDE_CONFIGS, altitude_env_cfg(plane, spawn, horizon=2000), attitude_env_cfg(plane, horizon=2000), TRIM_VA}`.

- [ ] **Step 1: Copy, rename, rewrite imports**

```bash
mkdir -p $NEW/src/falcons/envs && cd $NEW/src/falcons/envs && touch __init__.py
cp $OLD/envs/altitude.py altitude.py; cp $OLD/envs/attitude.py attitude.py; cp $OLD/envs/cpu_env.py cpu.py
cp $OLD/train/configs.py configs.py
sed -i \
  -e 's/backends\.warp\.altitude\.obs/falcons.sim.warp.altitude_obs/g' \
  -e 's/backends\.warp\.altitude\.reward/falcons.sim.warp.altitude_reward/g' \
  -e 's/backends\.warp\.sim/falcons.sim.warp/g' -e 's/backends\.cpu/falcons.sim.cpu/g' \
  -e 's/AircraftModelControl/Aircraft/g' \
  -e 's/from utils\.config import AircraftConfigManager/from falcons.aircraft.config import AircraftConfig/' \
  -e 's/AircraftConfigManager(\([^)]*\))\.load_config()/AircraftConfig(\1).load()/g' \
  -e 's/data\.vehicle_config/falcons.aircraft.params/g' -e 's/create_vehicle_config_from_json/load_params/g' \
  -e 's/from train\.configs/from falcons.envs.configs/g' \
  -e 's/^class WarpAltitudeEnv/class AltitudeEnv/' -e 's/^class WarpAttitudeEnv/class AttitudeEnv/' \
  -e 's/^class CoreAircraftEnv(gym\.Env)/class CpuEnv/' altitude.py attitude.py cpu.py configs.py
sed -i -e '/^import gymnasium\|^from gymnasium\|^import gym$/d' -e 's/super().__init__()//' cpu.py
sed -i -e 's/from utils\.util import Trajectory/from falcons.envs.trajectory import Trajectory/' cpu.py
```

- [ ] **Step 2: Slim `Trajectory` to what `cpu.py` and the LQR/MPPI loops use**

```bash
cd $NEW/src/falcons/envs && cp $OLD/utils/util.py trajectory.py
python3 - <<'PY'
p="trajectory.py"; t=open(p).read()
# drop the plotting / matlab / x-plane surface: everything from `def plot(` up to `def clear(`
a=t.index("    def plot("); b=t.index("    def clear(")
t=t[:a]+t[b:]
t=t.replace("from utils.plot_data import","# (plotting removed) from utils.plot_data import")
open(p,"w").write(t)
PY
grep -n "^from\|^import\|plot_data\|matplotlib" trajectory.py
```

Delete any remaining `plot_data`/`matplotlib` import lines. The class keeps `save_state`, `save_to_file`, `save_to_file_v2`, `load_from_df`, `clear`, `is_empty`, `dump`, `load`, `remove_last`.

- [ ] **Step 3: Trim `configs.py` to the two paper tasks**

Edit `configs.py`:
- `TASK_DEFAULTS`: keep only the `"altitude"` and `"attitude"` entries.
- Delete `FLARE_*`, `N_WP`, `FLOOR_CFG`, `FLARE_CFG`, `TRANSFER_MASK`, `DR_CFG`, `BANK_HEADING_CONFIGS`, `bank_heading_env_cfg`, `WP_SPEED`, `W_HOME`, `waypoint_env_cfg`, `waypoint_seq_env_cfg`.
- `altitude_env_cfg(aircraft, spawn, horizon=2000, dr=False, deploy=False, deploy_sensors=True, …)` → signature `altitude_env_cfg(aircraft, spawn, horizon=2000)`. Inside, every branch guarded by a removed flag is deleted; what remains is the `dynamic` path. Keep `SPAWN_START`, `RAMP_CONFIGS`, `PLANE_CONFIGS`, `ATTITUDE_CONFIGS`, `attitude_env_cfg`.
- Add at the end: `TRIM_VA = {"Airship_V7": 28.0, "Volantex_Ranger": 15.0}` copied from `$OLD/trl_paper/altitude/protocol.py:55-57` (include every plane that line lists).

Then: `cd $NEW && .venv/bin/python -c "from falcons.envs.configs import altitude_env_cfg, attitude_env_cfg, PLANE_CONFIGS; print(altitude_env_cfg('Airship_V7', PLANE_CONFIGS['Airship_V7']['spawn']))"` — must print a dict without error.

- [ ] **Step 4: Port the env tests + the obs golden**

```bash
cd $NEW/tests
cat $OLD/tests/test_attitude_env.py $OLD/tests/test_attitude_obs.py $OLD/tests/test_attitude_reward.py \
    $OLD/tests/test_attitude_schedule.py $OLD/tests/test_attitude_curriculum.py $OLD/tests/test_heading_invariance_warp.py > test_envs.py
sed -i -e 's/envs\.attitude import WarpAttitudeEnv/falcons.envs.attitude import AttitudeEnv/g' -e 's/WarpAttitudeEnv/AttitudeEnv/g' \
       -e 's/envs\.altitude import WarpAltitudeEnv/falcons.envs.altitude import AltitudeEnv/g' -e 's/WarpAltitudeEnv/AltitudeEnv/g' \
       -e 's/from train\.configs/from falcons.envs.configs/g' -e 's/from train\.tasks import evaluate_attitude/from falcons.train.tasks import evaluate_attitude/' test_envs.py
cat >> test_envs.py <<'EOF'


@pytest.mark.cuda
def test_attitude_reset_obs_matches_archive():
    import warp as wp
    from falcons.envs.attitude import AttitudeEnv
    from falcons.envs.configs import attitude_env_cfg
    g = np.load("/home/matteo/Projects/falcon-s-goldens/golden/attitude_obs.npy")
    wp.init(); torch.manual_seed(0)
    np.testing.assert_array_equal(AttitudeEnv(8, "cuda", attitude_env_cfg("Airship_V7")).reset().cpu().numpy(), g)
EOF
```

Deduplicate the import block; `test_evaluate_attitude_returns_sane_tuple` imports `evaluate_attitude` from `train.tasks`, which Task 9 provides — mark that one test `@pytest.mark.skip(reason="train.tasks arrives in Task 9")` for now and remove the mark in Task 9.

- [ ] **Step 5: Run everything so far, then commit**

Run: `cd $NEW && .venv/bin/pytest -q` → all pass except the one skip; `test_warp_altitude_step_matches_archive` (Task 4) now passes.

```bash
cd $NEW && git add -A && git commit -q -m "envs: altitude, attitude, cpu; configs trimmed to the paper's two tasks"
```

---

### Task 6: `controllers/` — LQR, MPPI, the checkpoint loader, renamed checkpoints

**Files:**
- Create: `src/falcons/controllers/{__init__,policies,altitude}.py`, `src/falcons/controllers/lqr/{__init__,control,attitude,reference}.py`, `src/falcons/controllers/mppi/{__init__,control,attitude,reference,cost,kernels}.py`, `checkpoints/*.pt` (39), `tests/test_checkpoints.py`, `tests/test_controllers.py`
- Source: `$OLD/controllers/**`, `$OLD/eval/controllers/{lqr,mppi}.py`, `$OLD/eval/altitude/_policy.py`, `$OLD/eval/attitude/_policy.py`, `$OLD/trained_agents/`

**Interfaces:**
- Produces: `falcons.controllers.policies.ckpt_path(algo, task, plane, seed, ckpt_dir=CKPT_DIR) -> Path`; `seeds(algo, task, plane, ckpt_dir) -> list[int]`; `load_policy(algo, task, plane, seed=0, device="cuda", env=None, stream=None, ckpt_dir=CKPT_DIR)` returning an object with `.act(obs) -> action` (learned: mean action; `lqr`/`mppi`: the attitude executors, need `env`, MPPI also `stream=(phi, hdot, va)`); `falcons.controllers.altitude.LQRAltitude(plane, init_altitude)`, `MPPIAltitude(plane, init_altitude, mppi_seed=0)` each with `.run(scenario: dict) -> dict` (was `LQRAdapter/MPPIAdapter.run_scenario`); `falcons.controllers.lqr.attitude.LQRAttitudeExecutor`, `falcons.controllers.mppi.attitude.MPPIAttitudeExecutor` (renamed only in import path; `MPPIAttitudeExecutor.__init__(plane, env, phi_s, hdot_s, va_s, seed=0)` replaces the `MPPI_SEED` env read).

- [ ] **Step 1: Rename and copy the checkpoints**

```bash
cd $OLD/trained_agents && mkdir -p $NEW/checkpoints && python3 - <<'PY'
import shutil, os
NEW="/home/matteo/Projects/falcon-s/checkpoints"
old_stem = {("ppo","altitude"): "warp_ppo_{ac}_dynamic", ("sac","altitude"): "warp_sac_{ac}_dynamic",
            ("td3","altitude"): "warp_td3_{ac}_dynamic", ("ppo","attitude"): "warp_attitude_{ac}_dynamic",
            ("sac","attitude"): "warp_sac_attitude_{ac}_dynamic", ("td3","attitude"): "warp_td3_attitude_{ac}_dynamic"}
rows = []
for (algo, task), stem in old_stem.items():
    for ac in ("Airship_V7", "Volantex_Ranger"):
        for s, suf in ((0, ""), (1, "_s1"), (2, "_s2")):
            src = stem.format(ac=ac) + suf + ".pt"; dst = f"{algo}_{task}_{ac}_s{s}.pt"
            shutil.copy(src, f"{NEW}/{dst}"); rows.append((src, dst))
for ac in ("Airship_A0S", "Navion", "Cirrus_SR22"):          # ground-effect closed-loop column only
    src = f"warp_ppo_{ac}_dynamic.pt"; dst = f"ppo_altitude_{ac}_s0.pt"
    shutil.copy(src, f"{NEW}/{dst}"); rows.append((src, dst))
open("/home/matteo/Projects/falcon-s-goldens/golden/CHECKPOINT_MAP.tsv","w").write("\n".join(f"{a}\t{b}" for a,b in rows)+"\n")
print(len(rows), "checkpoints copied")
PY
ls $NEW/checkpoints | wc -l      # 39
du -sh $NEW/checkpoints           # ~55-60 MB
```

- [ ] **Step 2: Copy the controller packages and rewrite imports**

```bash
mkdir -p $NEW/src/falcons/controllers && cd $NEW/src/falcons/controllers && touch __init__.py
cp -r $OLD/controllers/lqr lqr; cp -r $OLD/controllers/mppi mppi
find . -name "__pycache__" -prune -exec rm -rf {} +
grep -rl "controllers\.\|backends\|utils\.\|data\.vehicle_config\|envs\.\|train\.configs\|eval\." lqr mppi | xargs sed -i \
  -e 's/from controllers\./from falcons.controllers./g' -e 's/import controllers\./import falcons.controllers./g' \
  -e 's/backends\.warp\.sim/falcons.sim.warp/g' -e 's/backends\.cpu/falcons.sim.cpu/g' -e 's/AircraftModelControl/Aircraft/g' \
  -e 's/from utils\.config import AircraftConfigManager/from falcons.aircraft.config import AircraftConfig/' \
  -e 's/AircraftConfigManager(/AircraftConfig(/g' -e 's/\.load_config()/.load()/g' -e 's/\.get_lqr_gains_path()/.lqr_gains_path/g' \
  -e 's/data\.vehicle_config/falcons.aircraft.params/g' -e 's/create_vehicle_config_from_json/load_params/g' \
  -e 's/from envs\.cpu_env import CoreAircraftEnv/from falcons.envs.cpu import CpuEnv/' -e 's/CoreAircraftEnv/CpuEnv/g' \
  -e 's/from train\.configs/from falcons.envs.configs/g' \
  -e 's/from utils\.util import Trajectory/from falcons.envs.trajectory import Trajectory/'
grep -rn "gymnasium\|import gym\|Trajectory(\|\.plot(" lqr mppi
```

For each `if __name__ == "__main__":` block in `lqr/control.py` and `mppi/control.py` (the demo runs that build a `Trajectory` and call `.plot(...)`): delete the block. Delete the `gymnasium` import if it only served the deleted block. `controllers/lqr/control.py:54` reads `os.environ.get("LQR_GAINS_PATH") or self.airship.config_manager.lqr_gains_path` — after the sed that is what it says; keep it.

- [ ] **Step 3: Make the MPPI attitude seed a parameter**

In `mppi/attitude.py`, the constructor ends with the `MPPI_SEED` block (old lines 105–113). Replace:

```python
    def __init__(self, aircraft, env, phi_s, hdot_s, va_s, seed=0):
        ...
        # Pin the sampling noise. The warp aircraft base otherwise seeds this from entropy. This alone
        # does not make a rollout reproducible: the rollout-cost reduction in kernels.py uses
        # wp.atomic_add, whose thread ordering is not deterministic, so a single rollout is one draw.
        # Callers fly several draws with different seeds and report the spread.
        if seed is not None:
            self.mppi.seed(int(seed))
```

Same in `$OLD/eval/controllers/mppi.py:97` when it becomes `MPPIAltitude` in Step 5.

- [ ] **Step 4: Write `policies.py` (replaces both `_policy.py` files)**

```python
# src/falcons/controllers/policies.py
"""The one place that knows how checkpoints are named and loaded.

checkpoints/{algo}_{task}_{plane}_s{seed}.pt
  ppo -> falcons.train.ppo.Agent            (.actor_mean)
  sac -> falcons.train.sac.SquashedActor    (.mean_action)
  td3 -> falcons.train.td3.DetActor         (.mean_action)
lqr / mppi are not checkpoints: they are the classical attitude executors, built on an env.
"""
import os
from pathlib import Path

import torch

from falcons.paths import CKPT_DIR

ALGOS = ("ppo", "sac", "td3")
CLASSICAL = ("lqr", "mppi")
TASKS = ("altitude", "attitude")
OBS_ACT = {"altitude": (11, 2), "attitude": (15, 4)}   # observation and action widths per task


def ckpt_path(algo, task, plane, seed=0, ckpt_dir=CKPT_DIR) -> Path:
    return Path(ckpt_dir) / f"{algo}_{task}_{plane}_s{seed}.pt"


def seeds(algo, task, plane, ckpt_dir=CKPT_DIR) -> list:
    """Seeds that exist for this (algo, task, plane), ascending. Classical executors report [0]."""
    if algo in CLASSICAL:
        return [0]
    return sorted(int(p.stem.rsplit("_s", 1)[1]) for p in Path(ckpt_dir).glob(f"{algo}_{task}_{plane}_s*.pt"))


class Policy:
    """Uniform `.act(obs) -> action` over the three actor classes."""
    def __init__(self, fn):
        self.act = fn


def load_policy(algo, task, plane, seed=0, device="cuda", env=None, stream=None, ckpt_dir=CKPT_DIR):
    if algo in CLASSICAL:
        if env is None:
            raise ValueError(f"{algo} is a classical executor and needs env=")
        if task != "attitude":
            raise ValueError("classical executors load through falcons.controllers.altitude for the altitude task")
        if algo == "lqr":
            from falcons.controllers.lqr.attitude import LQRAttitudeExecutor
            return Policy(LQRAttitudeExecutor(plane, env).actor_mean)
        from falcons.controllers.mppi.attitude import MPPIAttitudeExecutor
        if stream is None:
            raise ValueError("mppi needs stream=(phi, hdot, va) for its preview horizon")
        return Policy(MPPIAttitudeExecutor(plane, env, *stream, seed=seed).actor_mean)
    path = ckpt_path(algo, task, plane, seed, ckpt_dir)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint {path}; have seeds {seeds(algo, task, plane, ckpt_dir)}")
    state = torch.load(path, map_location=device)
    n_obs, n_act = OBS_ACT[task]
    if algo == "ppo":
        from falcons.train.ppo import Agent
        net = Agent(n_obs, n_act).to(device); net.load_state_dict(state); net.eval()
        return Policy(net.actor_mean)
    if algo == "sac":
        from falcons.train.sac import SquashedActor
        net = SquashedActor(n_obs, n_act).to(device); net.load_state_dict(state); net.eval()
        return Policy(net.mean_action)
    from falcons.train.td3 import DetActor
    net = DetActor(n_obs, n_act).to(device); net.load_state_dict(state); net.eval()
    return Policy(net.mean_action)
```

`Agent`, `SquashedActor`, `DetActor` come from `falcons.train`, created in Task 9; until then copy the three class definitions verbatim into `src/falcons/train/{ppo,sac,td3}.py` now (only the classes and `layer_init`/`mlp` helpers they need, plus `NET` constants) so this task is testable. Task 9 fills in the training loops around them.

- [ ] **Step 5: `altitude.py` — the two acquisition adapters**

```bash
cd $NEW/src/falcons/controllers
{ echo '"""LQR and MPPI on the altitude-acquisition scenario: the classical rows of the altitude table.'
  echo 'Both read the reference constants (TRAJ_CONST_ALT, TRAJ_ACQ_H0, TRAJ_REF_VA, LQR_GAINS_PATH) that'
  echo 'falcons.benchmark.altitude bakes into the child process environment before importing them."""'
  cat $OLD/eval/controllers/lqr.py; echo; cat $OLD/eval/controllers/mppi.py; } > altitude.py
sed -i -e 's/from controllers\./from falcons.controllers./g' -e 's/backends\.warp\.sim\.aircraft import AircraftModelControl/falcons.sim.warp.aircraft import Aircraft/' \
       -e 's/AircraftModelControl/Aircraft/g' -e 's/from data\.vehicle_config import create_vehicle_config_from_json/from falcons.aircraft.params import load_params/' \
       -e 's/create_vehicle_config_from_json/load_params/g' -e 's/^class LQRAdapter/class LQRAltitude/' -e 's/^class MPPIAdapter/class MPPIAltitude/' \
       -e 's/    def run_scenario(self, scenario: dict) -> dict:/    def run(self, scenario: dict) -> dict:/' altitude.py
```

Deduplicate the merged import block. In `MPPIAltitude.__init__` add `mppi_seed=0` and replace the `os.environ.get("MPPI_SEED")` block with `self.mppi.seed(int(mppi_seed))` (same shape as Step 3). Leave `MPPI_SAMPLES` / `MPPI_HORIZON` env reads in place — they are the paper's defaults (1000, 100); turn them into keyword arguments `samples=1000, horizon=100` with the env read removed.

- [ ] **Step 6: Tests**

```python
# tests/test_checkpoints.py
import pytest, torch
from falcons.controllers.policies import ALGOS, TASKS, OBS_ACT, seeds, load_policy
from falcons.paths import CKPT_DIR

CELLS = [(a, t, p, s) for a in ALGOS for t in TASKS for p in ("Airship_V7", "Volantex_Ranger") for s in (0, 1, 2)]
CELLS += [("ppo", "altitude", p, 0) for p in ("Airship_A0S", "Navion", "Cirrus_SR22")]


def test_thirty_nine_checkpoints_ship():
    assert len(list(CKPT_DIR.glob("*.pt"))) == 39


@pytest.mark.cuda
@pytest.mark.parametrize("algo,task,plane,seed", CELLS, ids=[f"{a}_{t}_{p}_s{s}" for a, t, p, s in CELLS])
def test_checkpoint_loads_and_acts(algo, task, plane, seed):
    assert seed in seeds(algo, task, plane)
    pol = load_policy(algo, task, plane, seed)
    n_obs, n_act = OBS_ACT[task]
    obs = torch.zeros(4, n_obs, device="cuda")
    a1, a2 = pol.act(obs), pol.act(obs)
    assert a1.shape == (4, n_act) and torch.equal(a1, a2)
```

```python
# tests/test_controllers.py
import numpy as np, pytest
from falcons.aircraft.config import AircraftConfig


def test_lqr_gains_identical_to_archive():
    for ac in ("Airship_V7", "Volantex_Ranger", "Navion", "Cirrus_SR22"):
        new = np.loadtxt(AircraftConfig(ac).lqr_gains_path, delimiter=",")
        old = np.loadtxt(f"/home/matteo/Projects/WIG_Plane_RL_Control/data/aircraft/{ac}/" +
                         {"Airship_V7": "airship_v7", "Volantex_Ranger": "volantex_ranger", "Navion": "Navion",
                          "Cirrus_SR22": "Cirrus_SR22"}[ac] + "_K_LQR.csv", delimiter=",")
        np.testing.assert_array_equal(new, old)


@pytest.mark.cuda
def test_mppi_attitude_executor_builds_with_seed():
    import warp as wp, torch
    from falcons.envs.attitude import AttitudeEnv
    from falcons.envs.configs import attitude_env_cfg
    from falcons.controllers.policies import load_policy
    wp.init()
    env = AttitudeEnv(1, "cuda", attitude_env_cfg("Airship_V7", horizon=52)); env.reset()
    z = np.zeros(50, dtype=np.float32)
    pol = load_policy("mppi", "attitude", "Airship_V7", env=env, stream=(z, z, z + env.base_vel), seed=3)
    env._obs_launch(); a = pol.act(wp.to_torch(env._obs))
    assert a.shape[-1] == 4
```

Run: `cd $NEW && .venv/bin/pytest tests/test_checkpoints.py tests/test_controllers.py -q` → 41 passed.

- [ ] **Step 7: Commit**

```bash
cd $NEW && git add -A && git commit -q -m "controllers: lqr, mppi, one checkpoint loader, 39 renamed checkpoints"
```

---

### Task 7a: `benchmark/` scaffolding + altitude protocol

**Files:**
- Create: `src/falcons/benchmark/__init__.py`, `src/falcons/benchmark/diff.py`, `src/falcons/benchmark/altitude.py`, `tests/golden/*` (copied from Task 0), `tests/test_benchmark_altitude.py`
- Source: `$OLD/trl_paper/altitude/protocol.py`

**Interfaces:**
- Produces: `falcons.benchmark.TOLERANCE: dict[str, dict]`; `falcons.benchmark.diff.compare_csv(new: Path, gold: Path, rule: dict) -> list[str]` (empty list = pass); `falcons.benchmark.altitude.episode_metrics(h, target, h0, thr) -> dict`; `aggregate(rows) -> dict`; `run_learned(plane, algo, seed, ckpt_dir) -> list[dict]`; `run_classical(plane, algo, mppi_draws, ckpt_dir) -> list[dict]`; `run(planes, ckpt_dir, results_dir, algos=("ppo","sac","td3","lqr","mppi"), mppi_draws=3) -> Path` writing `results/altitude.csv` with the archive's exact column set (`aircraft, method, n, n_seeds, survival, survival_std, acquired, acquired_std, rmse, rmse_std, settling, settling_std, settled_frac, overshoot, overshoot_std, energy, energy_std`); `python -m falcons.benchmark.altitude --child PLANE ALGO TARGET H0` prints one `JSON{…}` line.

- [ ] **Step 1: Tolerance table and CSV diff**

```python
# src/falcons/benchmark/__init__.py
"""Benchmark protocols. TOLERANCE is the single statement of what "reproduces" means per table:
learned and LQR rows are exact (seeded rollouts are bit-stable); MPPI is a sampling planner whose
cost reduction uses non-deterministic atomics, so its rows must land within the shipped spread;
wind cells are one Dryden draw each."""
TOLERANCE = {
    "altitude.csv":   {"key": ["aircraft", "method"], "exact_if": {"method": ["PPO", "SAC", "TD3", "LQR"]},
                       "sigma_if": {"method": ["MPPI"]}, "sigma_suffix": "_std", "abs": 0.0},
    "ge_trim.csv":    {"key": ["aircraft", "h_over_b"], "abs": 1e-9},
    "ge_energy.csv":  {"key": ["aircraft", "h_over_b"], "abs": 0.0},
    "maneuvers.csv":  {"key": ["aircraft", "maneuver", "method"], "exact_if": {"method": ["PPO", "SAC", "TD3", "LQR"]},
                       "sigma_if": {"method": ["MPPI"]}, "sigma_cols": {"phi_rmse_deg": "phi_rmse_std"}, "abs": 0.0},
    "robustness.csv": {"key": ["aircraft", "method", "disturbance", "severity"],
                       "exact_if": {"disturbance": ["nominal", "obs_noise", "sensor_delay", "action_delay"]},
                       "loose_if": {"disturbance": ["wind"]}, "loose_abs": {"survival": 0.25, "phi_rmse": 5.0}, "abs": 0.0},
    "throughput.csv": {"key": ["backend", "n_envs"], "rel": 0.5},   # wall-clock: hardware-dependent, order of magnitude
}
```

```python
# src/falcons/benchmark/diff.py
"""Compare a regenerated CSV against a golden one under a TOLERANCE rule. Returns a list of
human-readable mismatches; an empty list is a pass."""
import csv
import math


def _rows(path, key):
    with open(path) as f:
        rs = list(csv.DictReader(f))
    return {tuple(r[k] for k in key): r for r in rs}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def compare_csv(new, gold, rule):
    key = rule["key"]
    a, b = _rows(new, key), _rows(gold, key)
    out = []
    if set(a) != set(b):
        out.append(f"row sets differ: missing {sorted(set(b) - set(a))}, extra {sorted(set(a) - set(b))}")
    for k in sorted(set(a) & set(b)):
        ra, rb = a[k], b[k]
        mode = "exact"
        for col, vals in rule.get("sigma_if", {}).items():
            if ra.get(col) in vals: mode = "sigma"
        for col, vals in rule.get("loose_if", {}).items():
            if ra.get(col) in vals: mode = "loose"
        for col in rb:
            if col in key or col.endswith(rule.get("sigma_suffix", "_std")):
                continue
            va, vb = _num(ra.get(col)), _num(rb.get(col))
            if va is None or vb is None:
                if (ra.get(col) or "") != (rb.get(col) or ""): out.append(f"{k} {col}: {ra.get(col)!r} != {rb.get(col)!r}")
                continue
            if math.isnan(va) and math.isnan(vb):
                continue
            if mode == "sigma":
                sd_col = rule.get("sigma_cols", {}).get(col, col + rule.get("sigma_suffix", "_std"))
                tol = max(_num(rb.get(sd_col)) or 0.0, rule.get("abs", 0.0))
            elif mode == "loose":
                tol = rule["loose_abs"].get(col, rule.get("abs", 0.0))
            elif "rel" in rule:
                tol = abs(vb) * rule["rel"]
            else:
                tol = rule.get("abs", 0.0)
            if abs(va - vb) > tol:
                out.append(f"{k} {col}: {va} vs golden {vb} (tol {tol})")
    return out
```

```bash
mkdir -p $NEW/tests/golden && cp /home/matteo/Projects/falcon-s-goldens/golden/*.csv /home/matteo/Projects/falcon-s-goldens/golden/*.json $NEW/tests/golden/
cat > $NEW/tests/test_diff.py <<'EOF'
from falcons.benchmark import TOLERANCE
from falcons.benchmark.diff import compare_csv

def test_golden_equals_itself():
    for name, rule in TOLERANCE.items():
        p = f"tests/golden/{name}"
        try: open(p)
        except FileNotFoundError: continue
        assert compare_csv(p, p, rule) == []
EOF
cd $NEW && .venv/bin/pytest tests/test_diff.py -q
```

- [ ] **Step 2: Port `protocol.py` → `benchmark/altitude.py`**

```bash
cp $OLD/trl_paper/altitude/protocol.py $NEW/src/falcons/benchmark/altitude.py && cd $NEW/src/falcons/benchmark
sed -i -e 's/from eval\.altitude\._policy import load_policy/from falcons.controllers.policies import load_policy, seeds/' \
       -e 's/from eval\.altitude\._policy import seed_suffixes/from falcons.controllers.policies import seeds/' \
       -e 's/from envs\.altitude import WarpAltitudeEnv/from falcons.envs.altitude import AltitudeEnv/' -e 's/WarpAltitudeEnv/AltitudeEnv/g' \
       -e 's/from train\.configs/from falcons.envs.configs/g' \
       -e 's/from eval\.controllers\.lqr import LQRAdapter/from falcons.controllers.altitude import LQRAltitude/' \
       -e 's/from eval\.controllers\.mppi import MPPIAdapter/from falcons.controllers.altitude import MPPIAltitude/' \
       -e 's/LQRAdapter/LQRAltitude/g' -e 's/MPPIAdapter/MPPIAltitude/g' -e 's/\.run_scenario(/.run(/g' \
       -e 's/"-m", "trl_paper\.altitude\.protocol"/"-m", "falcons.benchmark.altitude"/' altitude.py
```

Then edit by hand, in this order:
1. Delete `OUT = …`, `ACQ_GAINS = …`, `CL_REPS = …`. Delete the `envelope`, `run_rl_envelope` functions and the `ENV_TARGET/ENV_OFFSETS` constants (the envelope table is not in the paper).
2. `run_rl(aircraft, algo, seed="")` → `run_learned(plane, algo, seed=0, ckpt_dir=CKPT_DIR)`; the `load_policy(algo, aircraft, 11, 2, "cuda", seed=seed)` call becomes `load_policy(algo, "altitude", plane, seed, ckpt_dir=ckpt_dir)` and `.actor_mean` → `.act`.
3. `run_classical(aircraft, algo)` → `run_classical(plane, algo, mppi_draws=3, ckpt_dir=CKPT_DIR)`: the gains path comes from `AircraftConfig(plane).acquisition_gains_path`; drop `PYTHONPATH=os.getcwd()` from the child env (the package is installed); MPPI repetitions loop `range(mppi_draws)` passing `MPPI_DRAW=k` in the env, and `child` passes `mppi_seed=int(os.environ.get("MPPI_DRAW", 0))` to `MPPIAltitude`.
4. `main(planes)` → `run(planes, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR, algos=RL_ALGOS + CL_ALGOS, mppi_draws=3)`; `seed_suffixes(algo, ac)` → `seeds(algo, "altitude", ac, ckpt_dir)`; the CSV is written to `Path(results_dir) / "altitude.csv"` with the same `cols` list. Return the path.
5. `if __name__ == "__main__":` handles only `--child PLANE ALGO TARGET H0`; the full run is reached through the CLI (Task 9).

Import `CKPT_DIR, RESULTS_DIR` from `falcons.paths` and `AircraftConfig` from `falcons.aircraft.config`.

- [ ] **Step 3: Gate — regenerate the learned rows and diff**

```python
# tests/test_benchmark_altitude.py
import csv, pytest
from falcons.benchmark import TOLERANCE
from falcons.benchmark.diff import compare_csv
from falcons.benchmark.altitude import episode_metrics, aggregate, run_learned, run


def test_episode_metrics_band_is_one_metre_and_overshoot_is_past_target():
    import numpy as np
    h = np.full(6000, 40.0); h[:100] = np.linspace(30, 41.5, 100); thr = np.full(6000, 0.4)
    m = episode_metrics(h, 40.0, 30.0, thr)
    assert m["overshoot"] == pytest.approx(1.5, abs=1e-6)          # excursion past the target, not |e|
    assert m["settled"] and abs(m["rmse"]) < 1e-9


@pytest.mark.cuda
def test_learned_rows_match_golden_volantex_ppo():
    a = aggregate(run_learned("Volantex_Ranger", "ppo", 0))
    gold = next(r for r in csv.DictReader(open("tests/golden/altitude.csv"))
                if r["aircraft"] == "Volantex_Ranger" and r["method"] == "PPO")
    # golden row is the 3-seed mean; seed 0 alone must equal it where std is 0 (survival, acquired)
    assert a["survival"] == float(gold["survival"]) and a["survival"] * a["settled_frac"] == pytest.approx(float(gold["acquired"]))


@pytest.mark.slow
@pytest.mark.cuda
def test_full_altitude_table_matches_golden(tmp_path):
    out = run(["Airship_V7", "Volantex_Ranger"], results_dir=tmp_path)
    assert compare_csv(out, "tests/golden/altitude.csv", TOLERANCE["altitude.csv"]) == []
```

Run: `cd $NEW && .venv/bin/pytest tests/test_benchmark_altitude.py -q -m "not slow"` → 2 passed. Then the slow one once: `.venv/bin/pytest tests/test_benchmark_altitude.py -q -m slow` (learned rows exact, MPPI within σ; ~30–60 min).

- [ ] **Step 4: Commit**

```bash
cd $NEW && git add -A && git commit -q -m "benchmark: tolerance table, csv diff, altitude protocol"
```

---

### Task 7b: `benchmark/ground_effect.py`

**Files:**
- Create: `src/falcons/benchmark/ground_effect.py`, `tests/test_benchmark_ge.py`
- Source: `$OLD/trl_paper/altitude/ge_trim.py`, `$OLD/trl_paper/altitude/ge_energy.py`

**Interfaces:**
- Produces: `run_trim(planes, results_dir) -> Path` (`ge_trim.csv`), `run_energy(planes, ckpt_dir, results_dir) -> Path` (`ge_energy.csv`), `theory_pct(h, P)`, `geometry(plane)`, `thrust_saving(...)`, `RATIOS`. Plotting functions are NOT ported here (Task 8).

- [ ] **Step 1: Merge the two scripts**

```bash
cd $NEW/src/falcons/benchmark
{ echo '"""Ground effect as the thrust required to fly: an open-loop level-flight trim at each height over span'
  echo '(what the airframe is offered) and a policy holding the band (what a controller collects)."""'
  cat $OLD/trl_paper/altitude/ge_trim.py; echo; echo "# ───── closed loop ─────"; cat $OLD/trl_paper/altitude/ge_energy.py; } > ground_effect.py
grep -n "^from scripts\|^import scripts\|scripts\." ground_effect.py
```

`ge_energy.py` imported something from `scripts/` — open that import, copy the function it names into `ground_effect.py` verbatim (it is one of `scripts/ground_effect.py`'s helpers), and delete the import. Then deduplicate the import block, keep one `RATIOS`, one `PLANES` (rename to `GE_PLANES` and set it from `falcons.aircraft.config.PLANES`), one `COLORS`.

- [ ] **Step 2: Rewrite imports and signatures**

```bash
sed -i -e 's/from backends\.warp\.sim\.aircraft import AircraftModelControl/from falcons.sim.warp.aircraft import Aircraft/' -e 's/AircraftModelControl/Aircraft/g' \
       -e 's/from data\.vehicle_config import create_vehicle_config_from_json/from falcons.aircraft.params import load_params/' -e 's/create_vehicle_config_from_json/load_params/g' \
       -e 's/from eval\.altitude\._policy import load_policy/from falcons.controllers.policies import load_policy/' \
       -e 's/from envs\.altitude import WarpAltitudeEnv/from falcons.envs.altitude import AltitudeEnv/' -e 's/WarpAltitudeEnv/AltitudeEnv/g' \
       -e 's/from train\.configs/from falcons.envs.configs/g' ground_effect.py
```

By hand: `main(planes)` of the trim half → `run_trim(planes, results_dir=RESULTS_DIR)` writing `results_dir/"ge_trim.csv"`; `main(planes)` of the energy half → `run_energy(planes, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR)` writing `ge_energy.csv`; `load_policy("ppo", aircraft, 11, 2, "cuda")` → `load_policy("ppo", "altitude", aircraft, 0, ckpt_dir=ckpt_dir)` and `.actor_mean` → `.act`. Move `figure(rows)`, `plot(rows)`, `replot()` to a scratch file `/home/matteo/Projects/falcon-s-goldens/ge_plots.py` for Task 8 and delete them here. Delete every `OUT = …` and the `__main__` blocks.

- [ ] **Step 3: Gate**

```python
# tests/test_benchmark_ge.py
import pytest
from falcons.benchmark import TOLERANCE
from falcons.benchmark.diff import compare_csv
from falcons.benchmark.ground_effect import run_trim, run_energy, theory_pct, geometry
from falcons.aircraft.config import PLANES


def test_theory_is_lifting_line_shape():
    P = geometry("Airship_V7")
    assert theory_pct(0.25 * P["span"], P) > theory_pct(6.0 * P["span"], P) > 0


@pytest.mark.cuda
def test_trim_table_matches_golden(tmp_path):
    out = run_trim(PLANES, results_dir=tmp_path)
    assert compare_csv(out, "tests/golden/ge_trim.csv", TOLERANCE["ge_trim.csv"]) == []


@pytest.mark.slow
@pytest.mark.cuda
def test_energy_table_matches_golden(tmp_path):
    out = run_energy(PLANES, results_dir=tmp_path)
    assert compare_csv(out, "tests/golden/ge_energy.csv", TOLERANCE["ge_energy.csv"]) == []
```

Check `geometry()`'s return keys in the source before writing the first assertion (it returns span/TR/AR-style values; use the key it actually exposes for span). Run: `.venv/bin/pytest tests/test_benchmark_ge.py -q -m "not slow"` → 2 passed; then `-m slow` once.

- [ ] **Step 4: Commit** — `git commit -q -m "benchmark: ground effect (trim + closed loop)"`

---

### Task 7c: `benchmark/maneuvers.py`

**Files:**
- Create: `src/falcons/benchmark/maneuvers.py`, `src/falcons/benchmark/streams.py`, `tests/test_benchmark_maneuvers.py`
- Source: `$OLD/trl_paper/maneuvers/{harness,streams,plot_track}.py` (the flying/aggregating parts of `plot_track`; `facet()` goes to Task 8)

**Interfaces:**
- Produces: `falcons.benchmark.streams.{maneuver_stream(name, phi_max_full, hdot_max_full, base_vel), MANEUVERS, DT}`; `falcons.benchmark.maneuvers.make_env(plane, steps, turbulence=None)`; `fly(plane, maneuver, controller_factory, turbulence=None, turb_W20=None) -> dict` (was `run_maneuver`, always `return_trace=True`); `time_in_band(trace, tol_deg, settle=50)`; `aggregate(reps, tol_deg)`; `controllers(plane, ckpt_dir) -> dict[method, list[(seed, factory)]]`; `cell(plane, maneuver, method, ckpt_dir, cache_dir, refresh=False, mppi_draws=5, mppi_seed=0, tol_deg=None) -> dict` (was `cached_run`); `run(planes, ckpt_dir, results_dir, refresh=False, mppi_draws=5, mppi_seed=0) -> Path` writing `maneuvers.csv` (archive columns incl. `n_runs`, `phi_rmse_std`) and `mppi_variance.json`.

- [ ] **Step 1: Copy and rewrite**

```bash
cd $NEW/src/falcons/benchmark
cp $OLD/trl_paper/maneuvers/streams.py streams.py
{ cat $OLD/trl_paper/maneuvers/harness.py; echo; echo "# ───── cells, caching, table ─────"; cat $OLD/trl_paper/maneuvers/plot_track.py; } > maneuvers.py
sed -i -e 's/from envs\.attitude import WarpAttitudeEnv/from falcons.envs.attitude import AttitudeEnv/' -e 's/WarpAttitudeEnv/AttitudeEnv/g' \
       -e 's/from train\.configs/from falcons.envs.configs/g' \
       -e 's/from trl_paper\.maneuvers\.streams import/from falcons.benchmark.streams import/' \
       -e 's/from trl_paper\.maneuvers\.harness import run_maneuver/from falcons.benchmark.maneuvers import fly/' \
       -e 's/from eval\.attitude\._policy import load_policy, seed_suffixes/from falcons.controllers.policies import load_policy, seeds/' \
       -e 's/from eval\.attitude\._policy import load_policy/from falcons.controllers.policies import load_policy/' \
       -e 's/def run_maneuver(/def fly(/' -e 's/run_maneuver(/fly(/g' -e 's/def cached_run(/def cell(/' -e 's/cached_run(/cell(/g' maneuvers.py
```

By hand:
- Delete `OUT`, `CACHE`, `REFRESH`, `REPS`, the `facet()` function (move it verbatim to `/home/matteo/Projects/falcon-s-goldens/facet.py` for Task 8), the `if __name__` block, and every `matplotlib` import that only `facet` used.
- `controllers(aircraft)` → `controllers(plane, ckpt_dir=CKPT_DIR)`: `seed_suffixes(algo, aircraft)` → `seeds(algo, "attitude", plane, ckpt_dir)`; the factory calls `load_policy(algo, "attitude", plane, seed, env=env, stream=(phi_s, hdot_s, va_s), ckpt_dir=ckpt_dir).act`. For MPPI the factory must forward the draw seed: `load_policy("mppi", …, seed=draw)`, so make the factory signature `factory(env, phi_s, hdot_s, va_s, draw=0)`.
- `cell(...)`: signature as in Interfaces. Replace the `os.environ["MPPI_SEED"] = str(k)` loop with `factory(e, p, h, v, draw=mppi_seed + k)`; cache path `Path(cache_dir) / f"{plane}_{maneuver}_{algo}{suf}{'' if n == 1 else f'_r{n}'}.pkl"` — **same file names as the archive** so the 39 MB trace fixture replays.
- `main()` → `run(planes, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR, refresh=False, mppi_draws=5, mppi_seed=0)`; `cache_dir = Path(results_dir) / "traces"`; writes `maneuvers.csv` and `mppi_variance.json`; returns the CSV path. The `facet(...)` call inside the loop is removed (Task 8 re-reads the cache).

- [ ] **Step 2: Gate on the trace fixture**

```python
# tests/test_benchmark_maneuvers.py
import shutil, pytest
from pathlib import Path
from falcons.benchmark import TOLERANCE
from falcons.benchmark.diff import compare_csv
from falcons.benchmark.streams import maneuver_stream, MANEUVERS

FIX = Path("/home/matteo/Projects/falcon-s-goldens/traces")


def test_streams_have_four_names_and_equal_length():
    s = [maneuver_stream(m, 0.6, 1.5, 28.0) for m in MANEUVERS]
    assert len(MANEUVERS) == 4 and all(len(x[0]) == len(x[1]) == len(x[2]) for x in s)


@pytest.mark.cuda
def test_table_replays_from_cached_traces(tmp_path):
    from falcons.benchmark.maneuvers import run
    shutil.copytree(FIX, tmp_path / "traces")
    out = run(["Airship_V7", "Volantex_Ranger"], results_dir=tmp_path)      # nothing re-flies
    assert compare_csv(out, "tests/golden/maneuvers.csv", {**TOLERANCE["maneuvers.csv"], "sigma_if": {}}) == []  # exact, MPPI included


@pytest.mark.slow
@pytest.mark.cuda
def test_one_learned_cell_reflies_identically(tmp_path):
    from falcons.benchmark.maneuvers import cell, controllers
    c = cell("Volantex_Ranger", "helix", "PPO", ckpt_dir=None, cache_dir=tmp_path, refresh=True, tol_deg=8.6)
    assert abs(c["phi"]["rmse"] - 0.42) < 0.005      # golden maneuvers.csv: Volantex helix PPO 0.42 ± 0.02
```

The second test is the gate: replaying the archive's cache through the new code must reproduce every cell including MPPI, exactly. Run: `.venv/bin/pytest tests/test_benchmark_maneuvers.py -q -m "not slow"` → 2 passed. For `cell(..., ckpt_dir=None)` pass `CKPT_DIR` explicitly if `None` is not accepted.

- [ ] **Step 3: Commit** — `git commit -q -m "benchmark: maneuver streams, harness, cells with per-seed cache"`

---

### Task 7d: `benchmark/robustness.py`

**Files:**
- Create: `src/falcons/benchmark/robustness.py`, `tests/test_benchmark_robustness.py`
- Source: `$OLD/trl_paper/maneuvers/robustness.py` (minus `_plot`, which goes to Task 8)

**Interfaces:**
- Produces: `run(planes, ckpt_dir, results_dir) -> Path` (`robustness.csv`, archive columns `aircraft, method, disturbance, severity, survival, n_survive, phi_rmse, hdot_rmse, va_rmse`), `eval_cell(...)`, `calibrate_turb(plane)`, `obs_noise_sigma(sev, va_scale, vz_scale)`, `SEVERITIES`, `ACTION_DELAY_MS`, `SENSOR_DELAY_MS`, `WIND_IU`.

- [ ] **Step 1: Port**

```bash
cp $OLD/trl_paper/maneuvers/robustness.py $NEW/src/falcons/benchmark/robustness.py && cd $NEW/src/falcons/benchmark
sed -i -e 's/from trl_paper\.maneuvers\.harness import/from falcons.benchmark.maneuvers import/' -e 's/run_maneuver/fly/g' -e 's/make_env/make_env/g' \
       -e 's/from trl_paper\.maneuvers\.streams import/from falcons.benchmark.streams import/' \
       -e 's/from eval\.attitude\._policy import load_policy/from falcons.controllers.policies import load_policy/' \
       -e 's/from envs\.attitude import WarpAttitudeEnv/from falcons.envs.attitude import AttitudeEnv/' -e 's/WarpAttitudeEnv/AttitudeEnv/g' \
       -e 's/from train\.configs/from falcons.envs.configs/g' robustness.py
```

By hand: delete `OUT`; `controllers(aircraft)` uses `load_policy("ppo", "attitude", plane, 0, …).act`; `main()` → `run(planes, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR)` writing `robustness.csv`; move `_plot(rows)` to `/home/matteo/Projects/falcon-s-goldens/robustness_plot.py`; delete `__main__`.

- [ ] **Step 2: Gate**

```python
# tests/test_benchmark_robustness.py
import pytest
from falcons.benchmark import TOLERANCE
from falcons.benchmark.diff import compare_csv
from falcons.benchmark.robustness import steps_from_ms, obs_noise_sigma, SEVERITIES


def test_delay_steps_and_noise_monotone():
    assert steps_from_ms(20) == 2 and steps_from_ms(100) == 10
    s = [obs_noise_sigma(sev, 1.0, 1.0) for sev in SEVERITIES]
    assert all(a <= b for a, b in zip(s[0], s[1])) and all(a <= b for a, b in zip(s[1], s[2]))


@pytest.mark.slow
@pytest.mark.cuda
def test_table_matches_golden(tmp_path):
    from falcons.benchmark.robustness import run
    out = run(["Airship_V7", "Volantex_Ranger"], results_dir=tmp_path)
    assert compare_csv(out, "tests/golden/robustness.csv", TOLERANCE["robustness.csv"]) == []
```

If `obs_noise_sigma` returns a dict rather than a tuple, compare per key. Run fast tier; run the slow one once (78 cells, ~1–2 h): non-wind cells exact, wind cells within the loose bound.

- [ ] **Step 3: Commit** — `git commit -q -m "benchmark: robustness by disturbance family"`

---

### Task 7e: `benchmark/throughput.py`

**Files:**
- Create: `src/falcons/benchmark/throughput.py`, `tests/test_benchmark_throughput.py`
- Source: `$OLD/eval/backend/parity.py`, `$OLD/eval/backend/benchmark.py`, `$OLD/scripts/bench_sim_throughput.py`

**Interfaces:**
- Produces: `parity(plane, ckpt_dir, n_per_target=64, horizon=1500) -> dict` (warp-vs-torch settled RMSE per target for the same PPO checkpoint on both plants), `throughput(plane, sizes=(64,256,1024,4096,16384), trials=10, results_dir) -> Path` (`throughput.csv`: `backend, n_envs, steps_per_s_mean, steps_per_s_std, ci95`), `run(plane, ckpt_dir, results_dir) -> Path`.

- [ ] **Step 1: Merge the three scripts into functions**

Copy the three files into one, then convert each module-level script (`AC = sys.argv[…]`, `run_warp()/run_torch()`, `bench()`, `main()`) into the functions named above. The parity check loads `load_policy("ppo", "altitude", plane, 0, ckpt_dir=ckpt_dir)` and drives `falcons.envs.altitude.AltitudeEnv` and `falcons.sim.torch.altitude.AltitudeEnv` with the same targets `[20, 40, 60, 80, 100]`; `stats(vals)` (mean, sample std, 95 % CI with t(df=9)) is kept verbatim from `bench_sim_throughput.py`. The CPU single-env timing from `bench_cpu` uses `falcons.sim.cpu.Aircraft`. `run` writes `throughput.csv` and `results_dir/"parity.json"`.

- [ ] **Step 2: Gate**

```python
# tests/test_benchmark_throughput.py
import pytest


def test_stats_ci_matches_t_distribution():
    from falcons.benchmark.throughput import stats
    m, sd, ci = stats([1.0] * 9 + [2.0])
    assert m == pytest.approx(1.1) and ci == pytest.approx(2.262 * sd / 10 ** 0.5, rel=1e-3)


@pytest.mark.cuda
def test_parity_warp_vs_torch_within_half_metre(tmp_path):
    from falcons.benchmark.throughput import parity
    d = parity("Volantex_Ranger", n_per_target=16, horizon=800)
    for t in d["targets"]:
        assert abs(d["warp"][t] - d["torch"][t]) < 0.5
```

Run both (parity takes ~1 min). Then one real throughput run: `.venv/bin/python -c "from falcons.benchmark.throughput import run; run('Airship_V7', results_dir='results')"` and compare the 16384-env warp row to `tests/golden/throughput.json` — same order of magnitude is the pass criterion (hardware-bound).

- [ ] **Step 3: Commit** — `git commit -q -m "benchmark: cpu/gpu parity and step throughput"`

---

### Task 8: `benchmark/tables.py` and `benchmark/figures.py`

**Files:**
- Create: `src/falcons/benchmark/tables.py`, `src/falcons/benchmark/figures.py`, `tests/test_tables.py`
- Source: `$OLD/trl_paper/altitude/emit_tex.py`; `$OLD/eval/altitude/viz.py`; `$OLD/trl_paper/maneuvers/plot_attitude_paper.py`; the parked `facet.py`, `ge_plots.py`, `robustness_plot.py`; `$OLD/trl_paper/maneuvers/plot_robustness_paper.py`; `$OLD/trl_paper/maneuvers/render_strip.py` + `$OLD/eval/attitude/anim.py::aircraft_glyph`

**Interfaces:**
- Produces: `tables.emit(results_dir) -> Path` (`tables.tex`, byte-identical to the archive's `results_v3_tables.tex` given identical CSVs); `figures.FIGURES: dict[str, callable]` with keys `altitude_viz_{ppo,sac,td3}_{Airship_V7,Volantex_Ranger}`, `attitude_exec`, `maneuver_facet_Airship_V7`, `maneuver_render`, `ge_energy`, `robustness_ppo`; `figures.render(names, ckpt_dir, results_dir) -> list[Path]` writing under `results_dir/"figures"`.

- [ ] **Step 1: `tables.py`**

```bash
cp $OLD/trl_paper/altitude/emit_tex.py $NEW/src/falcons/benchmark/tables.py && cd $NEW/src/falcons/benchmark
sed -i -e 's|^ALT = .*||' -e 's|^MAN = .*||' -e 's|^OUT = .*||' tables.py
```

Replace `main()` with:

```python
def emit(results_dir=RESULTS_DIR):
    """results/*.csv -> results/tables.tex. Generated; never hand-edited."""
    d = Path(results_dir)
    body = ["% Generated by falcons.benchmark.tables — do not hand-edit.",
            r"% Requires \usepackage{booktabs,multirow,graphicx} in the preamble.", ""]
    body.append(protocol_table(rows(d / "altitude.csv")))
    body.append(ge_table(rows(d / "ge_energy.csv"), rows(d / "ge_trim.csv")))
    body.append(maneuver_table(rows(d / "maneuvers.csv")))
    body.append(robustness_table(rows(d / "robustness.csv")))
    out = d / "tables.tex"; out.write_text("\n".join(body)); return out
```

Test:

```python
# tests/test_tables.py
import shutil
from falcons.benchmark.tables import emit

GOLD = "/home/matteo/Projects/falcon-s-goldens/golden"


def test_tables_tex_byte_identical_to_archive(tmp_path):
    for n in ("altitude", "ge_trim", "ge_energy", "maneuvers", "robustness"):
        shutil.copy(f"tests/golden/{n}.csv", tmp_path / f"{n}.csv")
    got = emit(tmp_path).read_text().splitlines()[3:]        # drop the generator banner
    want = open(f"{GOLD}/tables.tex").read().splitlines()[3:]
    assert got == want
```

- [ ] **Step 2: `figures.py` — port each plotter as a function**

Build `figures.py` with one function per figure, each `(ckpt_dir, results_dir) -> Path`, from these sources, removing every `sys.argv`, `pop_algo()`, `OUT`, `CACHE`, and `__main__`:

| function | source | notes |
|---|---|---|
| `altitude_viz(algo, plane, …)` | `eval/altitude/viz.py` `run()`+`main()` | `ALGO/AC` globals → parameters; `load_policy(algo, "altitude", plane, 0, …).act`; writes `altitude_{algo}_viz_{plane}.pdf` |
| `attitude_exec(planes, …)` | `plot_attitude_paper.py` | `rollout(ac)` uses `fly`-style env + PPO seed 0; reads/writes `results_dir/traces/attitude_exec_{plane}.pkl` (same names as archive) |
| `maneuver_facet(plane, …)` | parked `facet.py` | re-reads the cell cache via `maneuvers.cell(..., refresh=False)` |
| `ge_energy(…)` | parked `ge_plots.py` `plot(rows)` | reads `ge_energy.csv` + `ge_trim.csv`; `subplots(2, 1, figsize=(3.5, 4.1))` unchanged |
| `robustness_ppo(…)` | `plot_robustness_paper.py` | reads `robustness.csv` |
| `maneuver_render(planes, …)` | `render_strip.py` + `anim.py::aircraft_glyph` | **rewritten**: for each (plane, maneuver) read the PPO trace from the cell cache, draw the 3-D track coloured by bank over ±50° with `Line3DCollection` and the glyph at the final pose, crop as `render_strip._strip` did, no mp4/ffmpeg |

Then:

```python
FIGURES = {
    **{f"altitude_viz_{a}_{p}": (lambda a=a, p=p: (lambda ckpt_dir, results_dir: altitude_viz(a, p, ckpt_dir, results_dir)))()
       for a in ("ppo", "sac", "td3") for p in ("Airship_V7", "Volantex_Ranger")},
    "attitude_exec": lambda ckpt_dir, results_dir: attitude_exec(("Airship_V7", "Volantex_Ranger"), ckpt_dir, results_dir),
    "maneuver_facet_Airship_V7": lambda ckpt_dir, results_dir: maneuver_facet("Airship_V7", ckpt_dir, results_dir),
    "maneuver_render": lambda ckpt_dir, results_dir: maneuver_render(("Airship_V7", "Volantex_Ranger"), ckpt_dir, results_dir),
    "ge_energy": lambda ckpt_dir, results_dir: ge_energy(results_dir),
    "robustness_ppo": lambda ckpt_dir, results_dir: robustness_ppo(results_dir),
}


def render(names=None, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR):
    (Path(results_dir) / "figures").mkdir(parents=True, exist_ok=True)
    return [FIGURES[n](ckpt_dir, results_dir) for n in (names or FIGURES)]
```

- [ ] **Step 3: Gate — regenerate all 11 with the trace fixture and compare by eye**

```bash
cd $NEW && cp -r /home/matteo/Projects/falcon-s-goldens/traces results/traces && cp tests/golden/*.csv results/
.venv/bin/python -c "from falcons.benchmark.figures import render; print(*render(), sep='\n')"
ls results/figures | wc -l          # 11
```

Open each against `/home/matteo/Projects/falcon-s-goldens/golden/figures/` side by side (`Read` the PNG/PDF pairs). Identical layout, labels, colours and annotated numbers is the pass; `maneuver_render` must show the same eight tracks with the same bank colouring — the exact camera/crop may differ and that is accepted.

- [ ] **Step 4: Commit** — `git commit -q -m "benchmark: tex tables and the eleven paper figures"`

---

### Task 9: `train/` and the full CLI (`train`, `eval`, `benchmark`, `tables`, `figures`)

**Files:**
- Create/complete: `src/falcons/train/{__init__,ppo,sac,td3,tasks,curves}.py`, `src/falcons/cli.py`, `scripts/train_seeds.sh`, `tests/test_train.py`
- Source: `$OLD/train/{ppo,sac,td3,tasks,curves}.py`

**Interfaces:**
- Produces: `falcons.train.tasks.{evaluate_altitude(agent, plane, steps=2000, settle=200), evaluate_attitude(agent, plane, steps=2000, settle=200), train_altitude(plane, steps=None, seed=0, ckpt_dir, curves_dir), train_attitude(plane, steps=None, seed=0, ckpt_dir, curves_dir)}`; `falcons.train.sac.train(plane, task, steps, seed, ckpt_dir, curves_dir, **hp)`; `falcons.train.td3.train(...)`; `falcons.train.curves.log_curve(name, step, curves_dir, **metrics)`; CLI subcommands as in the spec; `falcons.benchmark.eval_one(algo, task, plane, seed, maneuver, n, plot, ckpt_dir, results_dir) -> dict`.

- [ ] **Step 1: Port the trainers**

```bash
cd $NEW/src/falcons/train && for f in ppo sac td3 tasks curves; do cp $OLD/train/$f.py $f.py; done
sed -i -e 's/from envs\.altitude import WarpAltitudeEnv/from falcons.envs.altitude import AltitudeEnv/' -e 's/WarpAltitudeEnv/AltitudeEnv/g' \
       -e 's/from envs\.attitude import WarpAttitudeEnv/from falcons.envs.attitude import AttitudeEnv/' -e 's/WarpAttitudeEnv/AttitudeEnv/g' \
       -e 's/from train\.configs/from falcons.envs.configs/g' -e 's/from train\./from falcons.train./g' -e 's/import train\./import falcons.train./g' \
       -e 's/from backends\.torch\.altitude import TorchAltitudeEnv/from falcons.sim.torch.altitude import AltitudeEnv as TorchAltitudeEnv/' *.py
```

By hand in `tasks.py`: keep `evaluate_altitude`, `_make_altitude_env` (warp only — drop the torch branch), `train_altitude`, `evaluate_attitude`, `train_attitude`; delete everything for bank_heading/waypoint/waypoint_seq. In `train_altitude`, keep only the `mode == "dynamic"` path: delete every `dr/deploy/transfer/warmstart/cat/scocat/lag/floor/flare/va_h` variable, branch and `load_state_dict` warm-start; `tag` logic collapses to `seed`; the save line becomes `torch.save(agent.state_dict(), ckpt_path("ppo", "altitude", plane, seed, ckpt_dir))`; `log_curve(...)` name becomes `f"ppo_altitude_{plane}_s{seed}"`. Same for `train_attitude` (`ckpt_path("ppo", "attitude", …)`, curve `f"ppo_attitude_{plane}_s{seed}"`). `evaluate_altitude` loses every flag except `steps, settle`. In `sac.py`/`td3.py`: `train_sac`/`train_td3` → `train`, the save becomes `ckpt_path(algo, task, plane, seed, ckpt_dir)`, `curve_tag` → `f"{algo}_{task}_{plane}_s{seed}"`, delete `main()`. `curves.py`: `DIR` → `curves_dir` parameter defaulting to `RESULTS_DIR / "curves"`.

- [ ] **Step 2: `eval_one` and the CLI**

Add to `src/falcons/benchmark/__init__.py`:

```python
def eval_one(algo, task, plane, seed=0, maneuver="circle", n=1, plot=False, ckpt_dir=None, results_dir=None):
    """One cell of the benchmark, printed. `benchmark altitude/maneuvers` compose this same code."""
    from falcons.paths import CKPT_DIR, RESULTS_DIR
    ckpt_dir, results_dir = ckpt_dir or CKPT_DIR, results_dir or RESULTS_DIR
    if task == "altitude":
        from falcons.benchmark.altitude import run_learned, run_classical, aggregate
        rows = run_learned(plane, algo, seed, ckpt_dir) if algo in ("ppo", "sac", "td3") else run_classical(plane, algo, n, ckpt_dir)
        out = aggregate(rows)
        if plot and algo in ("ppo", "sac", "td3"):
            from falcons.benchmark.figures import altitude_viz; altitude_viz(algo, plane, ckpt_dir, results_dir)
    else:
        from falcons.benchmark.maneuvers import controllers, cell
        from falcons.envs.configs import attitude_env_cfg
        import numpy as np
        tol = float(np.degrees(attitude_env_cfg(plane).get("sig_phi", 0.15)))
        out = cell(plane, maneuver, algo.upper(), ckpt_dir, results_dir / "traces", refresh=False, mppi_draws=n, tol_deg=tol)
        if plot:
            from falcons.benchmark.figures import maneuver_facet; maneuver_facet(plane, ckpt_dir, results_dir)
    for k, v in out.items():
        if isinstance(v, (int, float)): print(f"{k:14s} {v:.4g}")
    return out
```

Rewrite `cli.py`:

```python
"""`falcons` command line: argparse glue only. Every subcommand calls one importable function."""
import argparse
import sys
from pathlib import Path

from falcons.paths import CKPT_DIR, RESULTS_DIR

ALGOS, TASKS = ("ppo", "sac", "td3"), ("altitude", "attitude")
PLANES = ("Airship_V7", "Airship_A0S", "Volantex_Ranger", "Navion", "Cirrus_SR22")
PROTOCOLS = ("altitude", "maneuvers", "ground-effect", "robustness", "throughput")


def _dirs(p):
    p.add_argument("--checkpoints", type=Path, default=CKPT_DIR)
    p.add_argument("--results", type=Path, default=RESULTS_DIR)


def cmd_check(a):
    import torch, warp as wp
    from falcons.controllers.policies import load_policy, ALGOS as AL, TASKS as TK
    from falcons.sim.cpu.aircraft import Aircraft as CpuAircraft
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} | warp {wp.__version__}")
    n = 0
    for algo in AL:
        for task in TK:
            for plane in ("Airship_V7", "Volantex_Ranger"):
                for s in (0, 1, 2):
                    load_policy(algo, task, plane, s, ckpt_dir=a.checkpoints); n += 1
    CpuAircraft("Airship_V7").reset()
    print(f"{n} checkpoints load; cpu plant steps. OK")


def cmd_train(a):
    from falcons.train import tasks, sac, td3
    curves = a.results / "curves"
    if a.algo == "ppo":
        (tasks.train_altitude if a.task == "altitude" else tasks.train_attitude)(a.plane, a.steps, a.seed, a.checkpoints, curves)
    else:
        (sac if a.algo == "sac" else td3).train(a.plane, a.task, a.steps, a.seed, a.checkpoints, curves)


def cmd_eval(a):
    from falcons.benchmark import eval_one
    eval_one(a.algo, a.task, a.plane, a.seed, a.maneuver, a.n, a.plot, a.checkpoints, a.results)


def cmd_benchmark(a):
    from falcons.benchmark import altitude, maneuvers, ground_effect, robustness, throughput
    planes = a.planes or ["Airship_V7", "Volantex_Ranger"]
    if a.protocol == "altitude":      altitude.run(planes, a.checkpoints, a.results, mppi_draws=a.mppi_draws)
    elif a.protocol == "maneuvers":   maneuvers.run(planes, a.checkpoints, a.results, a.refresh, a.mppi_draws, a.mppi_seed)
    elif a.protocol == "ground-effect":
        ground_effect.run_trim(a.planes or list(PLANES), a.results); ground_effect.run_energy(a.planes or list(PLANES), a.checkpoints, a.results)
    elif a.protocol == "robustness":  robustness.run(planes, a.checkpoints, a.results)
    else:                             throughput.run(planes[0], a.checkpoints, a.results)


def cmd_tables(a):
    from falcons.benchmark.tables import emit
    print(emit(a.results))


def cmd_figures(a):
    from falcons.benchmark.figures import render
    print(*render(a.names or None, a.checkpoints, a.results), sep="\n")


def cmd_reproduce(a):
    from falcons.benchmark.reproduce import run
    return run(a.checkpoints, a.results, a.out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="falcons")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check"); _dirs(s)
    s = sub.add_parser("train"); _dirs(s)
    s.add_argument("--algo", choices=ALGOS, required=True); s.add_argument("--task", choices=TASKS, required=True)
    s.add_argument("--plane", choices=PLANES, required=True); s.add_argument("--seed", type=int, default=0); s.add_argument("--steps", type=int)
    s = sub.add_parser("eval"); _dirs(s)
    s.add_argument("--algo", choices=ALGOS + ("lqr", "mppi"), required=True); s.add_argument("--task", choices=TASKS, required=True)
    s.add_argument("--plane", choices=PLANES, required=True); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--maneuver", default="circle", choices=("circle", "figure8", "helix", "sturn"))
    s.add_argument("--n", type=int, default=1, help="repetitions (MPPI draws)"); s.add_argument("--plot", action="store_true")
    s = sub.add_parser("benchmark"); _dirs(s)
    s.add_argument("protocol", choices=PROTOCOLS); s.add_argument("--planes", nargs="*", choices=PLANES)
    s.add_argument("--refresh", action="store_true"); s.add_argument("--mppi-draws", type=int, default=5); s.add_argument("--mppi-seed", type=int, default=0)
    s = sub.add_parser("tables"); _dirs(s)
    s = sub.add_parser("figures"); _dirs(s); s.add_argument("names", nargs="*")
    s = sub.add_parser("reproduce"); _dirs(s); s.add_argument("--out", type=Path, default=RESULTS_DIR.parent / "reproduce")
    a = p.parse_args(argv)
    return {"check": cmd_check, "train": cmd_train, "eval": cmd_eval, "benchmark": cmd_benchmark,
            "tables": cmd_tables, "figures": cmd_figures, "reproduce": cmd_reproduce}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
```

`altitude.run` defaults `mppi_draws=3` (the paper's altitude sweep count); the CLI's `--mppi-draws` default 5 is the maneuver count — pass `3` explicitly for `benchmark altitude` in `cmd_benchmark` (`mppi_draws=3 if a.mppi_draws == 5 else a.mppi_draws`), and say so in `--help`.

- [ ] **Step 3: `scripts/train_seeds.sh`**

```bash
cat > $NEW/scripts/train_seeds.sh <<'EOF'
#!/usr/bin/env bash
# Tier 2: retrain the 36 paper checkpoints (3 algos x 2 tasks x 2 planes x 3 seeds), sequentially.
# Idempotent: a checkpoint that already exists is skipped. Hours per run on an RTX 4090; see README.
set -euo pipefail
cd "$(dirname "$0")/.."
for algo in ppo sac td3; do for task in altitude attitude; do for plane in Airship_V7 Volantex_Ranger; do for seed in 0 1 2; do
  ck="checkpoints/${algo}_${task}_${plane}_s${seed}.pt"
  if [ -f "$ck" ]; then echo "SKIP $ck"; continue; fi
  echo "=== $algo $task $plane s$seed  $(date +%H:%M:%S)"
  .venv/bin/falcons train --algo "$algo" --task "$task" --plane "$plane" --seed "$seed"
done; done; done; done
EOF
chmod +x $NEW/scripts/train_seeds.sh
```

- [ ] **Step 4: Tests and smoke runs**

```python
# tests/test_train.py
import pytest, torch


@pytest.mark.cuda
@pytest.mark.parametrize("algo,task", [(a, t) for a in ("ppo", "sac", "td3") for t in ("altitude", "attitude")])
def test_thousand_step_smoke(algo, task, tmp_path):
    from falcons.controllers.policies import ckpt_path
    from falcons.train import tasks, sac, td3
    if algo == "ppo":
        (tasks.train_altitude if task == "altitude" else tasks.train_attitude)("Volantex_Ranger", 20_000, 0, tmp_path, tmp_path)
    else:
        (sac if algo == "sac" else td3).train("Volantex_Ranger", task, 20_000, 0, tmp_path, tmp_path)
    assert ckpt_path(algo, task, "Volantex_Ranger", 0, tmp_path).exists()


@pytest.mark.cuda
def test_eval_cell_equals_benchmark_cell():
    import csv
    from falcons.benchmark import eval_one
    out = eval_one("ppo", "attitude", "Volantex_Ranger", 0, "helix")      # replays the cached cell
    gold = next(r for r in csv.DictReader(open("tests/golden/maneuvers.csv"))
                if (r["aircraft"], r["maneuver"], r["method"]) == ("Volantex_Ranger", "helix", "PPO"))
    assert out["phi"]["rmse"] == pytest.approx(float(gold["phi_rmse_deg"]), abs=0.005)
```

Remove the `skip` marker placed on `test_evaluate_attitude_returns_sane_tuple` in Task 5. Run: `.venv/bin/pytest tests/test_train.py tests/test_envs.py -q` → all pass (the 6 smoke runs take a few minutes). Then `.venv/bin/falcons check` and `.venv/bin/falcons eval --algo lqr --task attitude --plane Airship_V7 --maneuver circle` print without error.

- [ ] **Step 5: Commit** — `git commit -q -m "train: ppo/sac/td3 on the two paper tasks; cli: train, eval, benchmark, tables, figures"`

---

### Task 10: `reproduce`, `environment.json`, `test_reproduce.py`

**Files:**
- Create: `src/falcons/benchmark/reproduce.py`, `tests/test_reproduce.py`

**Interfaces:**
- Produces: `reproduce.environment() -> dict` (gpu, driver, cuda, torch, warp, python, commit, date); `reproduce.run(ckpt_dir, results_dir, out_dir) -> int` (0 = every table within tolerance; writes `out_dir/{*.csv,tables.tex,figures/,environment.json,REPORT.md}`).

- [ ] **Step 1: Write it**

```python
# src/falcons/benchmark/reproduce.py
"""The reproducibility claim, executable: regenerate every table and figure from the checkpoints
into a fresh directory, diff each CSV against the shipped results/ under TOLERANCE, write a report."""
import json, platform, subprocess, sys
from datetime import date
from pathlib import Path

from falcons.benchmark import TOLERANCE
from falcons.benchmark.diff import compare_csv


def environment():
    import torch, warp
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        drv = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    except FileNotFoundError:
        commit = None
    return {"date": date.today().isoformat(), "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "warp": warp.__version__, "gpu": gpu, "driver": drv, "commit": commit}


def run(ckpt_dir, results_dir, out_dir):
    from falcons.benchmark import altitude, maneuvers, ground_effect, robustness
    from falcons.benchmark.tables import emit
    from falcons.benchmark.figures import render
    from falcons.aircraft.config import PLANES
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    planes = ["Airship_V7", "Volantex_Ranger"]
    altitude.run(planes, ckpt_dir, out, mppi_draws=3)
    ground_effect.run_trim(PLANES, out); ground_effect.run_energy(PLANES, ckpt_dir, out)
    maneuvers.run(planes, ckpt_dir, out, refresh=True, mppi_draws=5, mppi_seed=0)
    robustness.run(planes, ckpt_dir, out)
    emit(out); render(None, ckpt_dir, out)
    env = environment(); json.dump(env, open(out / "environment.json", "w"), indent=1)
    report, fails = [f"# reproduce {env['date']} on {env['gpu']}", ""], 0
    for name, rule in TOLERANCE.items():
        if name == "throughput.csv":
            continue
        gold = Path(results_dir) / name
        if not gold.exists():
            report.append(f"- {name}: no shipped file to compare"); continue
        bad = compare_csv(out / name, gold, rule)
        report.append(f"- {name}: {'PASS' if not bad else 'FAIL'}" + "".join(f"\n    - {b}" for b in bad[:20]))
        fails += bool(bad)
    (out / "REPORT.md").write_text("\n".join(report) + "\n"); print("\n".join(report))
    return 1 if fails else 0
```

- [ ] **Step 2: Tests**

```python
# tests/test_reproduce.py
import shutil, pytest
from falcons.benchmark.reproduce import environment


def test_environment_records_versions():
    e = environment()
    assert e["torch"].startswith("2.7.0") and e["warp"] == "1.8.1" and e["python"].startswith("3.10")


@pytest.mark.cuda
def test_maneuver_table_on_fixture_is_exact(tmp_path):
    """The fast reproducibility check: replay the shipped-equivalent cache through the whole
    maneuver pipeline and demand a byte-for-byte table."""
    from falcons.benchmark.maneuvers import run
    from falcons.benchmark.diff import compare_csv
    shutil.copytree("/home/matteo/Projects/falcon-s-goldens/traces", tmp_path / "traces")
    out = run(["Airship_V7", "Volantex_Ranger"], results_dir=tmp_path)
    assert compare_csv(out, "tests/golden/maneuvers.csv", {"key": ["aircraft", "maneuver", "method"], "abs": 0.0}) == []


@pytest.mark.slow
@pytest.mark.cuda
def test_full_reproduce_passes(tmp_path):
    from falcons.benchmark.reproduce import run
    from falcons.paths import CKPT_DIR, RESULTS_DIR
    assert run(CKPT_DIR, RESULTS_DIR, tmp_path) == 0
```

Run the fast tier: `.venv/bin/pytest -q -m "not slow"` → everything green, under 30 s excluding the Task 9 smoke tests (mark those `@pytest.mark.slow` too if they exceed the budget).

- [ ] **Step 3: Commit** — `git commit -q -m "benchmark: reproduce with environment fingerprint and report"`

---

### Task 11: Final re-fly, shipped results, README, PROVENANCE, LICENSE

**Files:**
- Create: `results/{altitude,ge_trim,ge_energy,maneuvers,robustness,throughput}.csv`, `results/{mppi_variance.json,parity.json,tables.tex,environment.json}`, `results/figures/*` (11), `results/curves/*.csv` (36), `README.md`, `PROVENANCE.md`, `LICENSE`

- [ ] **Step 1: Full re-fly with no cache**

```bash
cd $NEW && rm -rf results/traces results/*.csv results/figures && mkdir -p results
.venv/bin/falcons benchmark altitude --mppi-draws 3
.venv/bin/falcons benchmark ground-effect
.venv/bin/falcons benchmark maneuvers --refresh
.venv/bin/falcons benchmark robustness
.venv/bin/falcons benchmark throughput --planes Airship_V7
.venv/bin/falcons tables && .venv/bin/falcons figures
.venv/bin/python -c "import json; from falcons.benchmark.reproduce import environment; json.dump(environment(), open('results/environment.json','w'), indent=1)"
for n in altitude ge_trim ge_energy maneuvers robustness; do
  .venv/bin/python -c "from falcons.benchmark import TOLERANCE; from falcons.benchmark.diff import compare_csv; b=compare_csv('results/$n.csv','tests/golden/$n.csv',TOLERANCE['$n.csv']); print('$n', 'PASS' if not b else b)"; done
```

Every line must print `PASS`. Learned/LQR rows are bit-identical to the archive; MPPI and wind rows differ within σ, as designed. Then copy the training curves:

```bash
mkdir -p results/curves && cd $OLD/training_logs/curves && for f in ppo_altitude_*_dynamic*.csv sac_altitude_*_dynamic*.csv td3_altitude_*_dynamic*.csv ppo_attitude_*_dynamic*.csv sac_attitude_*_dynamic*.csv td3_attitude_*_dynamic*.csv; do
  case $f in *Airship_V7*|*Volantex_Ranger*) n=$(echo $f | sed -E 's/_dynamic\.csv/_s0.csv/; s/_dynamic_s([12])\.csv/_s\1.csv/'); cp $f $NEW/results/curves/$n;; esac; done
ls $NEW/results/curves | wc -l      # 36
```

- [ ] **Step 2: `reproduce` against its own output**

```bash
cd $NEW && .venv/bin/falcons reproduce --out /tmp/falcon-s-reproduce && echo "REPRODUCE OK"
```

Exit 0 and every table `PASS` in `/tmp/falcon-s-reproduce/REPORT.md`. This is the run the README quotes.

- [ ] **Step 3: `PROVENANCE.md`**

```bash
cat > $NEW/PROVENANCE.md <<EOF
# Provenance

Fresh history. Code, data and checkpoints were curated from the development repository:

- source: $(cat /home/matteo/Projects/falcon-s-goldens/golden/SOURCE.txt | tr '\n' ' ')
- physics fixes that the shipped results depend on (all before the source commit): world-frame quaternion
  integration of body rates in the GPU backends; fleet-wide lateral moment sign correction; LQR gains
  re-derived on the corrected dynamics; V7 motor constant corrected to reach its declared trim.

## Checkpoint rename map

Old name (development repo, gitignored there) → shipped name.

EOF
awk -F'\t' '{printf "- `%s` → `%s`\n", $1, $2}' /home/matteo/Projects/falcon-s-goldens/golden/CHECKPOINT_MAP.tsv >> $NEW/PROVENANCE.md
cat >> $NEW/PROVENANCE.md <<'EOF'

## What was left behind

The committed virtual environment; 150 checkpoints from experiments not in the paper; the waypoint,
bank-heading, flare and constrained-RL tasks; the ground-effect-during-training study; the
X-Plane/Matlab plot overlays; 28 experiment scripts. All remain in the source repository at the commit above.
EOF
```

- [ ] **Step 4: `README.md`**

Write it with these sections, in this order, no others: title + one-paragraph abstract-level description; **Requirements** (Python 3.10, NVIDIA GPU + CUDA 12.6 driver, ~2 GB disk); **Install** (`git clone … && bash scripts/setup_venv.sh`); **Reproduce the paper** (`falcons reproduce`, what it regenerates, the runtime on an RTX 4090 measured in Step 2, and the two sentences: learned and LQR rows are bit-identical; MPPI rows and turbulence cells are draws and land within the shipped spread); **Commands** (the seven, one line each with the example from the spec); **Layout** (the tree from `docs/design.md`); **Tier 2: retraining** (`scripts/train_seeds.sh`, hours per run, that seed means will not reproduce bit-for-bit across GPUs); **Old → new** (table: `python -m train --task altitude --aircraft X --seed N` → `falcons train --algo ppo --task altitude --plane X --seed N`; `python -m train.sac …` → `falcons train --algo sac …`; `python -m eval --task altitude viz X --algo A` → `falcons eval --algo A --task altitude --plane X --plot`; `python -m trl_paper.altitude.protocol` → `falcons benchmark altitude`; `…maneuvers.plot_track` → `falcons benchmark maneuvers` + `falcons figures maneuver_facet_Airship_V7`; `…altitude.emit_tex` → `falcons tables`; `MPPI_SEED=k` → `--mppi-seed k`); **Citation** (bibtex stub with the paper title, authors left as in the manuscript). Every command in the README is run once before commit and must succeed.

- [ ] **Step 5: LICENSE**

The source repository has no license file. **Ask the user which license to apply before writing this file**; do not commit Task 11 without it.

- [ ] **Step 6: Commit and tag**

```bash
cd $NEW && .venv/bin/pytest -q -m "not slow" && git add -A && git commit -q -m "results: shipped tables, figures, curves; README, PROVENANCE, LICENSE" && git tag v1.0.0
du -sh .git checkpoints results     # sanity: repo ≈ 60–70 MB
```

---

### Task 12: Dockerfile (separate pass, after Task 11 is green)

**Files:**
- Create: `Dockerfile`, `.dockerignore`, README "Docker" subsection

- [ ] **Step 1: Write it**

```dockerfile
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /falcon-s
COPY requirements.lock pyproject.toml ./
RUN python3.10 -m venv .venv && .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.lock
COPY src src
COPY checkpoints checkpoints
COPY results results
COPY scripts scripts
COPY tests tests
RUN .venv/bin/pip install -e ".[test]" --no-deps
ENV PATH=/falcon-s/.venv/bin:$PATH
CMD ["falcons", "check"]
```

```bash
printf '.venv/\n.git/\nresults/traces/\n__pycache__/\n' > $NEW/.dockerignore
cd $NEW && docker build -t falcons:1.0.0 . && docker run --rm --gpus all falcons:1.0.0        # prints the check line
docker run --rm --gpus all falcons:1.0.0 pytest -q -m "not slow"
```

- [ ] **Step 2: Commit** — `git commit -q -m "docker: cuda 12.6 image with the locked environment"`

---

## Self-review

**Spec coverage.** Layout → Tasks 1–9 (one deviation: `controllers/lqr/` stays a three-file package rather than one `lqr.py`; 710 + 122 + 231 lines merged would be the largest file in the repo, against the spec's own "focused files" rule — noted in `docs/design.md` at Task 11). CLI seven commands → Task 9 (+ `reproduce` in 10). Regression gate → Task 0 goldens, per-task gates, Task 11 re-fly. Tests table → `test_parity` (4), `test_checkpoints` (6), `test_sim_*` (3, 4), `test_envs` (5), `test_reproduce` (10); the spec's `test_sim.py` is split into `test_sim_cpu.py`/`test_sim_warp.py`. Ships vs regenerates → Task 11 + `.gitignore` in Task 1. Environment → Task 1; Docker → Task 12. Migration order → Tasks 0–12 one-to-one. Two spec corrections carried into the plan: `eval/controllers/{lqr,mppi}.py` kept as `controllers/altitude.py` (Task 6) with the child process (7a); `maneuver_render` rewritten without ffmpeg (8).

**Placeholders.** None: every file has its content or an exact source path plus the exact edit; `LICENSE` is an explicit user decision, not a TBD.

**Type consistency.** `load_policy(algo, task, plane, seed, device, env, stream, ckpt_dir)` and `.act` used identically in Tasks 6, 7a–e, 8, 9; `seeds(algo, task, plane, ckpt_dir)` in 6/7a/7c; `cell(plane, maneuver, method, ckpt_dir, cache_dir, refresh, mppi_draws, mppi_seed, tol_deg)` in 7c/8/9; `compare_csv(new, gold, rule)` in 7a–d/10/11; `run(planes, ckpt_dir, results_dir, …)` shape shared by every protocol; `ckpt_path(algo, task, plane, seed, ckpt_dir)` in 6/9.
