"""The training loops, exercised end to end at the smallest budget that still executes one
update, and the guarantee that `falcons eval` prints the benchmark's own cell rather than a
second implementation of it."""
import csv
import os
import shutil
from pathlib import Path

import pytest

from falcons.train.ppo import N_ENVS as PPO_ENVS, ROLLOUT
from falcons.train.sac import N_ENVS as OFF_ENVS, WARMUP

GOLD = Path(__file__).parent / "golden"
# The 39 MB rollout cache lives outside the repo; point FALCONS_TRACE_FIXTURE at it to run the
# replay gate, which needs no rollout to be flown.
FIX = Path(os.environ.get("FALCONS_TRACE_FIXTURE", "/home/matteo/Projects/falcon-s-goldens/traces"))

# PPO: `train_loop` does `iters = total_steps // (ROLLOUT * N_ENVS)`, so anything under one full
# batch runs ZERO iterations -- no update, no curriculum hook, no curve. One batch is the
# smallest budget that runs one.                                  32 * 4096 = 131_072
PPO_STEPS = ROLLOUT * PPO_ENVS
# SAC/TD3: `iters = steps // N_ENVS` and an iteration only updates once `gstep = it * N_ENVS`
# clears WARMUP, so the first updating iteration is it = ceil(WARMUP / N_ENVS) = 79 and `iters`
# has to reach 79 + 1 = 80.                                          80 * 256 = 20_480
OFF_STEPS = (-(-WARMUP // OFF_ENVS) + 1) * OFF_ENVS
STEPS = {"ppo": PPO_STEPS, "sac": OFF_STEPS, "td3": OFF_STEPS}


@pytest.mark.cuda
@pytest.mark.parametrize("algo,task", [(a, t) for a in ("ppo", "sac", "td3") for t in ("altitude", "attitude")])
def test_one_update_smoke(algo, task, tmp_path):
    """Every trainer builds its env, runs at least one gradient update, evaluates, and writes both
    artefacts under the one naming scheme the rest of the package reads back: the checkpoint at
    `ckpt_path(...)` and the learning curve at `<algo>_<task>_<plane>_s<seed>.csv`."""
    from falcons.controllers.policies import ckpt_path
    from falcons.train import tasks, sac, td3
    steps = STEPS[algo]
    if algo == "ppo":
        (tasks.train_altitude if task == "altitude" else tasks.train_attitude)(
            "Volantex_Ranger", steps, 0, tmp_path, tmp_path)
    else:
        (sac if algo == "sac" else td3).train("Volantex_Ranger", task, steps, 0, tmp_path, tmp_path)
    assert ckpt_path(algo, task, "Volantex_Ranger", 0, tmp_path).exists()
    curve = tmp_path / f"{algo}_{task}_Volantex_Ranger_s0.csv"
    assert curve.exists(), "no learning curve under the ruling-7 name -> log_curve never ran"
    with open(curve) as f:
        rows = list(csv.reader(f))
    assert len(rows) >= 2 and rows[0][0] == "env_steps", f"curve has no data row: {rows}"


@pytest.mark.cuda
def test_eval_cell_equals_benchmark_cell(tmp_path):
    """`falcons eval` on one (plane, maneuver, method) has to reproduce the shipped table row for
    that cell exactly: the command is a view on the protocol, not a parallel one."""
    if not FIX.exists():
        pytest.skip("trace fixture not present")
    from falcons.benchmark import eval_one
    shutil.copytree(FIX, tmp_path / "traces")
    out = eval_one("ppo", "attitude", "Volantex_Ranger", 0, "helix", results_dir=tmp_path)
    with open(GOLD / "maneuvers.csv") as f:
        gold = next(r for r in csv.DictReader(f)
                    if (r["aircraft"], r["maneuver"], r["method"])
                    == ("Volantex_Ranger", "helix", "PPO"))
    assert out["phi"]["rmse"] == pytest.approx(float(gold["phi_rmse_deg"]), abs=0.005)
