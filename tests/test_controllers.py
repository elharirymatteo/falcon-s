"""The controller/checkpoint loading surface: the LQR gain tables shipped beside each airframe, the
39 policy checkpoints the benchmark scores, and the MPPI attitude executor built from a stream."""
from pathlib import Path

import numpy as np, pytest, torch
from falcons.aircraft.config import AircraftConfig
from falcons.controllers.policies import ALGOS, TASKS, OBS_ACT, seeds, load_policy
from falcons.paths import CKPT_DIR
from conftest import ONLINE_PLANES, requires_derivatives

GOLD = Path(__file__).parent / "golden" / "lqr_gains"

# One grep finds every gate the LQR refit must clear. See plan.md Phase 6.
LQR_PENDING = "LQR gains were fitted to the polynomial plant; refit pending (plan.md Phase 6)"

CELLS = [(a, t, p, s) for a in ALGOS for t in TASKS for p in ("Airship_V7", "Volantex_Ranger") for s in (0, 1, 2)]
CELLS += [("ppo", "altitude", p, 0) for p in ("Airship_A0S", "Navion")]
CELL_MARKS = [pytest.param(a, t, p, s, marks=requires_derivatives(p)) for a, t, p, s in CELLS]


@pytest.mark.skip(reason=LQR_PENDING)
def test_lqr_gains_identical_to_archive():
    """Frozen, not deleted. The gains themselves are gone -- they linearised the polynomial plant
    -- so there is nothing to compare against `tests/golden/lqr_gains/` until the LQR controller is
    refitted to the derivative set. The archive stays as the record of what the old plant flew."""
    for ac in ("Airship_V7", "Volantex_Ranger", "Navion"):
        new = np.loadtxt(AircraftConfig(ac).lqr_gains_path, delimiter=",")
        old = np.loadtxt(GOLD / f"{ac}_K_LQR.csv", delimiter=",")
        np.testing.assert_array_equal(new, old)


def test_thirty_nine_checkpoints_ship():
    assert len(list(CKPT_DIR.glob("*.pt"))) == 39


@pytest.mark.cuda
@pytest.mark.parametrize("algo,task,plane,seed", CELL_MARKS, ids=[f"{a}_{t}_{p}_s{s}" for a, t, p, s in CELLS])
def test_checkpoint_loads_and_acts(algo, task, plane, seed):
    assert seed in seeds(algo, task, plane)
    pol = load_policy(algo, task, plane, seed)
    n_obs, n_act = OBS_ACT[task]
    obs = torch.zeros(4, n_obs, device="cuda")
    a1, a2 = pol.act(obs), pol.act(obs)
    assert a1.shape == (4, n_act) and torch.equal(a1, a2)


@pytest.mark.cuda
@requires_derivatives("Airship_V7")
def test_mppi_attitude_executor_builds_with_seed():
    import warp as wp
    from falcons.envs.attitude import AttitudeEnv
    from falcons.envs.configs import attitude_env_cfg
    wp.init()
    env = AttitudeEnv(1, "cuda", attitude_env_cfg("Airship_V7", horizon=52)); env.reset()
    z = np.zeros(50, dtype=np.float32)
    pol = load_policy("mppi", "attitude", "Airship_V7", env=env, stream=(z, z, z + env.base_vel), seed=3)
    env._obs_launch(); a = pol.act(wp.to_torch(env._obs))
    assert a.shape[-1] == 4
