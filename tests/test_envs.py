"""Full env contract: runs a full horizon, auto-resets done envs, never returns NaN,
and returns the (obs, reward, done, info) shapes the PPO loop needs."""
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from conftest import GOLDENS_PENDING, requires_derivatives

# Every test here builds Airship_V7, whose OpenVSP data has not been extracted yet.
pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="warp env needs CUDA"),
              requires_derivatives("Airship_V7")]


def test_env_runs_and_autoresets_clean():
    import warp as wp
    from falcons.envs.attitude import AttitudeEnv
    from falcons.envs.configs import attitude_env_cfg
    wp.init()
    H = 250
    env = AttitudeEnv(128, "cuda", attitude_env_cfg("Airship_V7", horizon=H))
    obs = env.reset()
    assert obs.shape == (128, 15)
    a = torch.zeros((128, 4), device="cuda")
    seen_done = False
    for t in range(H + 10):
        obs, r, d, info = env.step(a)
        assert obs.shape == (128, 15) and r.shape == (128,) and d.shape == (128,)
        assert not torch.isnan(obs).any() and not torch.isnan(r).any()
        assert set(info) >= {"terminal_obs", "truncated", "reason"}
        seen_done |= bool(d.any())
    assert seen_done, "no env ever hit truncation over a full horizon"


def test_step_counter_resets_on_done():
    import warp as wp
    from falcons.envs.attitude import AttitudeEnv
    from falcons.envs.configs import attitude_env_cfg
    wp.init()
    H = 60
    env = AttitudeEnv(64, "cuda", attitude_env_cfg("Airship_V7", horizon=H))
    env.reset()
    a = torch.zeros((64, 4), device="cuda")
    for _ in range(H + 2):
        env.step(a)
    steps = wp.to_torch(env._step).cpu().numpy()
    assert steps.max() <= H, "step counter not reset after truncation"


def test_evaluate_attitude_returns_sane_tuple():
    import warp as wp
    import torch
    from falcons.train.tasks import evaluate_attitude
    from falcons.train.ppo import Agent
    from falcons.envs.attitude import AttitudeEnv
    from falcons.envs.configs import attitude_env_cfg
    wp.init()
    env = AttitudeEnv(64, "cuda", attitude_env_cfg("Airship_V7"))
    agent = Agent(env.num_obs, env.num_act).cuda()   # random init
    out = evaluate_attitude(agent, "Airship_V7", steps=300, settle=50)
    assert len(out) == 5
    rob, phi_rmse, hdot_rmse, va_rmse, settle = out
    assert 0.0 <= rob <= 1.0
    assert phi_rmse >= 0.0 and hdot_rmse >= 0.0 and va_rmse >= 0.0


"""compute_att_obs golden check: attitude/speed errors vs a python re-implementation
fed the SAME warp state."""


def test_att_obs_matches_reference():
    import warp as wp
    from falcons.envs.attitude import AttitudeEnv
    from falcons.envs.configs import attitude_env_cfg
    wp.init()
    env = AttitudeEnv(16, "cuda", attitude_env_cfg("Airship_V7"))
    env.reset()
    # force known, varied targets so the error channels are non-trivial
    n = env.n
    phi = np.linspace(-0.5, 0.5, n).astype(np.float32)
    hdot = np.linspace(-3.0, 3.0, n).astype(np.float32)
    va = np.linspace(0.9, 1.1, n).astype(np.float32) * env.base_vel
    # write through the schedule buffers (all 3 segments) since step() re-derives
    # _phi_tgt/_hdot_tgt/_va_tgt from _phi3/_hdot3/_va3 every call (Task 2).
    wp.copy(env._phi3, wp.array(np.tile(phi[:, None], (1, 3)), device="cuda"))
    wp.copy(env._hdot3, wp.array(np.tile(hdot[:, None], (1, 3)), device="cuda"))
    wp.copy(env._va3, wp.array(np.tile(va[:, None], (1, 3)), device="cuda"))
    # step once with zero action so the sim advances and obs recompute
    env.step(torch.zeros((n, 4), device="cuda"))
    obs = wp.to_torch(env._obs).cpu().numpy()

    s = env.model._state
    q = wp.to_torch(s["orientation"]).cpu().numpy()
    Va = wp.to_torch(env.model._Va).cpu().numpy()
    w = wp.to_torch(s["angular_vel"]).cpu().numpy()
    vb = wp.to_torch(s["linear_vel"]).cpu().numpy()
    alpha = wp.to_torch(env.model._alpha).cpu().numpy()
    beta = wp.to_torch(env.model._beta).cpu().numpy()
    prev = wp.to_torch(env._prev).cpu().numpy()
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    # world vertical speed = (quat_rotate(q, v_body))[2], q = (x,y,z,w)
    t = 2.0 * np.cross(q[:, :3], vb)
    vw = vb + qw[:, None] * t + np.cross(q[:, :3], t)
    vz = vw[:, 2]

    def clip5(x):
        return np.clip(x, -5, 5)

    exp = np.empty((n, 15), dtype=np.float32)
    exp[:, 0] = clip5((roll - phi) / (np.pi / 4))
    exp[:, 1] = clip5(((-vz) - hdot) / env.vz_scale)   # climb_rate = -vz
    exp[:, 2] = clip5((Va - va) / env.va_scale)
    exp[:, 3] = clip5(roll / (np.pi / 4))
    exp[:, 4] = clip5(pitch / (np.pi / 4))
    exp[:, 5] = clip5(w[:, 0] / 2.0)
    exp[:, 6] = clip5(w[:, 1] / 2.0)
    exp[:, 7] = clip5(w[:, 2] / 2.0)
    exp[:, 8] = clip5(alpha / 0.349)
    exp[:, 9] = clip5(beta / 0.349)
    exp[:, 10] = clip5(vz / env.vz_scale)
    exp[:, 11] = clip5(prev[:, 0])
    exp[:, 12] = clip5(prev[:, 1])
    exp[:, 13] = clip5(prev[:, 2])
    exp[:, 14] = clip5(prev[:, 3])
    assert obs.shape[1] == 15
    for i in range(15):
        assert np.allclose(obs[:, i], exp[:, i], atol=1e-4), f"channel {i} mismatch"


"""Reward peaks when attitude+speed are on target, and the energy floor penalizes
airspeed below the safety threshold."""


def _reward_for(env, roll, pitch, Va):
    """Drive the reward kernel with a synthesized attitude/speed via env internals."""
    import warp as wp
    n = env.n
    # build quaternion from (roll, pitch, yaw=0)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    q = np.zeros((n, 4), dtype=np.float32)
    q[:, 0] = sr * cp; q[:, 1] = cr * sp; q[:, 2] = -sr * sp; q[:, 3] = cr * cp
    wp.copy(env.model._state["orientation"], wp.array(q, dtype=wp.quatf, device="cuda"))
    wp.copy(env.model._Va, wp.array(np.full(n, Va, dtype=np.float32), device="cuda"))
    env._reward_launch()
    return wp.to_torch(env._rew).cpu().numpy().mean()


def test_reward_peaks_on_target_and_energy_floor_bites():
    import warp as wp
    from falcons.envs.attitude import AttitudeEnv
    from falcons.envs.configs import attitude_env_cfg
    wp.init()
    env = AttitudeEnv(32, "cuda", attitude_env_cfg("Airship_V7"))
    env.reset()
    env._phi_tgt.zero_(); env._hdot_tgt.zero_()
    wp.copy(env._va_tgt, wp.array(np.full(env.n, env.base_vel, dtype=np.float32), device="cuda"))
    env._prev.zero_(); env._act.zero_()

    # roll-only synthesized attitude (pitch=0) rotates body vel (V,0,0) about the x-axis, so
    # vz stays 0 regardless of roll -> climb_rate=0 matches hdot_tgt=0 in both cases below;
    # the hdot term does not confound the phi-error comparison.
    on_target = _reward_for(env, 0.0, 0.0, env.base_vel)
    off_attitude = _reward_for(env, 0.4, 0.0, env.base_vel)   # 23deg roll error
    assert on_target > off_attitude

    slow = _reward_for(env, 0.0, 0.0, 0.5 * env.base_vel)     # below va_safety
    ok_speed = _reward_for(env, 0.0, 0.0, env.base_vel)
    assert ok_speed > slow  # energy floor penalizes the slow case


def test_energy_floor_isolated():
    """Isolate the low-speed floor: hold the Va-tracking Gaussian maxed (va_tgt==Va, so
    va_err==0) in BOTH cases, varying only whether Va sits above vs below va_safety. The
    only reward difference is then the floor penalty, so the gap must equal the analytic
    floor value -- this fails if the `if deficit > 0.0` term is deleted or mis-signed."""
    import warp as wp
    from falcons.envs.attitude import AttitudeEnv
    from falcons.envs.configs import attitude_env_cfg
    wp.init()
    env = AttitudeEnv(32, "cuda", attitude_env_cfg("Airship_V7"))
    env.reset()
    env._phi_tgt.zero_(); env._hdot_tgt.zero_()
    env._prev.zero_(); env._act.zero_()

    # pick Va points straddling the ACTUAL configured va_safety threshold (absolute m/s),
    # so the test holds regardless of the va_safety / w_energy the config sets.
    vs = env.va_safety

    def _reward_at(Va):
        wp.copy(env._va_tgt, wp.array(np.full(env.n, Va, dtype=np.float32), device="cuda"))
        return _reward_for(env, 0.0, 0.0, Va)  # va_tgt == Va -> va_err == 0 (tracking maxed)

    above = _reward_at(vs + 1.0)   # Va > va_safety -> deficit < 0 -> floor inactive
    below = _reward_at(vs - 1.0)   # Va < va_safety -> deficit > 0 -> floor active
    assert above > below

    expected_gap = env.w_energy * (1.0 / vs) ** 2   # penalty at Va = va_safety - 1.0
    assert abs((above - below) - expected_gap) < 1e-3


"""Targets are piecewise-constant with two mid-episode switches at H/3 and 2H/3,
sampled within the configured ranges."""


def test_targets_switch_at_thirds_and_stay_in_range():
    import warp as wp
    from falcons.envs.attitude import AttitudeEnv
    from falcons.envs.configs import attitude_env_cfg
    wp.init()
    H = 300
    env = AttitudeEnv(64, "cuda", attitude_env_cfg("Airship_V7", horizon=H))
    env.reset()
    phi3 = wp.to_torch(env._phi3).cpu().numpy()
    # ranges: |phi| <= 45deg, |hdot| <= hdot_max (m/s), va in [0.85,1.15]*base_vel
    assert np.all(np.abs(phi3) <= np.radians(45) + 1e-4)
    hdot3 = wp.to_torch(env._hdot3).cpu().numpy()
    assert np.all(np.abs(hdot3) <= env.hdot_max + 1e-4)
    va3 = wp.to_torch(env._va3).cpu().numpy()
    assert np.all(va3 >= 0.85 * env.base_vel - 1e-3) and np.all(va3 <= 1.15 * env.base_vel + 1e-3)

    # active target equals segment 0 early, segment 1 past H/3, segment 2 past 2H/3
    def active_phi():
        return wp.to_torch(env._phi_tgt).cpu().numpy()
    a = torch.zeros((64, 4), device="cuda")
    env.step(a)  # step index becomes 1 -> still seg 0
    assert np.allclose(active_phi(), phi3[:, 0], atol=1e-5)
    for _ in range(H // 3):
        env.step(a)
    assert np.allclose(active_phi(), phi3[:, 1], atol=1e-5)
    for _ in range(H // 3):
        env.step(a)
    assert np.allclose(active_phi(), phi3[:, 2], atol=1e-5)


"""Target-magnitude curriculum: set_curriculum(frac) grows phi_max/hdot_max from a gentle
floor to the full range, mirroring bank_heading's dpsi curriculum."""


def test_full_range_at_construction():
    import warp as wp
    from falcons.envs.attitude import AttitudeEnv
    from falcons.envs.configs import attitude_env_cfg
    wp.init()
    env = AttitudeEnv(64, "cuda", attitude_env_cfg("Airship_V7"))
    assert env.phi_max_full == env.phi_max
    assert env.hdot_max_full == env.hdot_max


def test_curriculum_zero_shrinks_to_floor():
    import warp as wp
    from falcons.envs.attitude import AttitudeEnv
    from falcons.envs.configs import attitude_env_cfg
    wp.init()
    env = AttitudeEnv(64, "cuda", attitude_env_cfg("Airship_V7"))
    env.set_curriculum(0.0)
    assert abs(env.phi_max - env.phi_min) < 1e-6
    assert abs(env.hdot_max - env.hdot_min) < 1e-6


def test_curriculum_one_restores_full_range():
    import warp as wp
    from falcons.envs.attitude import AttitudeEnv
    from falcons.envs.configs import attitude_env_cfg
    wp.init()
    env = AttitudeEnv(64, "cuda", attitude_env_cfg("Airship_V7"))
    env.set_curriculum(0.0)
    env.set_curriculum(1.0)
    assert abs(env.phi_max - env.phi_max_full) < 1e-6
    assert abs(env.hdot_max - env.hdot_max_full) < 1e-6


def test_curriculum_zero_shrinks_sampled_targets():
    import warp as wp
    from falcons.envs.attitude import AttitudeEnv
    from falcons.envs.configs import attitude_env_cfg
    wp.init()
    env = AttitudeEnv(64, "cuda", attitude_env_cfg("Airship_V7"))
    env.set_curriculum(0.0)
    env.reset()
    phi3 = wp.to_torch(env._phi3).cpu().numpy()
    assert np.abs(phi3).max() <= env.phi_min + 1e-6
    hdot3 = wp.to_torch(env._hdot3).cpu().numpy()
    assert np.abs(hdot3).max() <= env.hdot_min + 1e-6


"""Heading-invariance test for the warp backend: altitude response with zero wind must be
identical for initial yaw 0 and 90 deg. Catches the quaternion-derivative frame swap in
falcons/sim/warp/aerodynamics.py."""


def _run(yaw_deg, steps=300):
    import warp as wp
    from falcons.envs.altitude import AltitudeEnv
    wp.init()
    n = 4
    env = AltitudeEnv(n, "cuda", {"aircraft": "Airship_V7", "target_mode": "fixed",
                                  "target_altitude": 60.0, "spawning_distance": 0.0})
    env.reset()
    half = math.radians(yaw_deg) / 2.0
    q = wp.quatf(0.0, 0.0, math.sin(half), math.cos(half))
    wp.copy(env.model._state["orientation"],
            wp.array([q] * n, dtype=wp.quatf, device="cuda"))
    act = torch.zeros(n, env.num_act, device="cuda")     # fixed identical controls
    alts = np.zeros(steps)
    for t in range(steps):
        env.step(act)
        alts[t] = -env.model._state["position"].numpy()[0, 2]
    return alts


def test_altitude_response_heading_invariant():
    a0 = _run(0.0)
    a90 = _run(90.0)
    assert np.max(np.abs(a0 - a90)) < 1e-2, \
        f"altitude response heading-dependent: max delta {np.max(np.abs(a0 - a90)):.3f} m"


@pytest.mark.cuda
@pytest.mark.xfail(reason=GOLDENS_PENDING, strict=False)
def test_attitude_reset_obs_matches_archive():
    import warp as wp
    from falcons.envs.attitude import AttitudeEnv
    from falcons.envs.configs import attitude_env_cfg
    g = np.load(Path(__file__).parent / "golden" / "attitude_obs.npy")
    wp.init(); torch.manual_seed(0)
    np.testing.assert_array_equal(AttitudeEnv(8, "cuda", attitude_env_cfg("Airship_V7")).reset().cpu().numpy(), g)
