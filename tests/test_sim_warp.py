"""Warp kernel guards: termination, the altitude observation vector, the sensor pipeline, the
frame the attitude integrator works in, and one whole AltitudeEnv step against the archive.

The first three come from standalone archive scripts -- two `run()` bodies behind `__main__` guards
that printed a PASS/FAIL table, and a sensor script that plotted truth against measurement and
asserted nothing. The numeric setups below are theirs, so what is asserted is the archive's own
reference (its numpy re-implementation of `check_done`, its re-implementation of `_extract_state`,
its case table) rather than numbers invented here.
"""
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

from conftest import GOLDENS_PENDING, requires_derivatives

# Every test here builds Airship_V7 from its OpenVSP derivative set.
pytestmark = requires_derivatives("Airship_V7")

from falcons.aircraft.params import load_params
from falcons.sim.warp.aircraft import Aircraft
from falcons.sim.warp.altitude_obs import compute_obs
from falcons.sim.warp.termination import check_termination_batch
from falcons.sim.torch.altitude import AltitudeEnv, quat_normalize

# ───────────────────────────── termination ─────────────────────────────
# Read from the airframe, not restated here: `alpha_max_deg` is the single hard incidence gate and
# a copy of it in the test would drift silently. It is NOT a stall -- the derivative model has no
# stall and keeps generating lift past this angle; it is the edge of the envelope the coefficients
# were fitted in, so a rollout beyond it is extrapolation rather than flight.
ALPHA_LIMIT = float(load_params("Airship_V7")["vehicle_params"].alpha_max)   # rad
VA_MIN = 10.0

# The crash line is now the CG reaching the ground, `-pos_z < 0`. It used to sit 0.293 m higher
# for Airship_V7, because the check measured WING height via the wing's z-offset from the CG --
# an offset that existed only to feed the empirical ground-effect model. That model is gone and so
# is the offset, which moves V7's crash altitude down by 29 cm.
#
# (pos_z, alpha_rad, Va, expected_reason)  reason: 0 none, 1 crash, 2 alpha/airspeed
CASES = [
    (-50.0, 0.0, 28.0, 0),    # normal cruise
    (+0.10, 0.0, 28.0, 1),    # 0.10 m BELOW the ground (pos_z is NED: positive is underground)
    (-0.05, 0.0, 28.0, 0),    # 5 cm up and still flying -- would have been a crash before
    (-50.0, 0.35, 28.0, 2),   # past alpha_max (0.35 > 0.3403)
    (-50.0, 0.33, 28.0, 0),   # just inside the envelope
    (-50.0, 0.0, 9.0, 2),     # low airspeed
    (+0.05, 0.50, 9.0, 1),    # underground AND past alpha -> crash wins (priority)
]


def numpy_reason(pos_z, alpha, Va):
    """CoreAircraftEnv.check_done, in numpy: the reference the kernel has to agree with."""
    if -pos_z < 0.0:
        return 1
    if abs(alpha) > ALPHA_LIMIT or Va < VA_MIN:
        return 2
    return 0


def test_termination_reference_covers_crash_stall_and_healthy():
    """The case table and the numpy reference must agree before either is used to judge the kernel,
    and between them the cases have to exercise all three verdicts."""
    assert [numpy_reason(*c[:3]) for c in CASES] == [c[3] for c in CASES]
    assert {c[3] for c in CASES} == {0, 1, 2}


@pytest.mark.cuda
def test_termination_kernel_matches_check_done():
    """Same seven states through check_termination_batch: the reason it reports must be the numpy
    one case for case (crash outranking stall in the last), and `terminated` must be set exactly
    when the reason is non-zero."""
    wp.init()
    n = len(CASES)
    pos = wp.array(np.array([[0.0, 0.0, c[0]] for c in CASES], dtype=np.float32),
                   dtype=wp.vec3f, device="cuda")
    alpha = wp.array(np.array([c[1] for c in CASES], dtype=np.float32), device="cuda")
    Va = wp.array(np.array([c[2] for c in CASES], dtype=np.float32), device="cuda")
    terminated = wp.zeros(n, dtype=wp.int32, device="cuda")
    reason = wp.zeros(n, dtype=wp.int32, device="cuda")

    wp.launch(check_termination_batch, dim=n,
              inputs=[pos, alpha, Va, float(ALPHA_LIMIT), VA_MIN,
                      terminated, reason], device="cuda")

    assert list(reason.numpy()) == [c[3] for c in CASES]
    assert list(terminated.numpy()) == [int(bool(c[3])) for c in CASES]


# ───────────────────────────── altitude observation ─────────────────────────────
TARGET_ALT = 50.0
TARGET_VA = 27.0
ELEV_MIN, ELEV_MAX = -20.0, 20.0
THR_MIN, THR_MAX = 0.0, 1.0
SCALES = np.array([25.0, 2.0, 15.0, np.pi / 4, 15.0, np.radians(20),
                   1.0, 100.0, 1.0, 10.0, 5.0])
N = 8           # envs
ALTS = np.linspace(20.0, 80.0, N)    # spread over altitudes so h_error varies across the batch
THROTTLE = 0.45


def build_warp():
    co = load_params("Airship_V7")
    AP = co["aero_params"].as_warp_struct()
    VP = co["vehicle_params"]
    VP.WP = VP.WP.as_warp_struct()
    VP.J = VP.J.as_warp_struct()
    VP.PP = VP.PP.as_warp_struct()
    if hasattr(VP, "sensor_system"):
        VP.sensor_system = None
    config = {"AP": AP, "VP": VP, "CL": co["control_limits"], "EP": co["environment_params"]}
    m = Aircraft(N, "cuda", config, save_history=False)
    m.solver_type = 1
    pos = np.zeros((N, 3), dtype=np.float32)
    pos[:, 2] = -ALTS
    vel = np.tile(np.array([28.0, 0.0, 0.0], dtype=np.float32), (N, 1))
    m.reset(
        {"position": pos, "linear_vel": vel, "angular_vel": np.zeros((N, 3), dtype=np.float32),
         "orientation": np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (N, 1))},
        {"elevator": 0.0, "aileron": 0.0, "rudder": 0.0,
         "throttle_left": THROTTLE, "throttle_right": THROTTLE,
         "elevator_dot": 0.0, "aileron_dot": 0.0, "rudder_dot": 0.0},
    )
    return m


def python_obs(m):
    """AltitudeKeepingEnv._extract_state, re-implemented against the SAME warp state the kernel
    reads (parity of that state against the numpy plant is a separate test)."""
    pos = m._state["position"].numpy()
    vel = m._state["linear_vel"].numpy()
    ang = m._state["angular_vel"].numpy()
    ori = m._state["orientation"].numpy()  # [x,y,z,w]
    Va = m._Va.numpy()
    alpha = m._alpha.numpy()
    elev = m._actuator_states["elevator"].numpy()
    elev_dot = m._actuator_states["elevator_dot"].numpy()
    thr = m._actuator_states["throttle_left"].numpy()

    out = np.zeros((N, 11))
    for i in range(N):
        x, y, z, w = ori[i]
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        elev_norm = (elev[i] - ELEV_MIN) / (ELEV_MAX - ELEV_MIN) * 2 - 1
        thr_norm = (thr[i] - THR_MIN) / (THR_MAX - THR_MIN) * 2 - 1
        raw = np.array([-pos[i][2] - TARGET_ALT, ang[i][1], Va[i] - TARGET_VA,
                        pitch, vel[i][2], alpha[i], elev_norm, elev_dot[i], thr_norm, 0.0, 0.0])
        out[i] = np.clip(raw / SCALES, -3.0, 3.0)
    return out


def kernel_obs(m):
    target = wp.array(np.full(N, TARGET_ALT, dtype=np.float32), device="cuda")
    target_va = wp.array(np.full(N, TARGET_VA, dtype=np.float32), device="cuda")
    ierr = wp.zeros(N, dtype=wp.float32, device="cuda")
    refrate = wp.zeros(N, dtype=wp.float32, device="cuda")
    obs = wp.zeros((N, 11), dtype=wp.float32, device="cuda")
    wp.launch(compute_obs, dim=N, inputs=[
        m._state["position"], m._state["linear_vel"], m._state["angular_vel"],
        m._state["orientation"], m._Va, m._alpha,
        m._actuator_states["elevator"], m._actuator_states["elevator_dot"],
        m._actuator_states["throttle_left"],
        ierr, refrate, target, target_va, ELEV_MIN, ELEV_MAX, THR_MIN, THR_MAX,
        15.0, 15.0, obs],
        device="cuda")
    return obs.numpy()


@pytest.mark.cuda
def test_obs_vector_is_the_scaled_error_state():
    """Shape and the components the spawn fixes: eleven channels per env, altitude error over the
    20-80 m spread scaled by 25 m, throttle mapped from [0, 1] onto [-1, 1], and the integral-error
    and reference-rate channels reading the (zero) arrays they are handed rather than the state."""
    wp.init()
    obs = kernel_obs(build_warp())
    assert obs.shape == (N, 11)
    assert np.allclose(obs[:, 0], np.clip((ALTS - TARGET_ALT) / SCALES[0], -3.0, 3.0), atol=1e-6)
    assert np.allclose(obs[:, 8], THROTTLE * 2 - 1, atol=1e-6)
    assert np.array_equal(obs[:, 9:], np.zeros((N, 2), dtype=np.float32))


@pytest.mark.cuda
def test_obs_kernel_matches_extract_state_over_fifty_steps():
    """The archive's own check: drive the plant 50 steps (elevator steps down at t=20 so pitch,
    alpha and the actuator channels all move) and compare the kernel with the python formula fed
    the same state. Its threshold was 1e-4."""
    wp.init()
    m = build_warp()
    worst = 0.0
    for t in range(50):
        elev = -0.05 if t > 20 else 0.0
        m.step({"elevator": elev, "aileron": 0.0, "rudder": 0.0,
                "throttle_left": THROTTLE, "throttle_right": THROTTLE})
        worst = max(worst, np.abs(kernel_obs(m) - python_obs(m)).max())
    assert worst < 1e-4


# ───────────────────────────── sensors ─────────────────────────────
SENSOR_SHAPES = {"gps_position": (1, 3), "gps_velocity": (1, 3), "gyroscope": (1, 3),
                 "attitude_sensor": (1, 4)}


@pytest.mark.cuda
def test_sensors_shapes_and_finite():
    """The sensor pipeline alongside the plant: 200 steps of one env, every step returning a
    measurement of every channel at the channel's own width, with a validity flag each, and
    nothing NaN or infinite on either the true or the measured side."""
    wp.init()
    co = load_params("Airship_V7")
    AP = co["aero_params"].as_warp_struct()
    VP = co["vehicle_params"]
    VP.WP = VP.WP.as_warp_struct()
    VP.J = VP.J.as_warp_struct()
    VP.PP = VP.PP.as_warp_struct()
    config = {"AP": AP, "VP": VP, "CL": co["control_limits"], "EP": co["environment_params"]}
    model = Aircraft(1, "cuda", config, save_history=False)
    model.reset({"position": np.array([0, 0, -10.0]), "linear_vel": np.array([20, 0, 0]),
                 "angular_vel": np.array([0, 0, 0]), "orientation": np.array([0, 0, 0, 1])})

    steps = 0
    for _ in range(200):
        state, sensors = model.step(return_sensors=True)
        assert sensors, "the sensor system returned nothing for a step"
        steps += 1
        for k, shape in SENSOR_SHAPES.items():
            meas = sensors[k].numpy()
            assert meas.shape == shape
            assert np.isfinite(meas).all(), f"{k} is not finite"
        assert set(sensors["validity"]) == set(SENSOR_SHAPES)
        assert all(v.numpy().shape == (1,) for v in sensors["validity"].values())
        for k in ("position", "linear_vel"):
            assert np.isfinite(state[k].numpy()).all(), f"true {k} is not finite"
    assert steps == 200


# ───────────────────────────── attitude integration frame ─────────────────────────────
def _euler_deg(q):
    x, y, z, w = q.tolist()
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def test_body_pitch_rate_at_yaw90_produces_pitch():
    """Analytic attitude-propagation test: a pure BODY pitch rate must produce pitch regardless of
    heading. Catches the world/body quaternion-derivative frame swap."""
    env = AltitudeEnv(1, "cpu", {"aircraft": "Volantex_Ranger"})
    p = torch.zeros(1, 3)
    v = torch.zeros(1, 3)
    w = torch.tensor([[0.0, 0.5, 0.0]])                # pure body pitch rate [rad/s]
    q = torch.tensor([[0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]])  # yaw = 90 deg
    dt = 0.001
    for _ in range(1000):                              # integrate 1 s; Fb=Mb=0 keeps w constant
        _, _, _, dq = env._deriv(p, v, w, q, torch.zeros(1, 3), torch.zeros(1, 3))
        q = quat_normalize(q + dt * dq)
    roll, pitch, yaw = _euler_deg(q[0])
    assert abs(pitch - math.degrees(0.5)) < 0.5        # expect 28.6 deg pitch
    assert abs(roll) < 0.5                             # buggy form puts the 28.6 deg here
    assert abs(yaw - 90.0) < 0.5


# ───────────────────────────── whole-env step vs the archive ─────────────────────────────
@pytest.mark.cuda
@pytest.mark.xfail(reason=GOLDENS_PENDING, strict=False)
def test_warp_altitude_step_matches_archive():
    """One reset + one zero-action step from a fixed seed must equal the archive's output exactly.
    Anything else means the copy changed the plant."""
    from falcons.envs.altitude import AltitudeEnv
    from falcons.envs.configs import altitude_env_cfg, PLANE_CONFIGS
    g = np.load(Path(__file__).parent / "golden" / "warp_step.npz")
    wp.init(); torch.manual_seed(0)
    env = AltitudeEnv(8, "cuda", altitude_env_cfg("Airship_V7", PLANE_CONFIGS["Airship_V7"]["spawn"]))
    obs0 = env.reset().clone()
    obs1, rew, done, _ = env.step(torch.zeros(8, env.num_act, device="cuda"))
    np.testing.assert_array_equal(obs0.cpu().numpy(), g["obs0"])
    np.testing.assert_array_equal(obs1.cpu().numpy(), g["obs1"])
    np.testing.assert_array_equal(rew.cpu().numpy(), g["rew"])
