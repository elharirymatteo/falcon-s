import warp as wp

# 9-dim altitude-keeping observation, matching AltitudeKeepingEnv._extract_state:
#   [h_error, pitch_rate, Va_error, pitch, vz, alpha, elev_norm, elev_rate, thr_norm]
# each divided by a physical scale and clipped to [-3, 3].
SCALE0 = 25.0
SCALE1 = 2.0
SCALE2 = 15.0
SCALE3 = wp.constant(0.7853981633974483)   # pi/4
SCALE4 = 15.0
SCALE5 = wp.constant(0.3490658503988659)   # radians(20)
SCALE6 = 1.0
SCALE7 = 100.0
SCALE8 = 1.0
SCALE9 = 10.0   # integral of altitude error [m*s]
SCALE10 = 5.0   # reference climb rate (feedforward) [m/s]


@wp.func
def _clip3(x: wp.float32) -> wp.float32:
    return wp.clamp(x, -3.0, 3.0)


@wp.kernel
def compute_obs(
    position: wp.array(dtype=wp.vec3f),
    linear_vel: wp.array(dtype=wp.vec3f),
    angular_vel: wp.array(dtype=wp.vec3f),
    orientation: wp.array(dtype=wp.quatf),
    Va: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    elevator_state: wp.array(dtype=wp.float32),
    elevator_dot: wp.array(dtype=wp.float32),
    throttle_state: wp.array(dtype=wp.float32),
    integral_error: wp.array(dtype=wp.float32),
    ref_rate: wp.array(dtype=wp.float32),
    target_altitude: wp.array(dtype=wp.float32),
    target_airspeed: wp.array(dtype=wp.float32),   # per-env (A3 va_h task); constant = trim elsewhere
    elevator_min: wp.float32,
    elevator_max: wp.float32,
    throttle_min: wp.float32,
    throttle_max: wp.float32,
    va_scale: wp.float32,
    vz_scale: wp.float32,
    obs: wp.array(dtype=wp.float32, ndim=2),
):
    tid = wp.tid()
    q = orientation[tid]  # wp.quatf is [x, y, z, w]
    pitch = wp.asin(wp.clamp(2.0 * (q[3] * q[1] - q[2] * q[0]), -1.0, 1.0))

    elev_norm = (elevator_state[tid] - elevator_min) / (elevator_max - elevator_min) * 2.0 - 1.0
    thr_norm = (throttle_state[tid] - throttle_min) / (throttle_max - throttle_min) * 2.0 - 1.0

    obs[tid, 0] = _clip3((-position[tid][2] - target_altitude[tid]) / SCALE0)
    obs[tid, 1] = _clip3(angular_vel[tid][1] / SCALE1)
    obs[tid, 2] = _clip3((Va[tid] - target_airspeed[tid]) / va_scale)
    obs[tid, 3] = _clip3(pitch / SCALE3)
    obs[tid, 4] = _clip3(linear_vel[tid][2] / vz_scale)
    obs[tid, 5] = _clip3(alpha[tid] / SCALE5)
    obs[tid, 6] = _clip3(elev_norm / SCALE6)
    obs[tid, 7] = _clip3(elevator_dot[tid] / SCALE7)
    obs[tid, 8] = _clip3(thr_norm / SCALE8)
    obs[tid, 9] = _clip3(integral_error[tid] / SCALE9)
    obs[tid, 10] = _clip3(ref_rate[tid] / SCALE10)
