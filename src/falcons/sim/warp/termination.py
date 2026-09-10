import warp as wp

# Termination is two conditions, and `alpha_max` is passed in ALREADY as the limit -- no
# multiplier is applied here. It used to be `2.0 * stall_angle` while its only caller passed
# `2 * VP.stall_angle`, so the effective limit was 4x the stall angle, or 52 deg. The limit is
# now one explicit number from the airframe JSON (`alpha_max_deg`), applied once.
#
# It is not a stall: the derivative aero model has no stall and keeps generating lift past this.
# It is the edge of the envelope the coefficients were fitted in.
@wp.kernel
def check_termination_kernel(
    position: wp.array(dtype=wp.vec3f),
    alpha: wp.array(dtype=wp.float32),
    env_idx: int,
    alpha_max: float,
    crashed_flag: wp.array(dtype=wp.int32),
    stalled_flag: wp.array(dtype=wp.int32)
):
    # Crash: the CG reaching the ground. This was the wing height, offset by the wing's z
    # displacement from the CG, until the empirical ground-effect model that needed that offset
    # was replaced by the measured OpenVSP sweep.
    if -position[env_idx][2] < 0.0:
        crashed_flag[0] = 1

    if wp.abs(alpha[env_idx]) > alpha_max:
        stalled_flag[0] = 1


# Per-env batched termination, matching CoreAircraftEnv.check_done:
#   crash:  -pos_z < 0
#   alpha:  |alpha| > alpha_max (rad)  OR  Va < va_min
# reason code: 0 none, 1 crashed, 2 alpha/airspeed (crash takes priority).
@wp.kernel
def check_termination_batch(
    position: wp.array(dtype=wp.vec3f),
    alpha: wp.array(dtype=wp.float32),
    Va: wp.array(dtype=wp.float32),
    alpha_max: wp.float32,
    va_min: wp.float32,
    terminated: wp.array(dtype=wp.int32),
    reason: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if -position[tid][2] < 0.0:
        terminated[tid] = 1
        reason[tid] = 1
    elif wp.abs(alpha[tid]) > alpha_max or Va[tid] < va_min:
        terminated[tid] = 1
        reason[tid] = 2
    else:
        terminated[tid] = 0
        reason[tid] = 0