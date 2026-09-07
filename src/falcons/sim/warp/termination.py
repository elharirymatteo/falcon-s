import warp as wp

# WARP kernel for termination checking
@wp.kernel
def check_termination_kernel(
    position: wp.array(dtype=wp.vec3f),
    alpha: wp.array(dtype=wp.float32),
    env_idx: int,
    wing_z_offset: float,
    stall_angle: float,
    crashed_flag: wp.array(dtype=wp.int32),
    stalled_flag: wp.array(dtype=wp.int32)
):
    # Check altitude (position z is negative when above ground)
    altitude = -position[env_idx][2]
    if altitude < -wing_z_offset:
        crashed_flag[0] = 1
    
    # Check stall angle
    if wp.abs(alpha[env_idx]) > 2.0 * stall_angle:
        stalled_flag[0] = 1


# Per-env batched termination, matching CoreAircraftEnv.check_done:
#   crash:  -pos_z + wing_cg_z < 0
#   stall:  |alpha| > alpha_limit (rad)  OR  Va < va_min
# reason code: 0 none, 1 crashed, 2 stalled (crash takes priority).
@wp.kernel
def check_termination_batch(
    position: wp.array(dtype=wp.vec3f),
    alpha: wp.array(dtype=wp.float32),
    Va: wp.array(dtype=wp.float32),
    wing_cg_z: wp.float32,
    alpha_limit: wp.float32,
    va_min: wp.float32,
    terminated: wp.array(dtype=wp.int32),
    reason: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if (-position[tid][2] + wing_cg_z) < 0.0:
        terminated[tid] = 1
        reason[tid] = 1
    elif wp.abs(alpha[tid]) > alpha_limit or Va[tid] < va_min:
        terminated[tid] = 1
        reason[tid] = 2
    else:
        terminated[tid] = 0
        reason[tid] = 0