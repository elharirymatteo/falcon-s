import warp as wp

# GPU kernel for history saving
@wp.kernel
def save_history_kernel(
    step: int,
    env_idx: int,
    # Current state
    alpha: wp.array(dtype=wp.float32),
    beta: wp.array(dtype=wp.float32),
    Va: wp.array(dtype=wp.float32),
    hr: wp.array(dtype=wp.float32),
    Fb: wp.array(dtype=wp.vec3f),
    Mb: wp.array(dtype=wp.vec3f),
    Fb_aero: wp.array(dtype=wp.vec3f),
    Mb_aero: wp.array(dtype=wp.vec3f),
    Fb_thrust: wp.array(dtype=wp.vec3f),
    Mb_thrust: wp.array(dtype=wp.vec3f),
    Fb_g: wp.array(dtype=wp.vec3f),
    C_L: wp.array(dtype=wp.float32),
    C_D: wp.array(dtype=wp.float32),
    C_L_ige: wp.array(dtype=wp.float32),
    C_D_ige: wp.array(dtype=wp.float32),
    Cl: wp.array(dtype=wp.float32),
    Cm: wp.array(dtype=wp.float32),
    Cn: wp.array(dtype=wp.float32),
    elevator: wp.array(dtype=wp.float32),
    aileron: wp.array(dtype=wp.float32),
    rudder: wp.array(dtype=wp.float32),
    throttle_left: wp.array(dtype=wp.float32),
    throttle_right: wp.array(dtype=wp.float32),
    position: wp.array(dtype=wp.vec3f),
    orientation: wp.array(dtype=wp.quatf),
    linear_vel: wp.array(dtype=wp.vec3f),
    angular_vel: wp.array(dtype=wp.vec3f),
    # History buffers
    history_alpha: wp.array(dtype=wp.float32),
    history_beta: wp.array(dtype=wp.float32),
    history_Va: wp.array(dtype=wp.float32),
    history_hr: wp.array(dtype=wp.float32),
    history_Fb: wp.array2d(dtype=wp.float32),
    history_Mb: wp.array2d(dtype=wp.float32),
    history_Fb_aero: wp.array2d(dtype=wp.float32),
    history_Mb_aero: wp.array2d(dtype=wp.float32),
    history_Fb_thrust: wp.array2d(dtype=wp.float32),
    history_Mb_thrust: wp.array2d(dtype=wp.float32),
    history_Fb_g: wp.array2d(dtype=wp.float32),
    history_C_L: wp.array(dtype=wp.float32),
    history_C_D: wp.array(dtype=wp.float32),
    history_C_L_ige: wp.array(dtype=wp.float32),
    history_C_D_ige: wp.array(dtype=wp.float32),
    history_Cl: wp.array(dtype=wp.float32),
    history_Cm: wp.array(dtype=wp.float32),
    history_Cn: wp.array(dtype=wp.float32),
    history_elevator: wp.array(dtype=wp.float32),
    history_aileron: wp.array(dtype=wp.float32),
    history_rudder: wp.array(dtype=wp.float32),
    history_throttle_left: wp.array(dtype=wp.float32),
    history_throttle_right: wp.array(dtype=wp.float32),
    history_position: wp.array2d(dtype=wp.float32),
    history_orientation: wp.array2d(dtype=wp.float32),
    history_linear_vel: wp.array2d(dtype=wp.float32),
    history_angular_vel: wp.array2d(dtype=wp.float32)
) -> None:
    # Save scalars
    history_alpha[step] = alpha[env_idx]
    history_beta[step] = beta[env_idx]
    history_Va[step] = Va[env_idx]
    history_hr[step] = hr[env_idx]
    
    # Save vectors (forces)
    fb = Fb[env_idx]
    history_Fb[step, 0] = fb[0]
    history_Fb[step, 1] = fb[1]
    history_Fb[step, 2] = fb[2]
    
    mb = Mb[env_idx]
    history_Mb[step, 0] = mb[0]
    history_Mb[step, 1] = mb[1]
    history_Mb[step, 2] = mb[2]
    
    fb_aero = Fb_aero[env_idx]
    history_Fb_aero[step, 0] = fb_aero[0]
    history_Fb_aero[step, 1] = fb_aero[1]
    history_Fb_aero[step, 2] = fb_aero[2]
    
    mb_aero = Mb_aero[env_idx]
    history_Mb_aero[step, 0] = mb_aero[0]
    history_Mb_aero[step, 1] = mb_aero[1]
    history_Mb_aero[step, 2] = mb_aero[2]
    
    fb_thrust = Fb_thrust[env_idx]
    history_Fb_thrust[step, 0] = fb_thrust[0]
    history_Fb_thrust[step, 1] = fb_thrust[1]
    history_Fb_thrust[step, 2] = fb_thrust[2]
    
    mb_thrust = Mb_thrust[env_idx]
    history_Mb_thrust[step, 0] = mb_thrust[0]
    history_Mb_thrust[step, 1] = mb_thrust[1]
    history_Mb_thrust[step, 2] = mb_thrust[2]
    
    fb_g = Fb_g[env_idx]
    history_Fb_g[step, 0] = fb_g[0]
    history_Fb_g[step, 1] = fb_g[1]
    history_Fb_g[step, 2] = fb_g[2]
    
    # Save coefficients
    history_C_L[step] = C_L[env_idx]
    history_C_D[step] = C_D[env_idx]
    history_C_L_ige[step] = C_L_ige[env_idx]
    history_C_D_ige[step] = C_D_ige[env_idx]
    history_Cl[step] = Cl[env_idx]
    history_Cm[step] = Cm[env_idx]
    history_Cn[step] = Cn[env_idx]
    
    # Save actions
    history_elevator[step] = elevator[env_idx]
    history_aileron[step] = aileron[env_idx]
    history_rudder[step] = rudder[env_idx]
    history_throttle_left[step] = throttle_left[env_idx]
    history_throttle_right[step] = throttle_right[env_idx]
    
    # Save states
    pos = position[env_idx]
    history_position[step, 0] = pos[0]
    history_position[step, 1] = pos[1]
    history_position[step, 2] = pos[2]
    
    quat = orientation[env_idx]
    history_orientation[step, 0] = quat[0]
    history_orientation[step, 1] = quat[1]
    history_orientation[step, 2] = quat[2]
    history_orientation[step, 3] = quat[3]
    
    vel = linear_vel[env_idx]
    history_linear_vel[step, 0] = vel[0]
    history_linear_vel[step, 1] = vel[1]
    history_linear_vel[step, 2] = vel[2]
    
    omega = angular_vel[env_idx]
    history_angular_vel[step, 0] = omega[0]
    history_angular_vel[step, 1] = omega[1]
    history_angular_vel[step, 2] = omega[2]