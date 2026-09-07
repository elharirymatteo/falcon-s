import warp as wp

@wp.func
def scale_action(action: wp.float32, lower_limit: wp.float32, upper_limit: wp.float32) -> wp.float32:
    return (action + 1.0) * 0.5 * (upper_limit - lower_limit) + lower_limit

@wp.func
def clip_value(value: wp.float32, min_val: wp.float32, max_val: wp.float32) -> wp.float32:
    return wp.min(wp.max(value, min_val), max_val)

@wp.func
def first_order_dynamics_euler(action: wp.float32, x: wp.float32, dt: wp.float32, T: wp.float32) -> wp.float32:
    return x + dt * (1. / T) * (action - x)

@wp.func
def second_order_dynamics_euler(action: wp.float32, x: wp.float32, x_dot: wp.float32, dt: wp.float32, omega_0: wp.float32, zeta: wp.float32):
    x_dot_new = x_dot + dt * (-2. * zeta * omega_0 * x_dot - omega_0 * omega_0 * x + omega_0 * omega_0 * action)
    x_new = x + dt * x_dot
    return x_new, x_dot_new

@wp.func
def first_order_dynamics_rk45(action: wp.float32, x: wp.float32, dt: wp.float32, T: wp.float32) -> wp.float32:
    """
    RK45 integration for first-order dynamics: dx/dt = (1/T) * (action - x)
    """
    # RK45 coefficients
    # DP-specific coefficients
    a21 = 1.0/5.0
    a31, a32 = 3.0/40.0, 9.0/40.0
    a41, a42, a43 = 44.0/45.0, -56.0/15.0, 32.0/9.0
    a51, a52, a53, a54 = 19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0
    a61, a62, a63, a64, a65 = 9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0

    # 5th-order weights
    b1, b3, b4, b5, b6 = 35.0/384.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0
    
    # Derivative function inlined: f(x_val) = (1/T) * (action - x_val)
    # RK45 stages
    k1 = dt * ((1.0 / T) * (action - x))
    k2 = dt * ((1.0 / T) * (action - (x + a21 * k1)))
    k3 = dt * ((1.0 / T) * (action - (x + a31 * k1 + a32 * k2)))
    k4 = dt * ((1.0 / T) * (action - (x + a41 * k1 + a42 * k2 + a43 * k3)))
    k5 = dt * ((1.0 / T) * (action - (x + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)))
    k6 = dt * ((1.0 / T) * (action - (x + a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)))
    
    # 4th order estimate
    return x + b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6

@wp.func
def second_order_dynamics_rk45(action: wp.float32, x: wp.float32, x_dot: wp.float32, dt: wp.float32, omega_0: wp.float32, zeta: wp.float32):
    """
    RK45 integration for second-order dynamics
    State vector: [x, x_dot]
    Derivatives: [x_dot, -2*zeta*omega_0*x_dot - omega_0^2*x + omega_0^2*action]
    """
    # RK45 coefficients
    # DP-specific coefficients
    a21 = 1.0/5.0
    a31, a32 = 3.0/40.0, 9.0/40.0
    a41, a42, a43 = 44.0/45.0, -56.0/15.0, 32.0/9.0
    a51, a52, a53, a54 = 19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0
    a61, a62, a63, a64, a65 = 9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0

    # 5th-order weights
    b1, b3, b4, b5, b6 = 35.0/384.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0
    
    # Define derivative functions inline
    # f1: dx/dt = x_dot
    # f2: dx_dot/dt = -2*zeta*omega_0*x_dot - omega_0^2*x + omega_0^2*action
    
    # RK45 stages for both state variables
    k1_x = dt * x_dot
    k1_xdot = dt * (-2.0 * zeta * omega_0 * x_dot - omega_0 * omega_0 * x + omega_0 * omega_0 * action)
    
    k2_x = dt * (x_dot + a21 * k1_xdot)
    k2_xdot = dt * (-2.0 * zeta * omega_0 * (x_dot + a21 * k1_xdot) - omega_0 * omega_0 * (x + a21 * k1_x) + omega_0 * omega_0 * action)
    
    k3_x = dt * (x_dot + a31 * k1_xdot + a32 * k2_xdot)
    k3_xdot = dt * (-2.0 * zeta * omega_0 * (x_dot + a31 * k1_xdot + a32 * k2_xdot) - omega_0 * omega_0 * (x + a31 * k1_x + a32 * k2_x) + omega_0 * omega_0 * action)
    
    k4_x = dt * (x_dot + a41 * k1_xdot + a42 * k2_xdot + a43 * k3_xdot)
    k4_xdot = dt * (-2.0 * zeta * omega_0 * (x_dot + a41 * k1_xdot + a42 * k2_xdot + a43 * k3_xdot) - omega_0 * omega_0 * (x + a41 * k1_x + a42 * k2_x + a43 * k3_x) + omega_0 * omega_0 * action)
    
    k5_x = dt * (x_dot + a51 * k1_xdot + a52 * k2_xdot + a53 * k3_xdot + a54 * k4_xdot)
    k5_xdot = dt * (-2.0 * zeta * omega_0 * (x_dot + a51 * k1_xdot + a52 * k2_xdot + a53 * k3_xdot + a54 * k4_xdot) - omega_0 * omega_0 * (x + a51 * k1_x + a52 * k2_x + a53 * k3_x + a54 * k4_x) + omega_0 * omega_0 * action)
    
    k6_x = dt * (x_dot + a61 * k1_xdot + a62 * k2_xdot + a63 * k3_xdot + a64 * k4_xdot + a65 * k5_xdot)
    k6_xdot = dt * (-2.0 * zeta * omega_0 * (x_dot + a61 * k1_xdot + a62 * k2_xdot + a63 * k3_xdot + a64 * k4_xdot + a65 * k5_xdot) - omega_0 * omega_0 * (x + a61 * k1_x + a62 * k2_x + a63 * k3_x + a64 * k4_x + a65 * k5_x) + omega_0 * omega_0 * action)
    
    # 4th order estimates
    x_new = x + b1*k1_x + b3*k3_x + b4*k4_x + b5*k5_x + b6*k6_x
    x_dot_new = x_dot + b1*k1_xdot + b3*k3_xdot + b4*k4_xdot + b5*k5_xdot + b6*k6_xdot
    
    return x_new, x_dot_new


@wp.func
def process_actuator_complete(
    # Input commands (from RL agent)
    elevator_cmd: wp.float32,
    aileron_cmd: wp.float32,
    rudder_cmd: wp.float32,
    throttle_left_cmd: wp.float32,
    throttle_right_cmd: wp.float32,

    # Time step
    dt: wp.float32,

    # Dynamics parameters
    T_s: wp.float32,
    zeta: wp.float32,
    omega_0: wp.float32,
    
    # Actuator limits
    elevator_limits: wp.vec2,
    aileron_limits: wp.vec2,
    rudder_limits: wp.vec2,
    throttle_limits: wp.vec2,
    
    # Output states (updated for next iteration)
    elevator_state: wp.float32,
    aileron_state: wp.float32,
    rudder_state: wp.float32,
    throttle_left_state: wp.float32,
    throttle_right_state: wp.float32,

    # Output derivatives for second-order actuators
    elevator_dot: wp.float32,
    aileron_dot: wp.float32,
    rudder_dot: wp.float32,

    # Solver type (0 for Euler, 1 for RK45)
    solver_type: wp.int32,
) -> None:

    # Process elevator (second-order dynamics)
    clipped_elevator = clip_value(elevator_cmd, -1.0, 1.0)
    scaled_elevator = scale_action(clipped_elevator, elevator_limits[0], elevator_limits[1])
    
    if solver_type == 0:
        x_new, x_dot_new = second_order_dynamics_euler(scaled_elevator, elevator_state, elevator_dot, dt, omega_0, zeta)
    else:  # RK45
        x_new, x_dot_new = second_order_dynamics_rk45(scaled_elevator, elevator_state, elevator_dot, dt, omega_0, zeta)
    
    elevator_state = clip_value(x_new, elevator_limits[0], elevator_limits[1])
    elevator_dot = x_dot_new


    # Process aileron (second-order dynamics)
    clipped_aileron = clip_value(aileron_cmd, -1.0, 1.0)
    scaled_aileron = scale_action(clipped_aileron, aileron_limits[0], aileron_limits[1])
    
    if solver_type == 0:
        x_new, x_dot_new = second_order_dynamics_euler(scaled_aileron, aileron_state, aileron_dot, dt, omega_0, zeta)
    else:  # RK45
        x_new, x_dot_new = second_order_dynamics_rk45(scaled_aileron, aileron_state, aileron_dot, dt, omega_0, zeta)
    
    aileron_state = clip_value(x_new, aileron_limits[0], aileron_limits[1])
    aileron_dot = x_dot_new


    # Process rudder (second-order dynamics)
    clipped_rudder = clip_value(rudder_cmd, -1.0, 1.0)
    scaled_rudder = scale_action(clipped_rudder, rudder_limits[0], rudder_limits[1])
    
    if solver_type == 0:
        x_new, x_dot_new = second_order_dynamics_euler(scaled_rudder, rudder_state, rudder_dot, dt, omega_0, zeta)
    else:  # RK45
        x_new, x_dot_new = second_order_dynamics_rk45(scaled_rudder, rudder_state, rudder_dot, dt, omega_0, zeta)
    
    rudder_state = clip_value(x_new, rudder_limits[0], rudder_limits[1])
    rudder_dot = x_dot_new


    # Process left throttle (first-order dynamics)
    clipped_throttle_left = clip_value(throttle_left_cmd, -1.0, 1.0)
    scaled_throttle_left = scale_action(clipped_throttle_left, throttle_limits[0], throttle_limits[1])

    if solver_type == 0:
        throttle_left_state = first_order_dynamics_euler(scaled_throttle_left, throttle_left_state, dt, T_s)
    else:  # RK45
        throttle_left_state = first_order_dynamics_rk45(scaled_throttle_left, throttle_left_state, dt, T_s)
    
    throttle_left_state = clip_value(throttle_left_state, throttle_limits[0], throttle_limits[1])


    # Process right throttle (first-order dynamics)
    clipped_throttle_right = clip_value(throttle_right_cmd, -1.0, 1.0)
    scaled_throttle_right = scale_action(clipped_throttle_right, throttle_limits[0], throttle_limits[1])    
    if solver_type == 0:
        throttle_right_state = first_order_dynamics_euler(scaled_throttle_right, throttle_right_state, dt, T_s)
    else:  # RK45
        throttle_right_state = first_order_dynamics_rk45(scaled_throttle_right, throttle_right_state, dt, T_s)   
    throttle_right_state = clip_value(throttle_right_state, throttle_limits[0], throttle_limits[1])

    return elevator_state, aileron_state, rudder_state, throttle_left_state, throttle_right_state, elevator_dot, aileron_dot, rudder_dot