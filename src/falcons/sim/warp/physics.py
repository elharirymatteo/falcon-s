import warp as wp

from falcons.aircraft.params import (
    AerodynamicsParametersStruct,
    PropulsionParametersStruct,
    WingParametersStruct,
)

from falcons.sim.warp.aerodynamics import compute_all_coeffs, euler_update, rk45_update, compute_derivatives, compute_aerodynamic_parameters, compute_thrust_forces_and_moments, compute_aerodynamic_forces_and_moments, compute_forces_applied_to_body, compute_lift_and_drag_forces
from falcons.sim.warp.actuators import process_actuator_complete


@wp.kernel
def aircraft_simulation_step(
    # State inputs/outputs
    position: wp.array(dtype=wp.vec3f),
    linear_vel: wp.array(dtype=wp.vec3f),
    angular_vel: wp.array(dtype=wp.vec3f),
    orientation: wp.array(dtype=wp.quatf),
    
    # Action inputs
    elevator_cmd: wp.array(dtype=wp.float32),
    aileron_cmd: wp.array(dtype=wp.float32),
    rudder_cmd: wp.array(dtype=wp.float32),
    throttle_left_cmd: wp.array(dtype=wp.float32),
    throttle_right_cmd: wp.array(dtype=wp.float32),
    
    # Actuator states
    elevator_state: wp.array(dtype=wp.float32),
    aileron_state: wp.array(dtype=wp.float32),
    rudder_state: wp.array(dtype=wp.float32),
    throttle_left_state: wp.array(dtype=wp.float32),
    throttle_right_state: wp.array(dtype=wp.float32),
    elevator_dot: wp.array(dtype=wp.float32),
    aileron_dot: wp.array(dtype=wp.float32),
    rudder_dot: wp.array(dtype=wp.float32),
    
    # Aerodynamic parameters
    Va: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    beta: wp.array(dtype=wp.float32),
    rho: wp.array(dtype=wp.float32),
    Q: wp.array(dtype=wp.float32),
    hr: wp.array(dtype=wp.float32),
    airspeed_vector: wp.array(dtype=wp.vec3f),
    
    # Coefficients (temporary variables)
    C_D: wp.array(dtype=wp.float32),
    C_Y: wp.array(dtype=wp.float32),
    C_L: wp.array(dtype=wp.float32),
    Cl: wp.array(dtype=wp.float32),
    Cm: wp.array(dtype=wp.float32),
    Cn: wp.array(dtype=wp.float32),
    C_D_free: wp.array(dtype=wp.float32),
    C_L_free: wp.array(dtype=wp.float32),
    
    # Force/moment intermediates
    D_tot: wp.array(dtype=wp.float32),
    Y_tot: wp.array(dtype=wp.float32),
    L_tot: wp.array(dtype=wp.float32),
    F1_b: wp.array(dtype=wp.vec3f),
    F2_b: wp.array(dtype=wp.vec3f),
    Fb_thrust: wp.array(dtype=wp.vec3f),
    M_prop_left: wp.array(dtype=wp.float32),
    M_prop_right: wp.array(dtype=wp.float32),
    Mb_thrust: wp.array(dtype=wp.vec3f),
    Fw_aero: wp.array(dtype=wp.vec3f),
    Fb_aero: wp.array(dtype=wp.vec3f),
    Mb_aero: wp.array(dtype=wp.vec3f),
    q_aero: wp.array(dtype=wp.quatf),
    gravity_vector_body: wp.array(dtype=wp.vec3f),
    Fb_g: wp.array(dtype=wp.vec3f),
    Fb: wp.array(dtype=wp.vec3f),
    Mb: wp.array(dtype=wp.vec3f),
    
    # Wind effects
    total_linear_gusts: wp.array(dtype=wp.vec3f),
    
    # Configuration parameters
    Mac: wp.float32,
    dt: wp.float32,
    g: wp.float32,
    m: wp.float32,
    ge_enable: bool,

    # Configuration structs
    AP: AerodynamicsParametersStruct,
    WP: WingParametersStruct,
    J: wp.mat33,                     
    PP: PropulsionParametersStruct, 
    
    # Actuator parameters
    T_s: wp.float32,
    zeta: wp.float32,
    omega_0: wp.float32,
    elevator_limits: wp.vec2,
    aileron_limits: wp.vec2,
    rudder_limits: wp.vec2,
    throttle_limits: wp.vec2,
    
    # Solver type
    solver_type: wp.int32
) -> None:
    
    tid = wp.tid()
    
    # Step 1: Process actuator dynamics
    temp_elevator, temp_aileron, temp_rudder, temp_throttle_left, temp_throttle_right, temp_elevator_dot, temp_aileron_dot, temp_rudder_dot = \
        process_actuator_complete(
            elevator_cmd[tid], aileron_cmd[tid], rudder_cmd[tid], 
            throttle_left_cmd[tid], throttle_right_cmd[tid], 
            dt, T_s, zeta, omega_0, 
            elevator_limits, aileron_limits, rudder_limits, throttle_limits,
            elevator_state[tid], aileron_state[tid], rudder_state[tid], 
            throttle_left_state[tid], throttle_right_state[tid], 
            elevator_dot[tid], aileron_dot[tid], rudder_dot[tid],
            solver_type
        )

    elevator_state[tid] = temp_elevator
    aileron_state[tid] = temp_aileron
    rudder_state[tid] = temp_rudder
    throttle_left_state[tid] = temp_throttle_left
    throttle_right_state[tid] = temp_throttle_right
    elevator_dot[tid] = temp_elevator_dot
    aileron_dot[tid] = temp_aileron_dot
    rudder_dot[tid] = temp_rudder_dot

    # Step 2: Compute aerodynamic parameters
    temp_alpha, temp_beta, temp_rho, temp_Q, temp_hr, Va_temp, airspeed_vector_temp = \
        compute_aerodynamic_parameters(
            position[tid], linear_vel[tid], WP, rho[tid], Va[tid], 
            alpha[tid], beta[tid], Q[tid], hr[tid],
            total_linear_gusts[tid], airspeed_vector[tid]
        )

    alpha[tid] = temp_alpha
    beta[tid] = temp_beta
    rho[tid] = temp_rho
    Q[tid] = temp_Q
    hr[tid] = temp_hr
    Va[tid] = Va_temp
    airspeed_vector[tid] = airspeed_vector_temp

    # Step 3: Compute all aerodynamic coefficients.
    # The actuator states are in DEGREES -- they are scaled from each airframe's
    # min/max_deflection, which the JSONs give in degrees -- and the derivative set is per
    # radian. This is the boundary, so this is where the conversion happens.
    temp_C_D, temp_C_S, temp_C_L, temp_Cl, temp_Cm, temp_Cn, temp_C_D_free, temp_C_L_free = \
        compute_all_coeffs(
            AP, WP.span, WP.mac,
            alpha[tid], beta[tid], Va[tid],
            wp.radians(elevator_state[tid]), wp.radians(aileron_state[tid]),
            wp.radians(rudder_state[tid]),
            angular_vel[tid][0], angular_vel[tid][1], angular_vel[tid][2],
            -position[tid][2], ge_enable
        )

    C_D[tid] = temp_C_D
    C_Y[tid] = temp_C_S
    C_L[tid] = temp_C_L
    Cl[tid] = temp_Cl
    Cm[tid] = temp_Cm
    Cn[tid] = temp_Cn
    C_D_free[tid] = temp_C_D_free
    C_L_free[tid] = temp_C_L_free

    # Step 4: Compute lift and drag forces
    temp_D_tot, temp_Y_tot, temp_L_tot = \
        compute_lift_and_drag_forces(
            Q[tid], WP, C_D[tid], C_Y[tid], C_L[tid], D_tot[tid], Y_tot[tid], L_tot[tid]
        )

    D_tot[tid] = temp_D_tot
    Y_tot[tid] = temp_Y_tot
    L_tot[tid] = temp_L_tot

    # Step 5: Compute aerodynamic forces and moments
    temp_Fw_aero, temp_Mb_aero = \
        compute_aerodynamic_forces_and_moments(
            D_tot[tid], Y_tot[tid], L_tot[tid], Cl[tid], Cm[tid], Cn[tid],
            Q[tid], WP, Mac, Mb_aero[tid], Fw_aero[tid]
        )

    Fw_aero[tid] = temp_Fw_aero
    Mb_aero[tid] = temp_Mb_aero

    # Step 6: Compute thrust forces and moments
    temp_Fb_thrust, temp_Mb_thrust = \
        compute_thrust_forces_and_moments(
            Va[tid], throttle_left_state[tid], throttle_right_state[tid], rho[tid], PP,
            F1_b[tid], F2_b[tid], Fb_thrust[tid], M_prop_left[tid], M_prop_right[tid], Mb_thrust[tid]
        )

    Fb_thrust[tid] = temp_Fb_thrust
    Mb_thrust[tid] = temp_Mb_thrust

    # Step 7: Compute total forces applied to body
    temp_Fb, temp_Mb, temp_Fb_g, temp_Fb_aero = \
        compute_forces_applied_to_body(
            alpha[tid], beta[tid], orientation[tid], Fw_aero[tid], Mb_aero[tid],
            Fb_thrust[tid], Mb_thrust[tid], g, m,
            q_aero[tid], Fb_aero[tid], gravity_vector_body[tid], Fb_g[tid],
            Fb[tid], Mb[tid]
        )

    Fb[tid] = temp_Fb
    Mb[tid] = temp_Mb
    Fb_g[tid] = temp_Fb_g
    Fb_aero[tid] = temp_Fb_aero

    # Step 8: Update state using Euler or RK45 integration based on solver_type
    if solver_type == 0:
        temp_position, temp_linear_vel, temp_angular_vel, temp_orientation = \
            euler_update(
                position[tid], linear_vel[tid], angular_vel[tid], orientation[tid], dt,
                Fb[tid], Mb[tid], m, J
            )
    else:  # RK45
        temp_position, temp_linear_vel, temp_angular_vel, temp_orientation = \
            rk45_update(
                position[tid], linear_vel[tid], angular_vel[tid], orientation[tid], dt,
                Fb[tid], Mb[tid], m, J
            )

    position[tid] = temp_position
    linear_vel[tid] = temp_linear_vel
    angular_vel[tid] = temp_angular_vel
    orientation[tid] = temp_orientation