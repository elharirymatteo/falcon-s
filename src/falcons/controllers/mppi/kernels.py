import warp as wp

import os as _os
# 1 = reject numerically-blown-up rollouts with a large finite cost instead of letting their
# inf/NaN poison the softmax (see the longitudinal cost assembly below). Set 0 to reproduce the
# historical unguarded behaviour, in which a single NaN made every weight NaN and froze the plan.
COST_NAN_GUARD = wp.constant(int(_os.environ.get("MPPI_COST_NAN_GUARD", "1")))
from falcons.aircraft.params import WingParametersStruct, AerodynamicsParametersStruct, PropulsionParametersStruct
from falcons.sim.warp.aerodynamics import (
    compute_aerodynamic_parameters,
    compute_all_coeffs,
    compute_lift_and_drag_forces,
    compute_aerodynamic_forces_and_moments,
    compute_thrust_forces_and_moments,
    compute_forces_applied_to_body,
    compute_derivatives,
    euler_update,
)

from falcons.controllers.mppi.cost import *

from falcons.sim.warp.actuators import process_actuator_complete
from falcons.controllers.mppi.reference import compute_reference

@wp.kernel
def _sample_inputs_sequences(number_of_iterations:wp.int32,
                             seed:wp.int32,
                             previous_action: wp.array(dtype=wp.float32), 
                             noise_std_dev: wp.float32,
                             noisy_action: wp.array(dtype=wp.float32), 
                             bias_cost_term:wp.array(dtype=wp.float32),
                             temperature:wp.float32,
):
    tid = wp.tid() 
    state = wp.rand_init(seed, tid)

    noisy_action[tid] = wp.clamp(
            previous_action[tid % number_of_iterations] + noise_std_dev * wp.randn(state),
            -1.0,
            1.0
        )
    
    if noise_std_dev != 0.0:
        # bias_cost = temperature/2.0 * ((2.0*noisy_action[tid] - previous_action[tid]) * 1.0/noise_std_dev * previous_action[tid])
        bias_cost = temperature * (noisy_action[tid] * 1.0/wp.sqrt(noise_std_dev) * previous_action[tid % number_of_iterations])
    else:
        bias_cost = 0.0
    
    wp.atomic_add(bias_cost_term, tid // number_of_iterations, bias_cost)



@wp.kernel
def _convert_inputs_sequences_into_trajectories(number_of_iterations:wp.int32,
                                                position:wp.array(dtype=wp.vec3f),
                                                linear_vel:wp.array(dtype=wp.vec3f),
                                                angular_vel:wp.array(dtype=wp.vec3f),
                                                orientation:wp.array(dtype=wp.quatf),
                                                position_memory:wp.array(dtype=wp.vec3f),
                                                linear_vel_memory:wp.array(dtype=wp.vec3f),
                                                angular_vel_memory:wp.array(dtype=wp.vec3f),
                                                orientation_memory:wp.array(dtype=wp.quatf),
                                                alpha_memory:wp.array(dtype=wp.float32),
                                                beta_memory:wp.array(dtype=wp.float32),
                                                Mac:wp.float32,
                                                rho:wp.array(dtype=wp.float32),
                                                Va:wp.array(dtype=wp.float32),
                                                alpha:wp.array(dtype=wp.float32),
                                                beta:wp.array(dtype=wp.float32),
                                                Q:wp.array(dtype=wp.float32),
                                                hr:wp.array(dtype=wp.float32),
                                                elevator:wp.array(dtype=wp.float32),
                                                aileron:wp.array(dtype=wp.float32),
                                                rudder:wp.array(dtype=wp.float32),
                                                AP: AerodynamicsParametersStruct,
                                                C_L: wp.array(dtype=wp.float32),
                                                C_D: wp.array(dtype=wp.float32),
                                                WP: WingParametersStruct,
                                                D_tot: wp.array(dtype=wp.float32),
                                                L_tot: wp.array(dtype=wp.float32),
                                                ge_enable: bool,
                                                C_Y: wp.array(dtype=wp.float32),
                                                Y_tot: wp.array(dtype=wp.float32),
                                                Cl: wp.array(dtype=wp.float32),
                                                Cm: wp.array(dtype=wp.float32),
                                                Cn: wp.array(dtype=wp.float32),
                                                Mb_aero: wp.array(dtype=wp.vec3f),
                                                Fw_aero: wp.array(dtype=wp.vec3f),
                                                Fb_thrust: wp.array(dtype=wp.vec3f),
                                                Mb_thrust: wp.array(dtype=wp.vec3f),
                                                g: wp.float32,
                                                m: wp.float32,
                                                q_aero: wp.array(dtype=wp.quatf),
                                                Fb_aero: wp.array(dtype=wp.vec3f),
                                                gravity_vector_body: wp.array(dtype=wp.vec3f),
                                                Fb_g: wp.array(dtype=wp.vec3f),
                                                Fb: wp.array(dtype=wp.vec3f),
                                                J:wp.mat33,
                                                Mb: wp.array(dtype=wp.vec3f),
                                                dt:wp.float32,
                                                throttle: wp.array(dtype=wp.float32),
                                                PP: PropulsionParametersStruct,
                                                F1_b: wp.array(dtype=wp.vec3f),
                                                F2_b: wp.array(dtype=wp.vec3f),
                                                M_prop_left: wp.array(dtype=wp.float32),
                                                M_prop_right: wp.array(dtype=wp.float32),
                                                T_s: wp.float32,
                                                zeta: wp.float32,
                                                omega_0: wp.float32,
                                                elevator_limits: wp.vec2,
                                                aileron_limits: wp.vec2,
                                                rudder_limits: wp.vec2,
                                                throttle_limits: wp.vec2,
                                                elevator_state: wp.array(dtype=wp.float32),
                                                aileron_state: wp.array(dtype=wp.float32),
                                                rudder_state: wp.array(dtype=wp.float32),
                                                throttle_left_state: wp.array(dtype=wp.float32),
                                                throttle_right_state: wp.array(dtype=wp.float32),
                                                elevator_dot: wp.array(dtype=wp.float32),
                                                aileron_dot: wp.array(dtype=wp.float32),
                                                rudder_dot: wp.array(dtype=wp.float32),

):
    tid = wp.tid()

    for iter in range(number_of_iterations):
        
        # Step 1: Process actuator dynamics
        temp_elevator, temp_aileron, temp_rudder, temp_throttle_left, temp_throttle_right, temp_elevator_dot, temp_aileron_dot, temp_rudder_dot = \
            process_actuator_complete(elevator[tid * number_of_iterations + iter], aileron[tid * number_of_iterations + iter], rudder[tid * number_of_iterations + iter],
                                      throttle[tid * number_of_iterations + iter], throttle[tid * number_of_iterations + iter], dt, T_s, zeta, omega_0,
                                      elevator_limits, aileron_limits, rudder_limits, throttle_limits, elevator_state[tid], aileron_state[tid], rudder_state[tid],
                                      throttle_left_state[tid], throttle_right_state[tid], elevator_dot[tid], aileron_dot[tid], rudder_dot[tid], wp.int32(0))

        elevator_state[tid] = temp_elevator
        aileron_state[tid] = temp_aileron
        rudder_state[tid] = temp_rudder  
        throttle_left_state[tid] = temp_throttle_left
        throttle_right_state[tid] = temp_throttle_right
        elevator_dot[tid] = temp_elevator_dot
        aileron_dot[tid] = temp_aileron_dot
        rudder_dot[tid] = temp_rudder_dot


        # Step 2: Compute aerodynamic parameters
        temp_alpha, temp_beta, temp_rho, temp_Q, temp_hr, Va_temp, _airspeed_vec = \
            compute_aerodynamic_parameters(
                position[tid], linear_vel[tid], WP, rho[tid], Va[tid],
                alpha[tid], beta[tid], Q[tid], hr[tid],
                wp.vec3f(0.0, 0.0, 0.0), linear_vel[tid]
            )

        alpha[tid] = temp_alpha
        beta[tid] = temp_beta
        rho[tid] = temp_rho
        Q[tid] = temp_Q
        hr[tid] = temp_hr
        Va[tid] = Va_temp

        # Step 3: Reset coefficients to zero
        C_L[tid] = 0.0
        # Step 4: Compute all aerodynamic coefficients. Actuator states are in DEGREES and the
        # derivative set is per radian, so they convert at the call, exactly as in the plant.
        temp_C_D, temp_C_S, temp_C_L, temp_Cl, temp_Cm, temp_Cn, _cdf, _clf = \
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

        # Step 5: Compute lift and drag forces
        temp_D_tot, temp_Y_tot, temp_L_tot = \
            compute_lift_and_drag_forces(
                Q[tid], WP, C_D[tid], C_Y[tid], C_L[tid], D_tot[tid], Y_tot[tid], L_tot[tid]
            )

        D_tot[tid] = temp_D_tot
        Y_tot[tid] = temp_Y_tot
        L_tot[tid] = temp_L_tot

        # Step 6: Compute aerodynamic forces and moments
        temp_Fw_aero, temp_Mb_aero = \
            compute_aerodynamic_forces_and_moments(
                D_tot[tid], Y_tot[tid], L_tot[tid], Cl[tid], Cm[tid], Cn[tid],
                Q[tid], WP, Mac, Mb_aero[tid], Fw_aero[tid]
            )

        Fw_aero[tid] = temp_Fw_aero
        Mb_aero[tid] = temp_Mb_aero

        # Step 7: Compute thrust forces and moments
        temp_Fb_thrust, temp_Mb_thrust = \
            compute_thrust_forces_and_moments(
                Va[tid], throttle_left_state[tid], throttle_right_state[tid], rho[tid], PP,
                F1_b[tid], F2_b[tid], Fb_thrust[tid], M_prop_left[tid], M_prop_right[tid], Mb_thrust[tid]
            )

        Fb_thrust[tid] = temp_Fb_thrust
        Mb_thrust[tid] = temp_Mb_thrust

        # Step 8: Compute total forces applied to body
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

        # Step 9: Update state using Euler integration
        temp_position, temp_linear_vel, temp_angular_vel, temp_orientation = \
            euler_update(
                position[tid], linear_vel[tid], angular_vel[tid], orientation[tid], dt,
                Fb[tid], Mb[tid], m, J
            )

        position[tid] = temp_position
        linear_vel[tid] = temp_linear_vel
        angular_vel[tid] = temp_angular_vel
        orientation[tid] = temp_orientation

        position_memory[tid*number_of_iterations+iter] = position[tid]
        linear_vel_memory[tid*number_of_iterations+iter] = linear_vel[tid]
        angular_vel_memory[tid*number_of_iterations+iter] = angular_vel[tid]
        orientation_memory[tid*number_of_iterations+iter] = orientation[tid]
        alpha_memory[tid*number_of_iterations+iter] = alpha[tid]
        beta_memory[tid*number_of_iterations+iter] = beta[tid]



@wp.kernel
def _evaluate_trajectories(
    number_of_iterations: wp.int32,
    position_memory: wp.array(dtype=wp.vec3f),
    linear_vel_memory: wp.array(dtype=wp.vec3f),
    angular_vel_memory: wp.array(dtype=wp.vec3f),
    orientation_memory: wp.array(dtype=wp.quatf),
    alpha_memory: wp.array(dtype=wp.float32),
    beta_memory: wp.array(dtype=wp.float32),
    costs: wp.array(dtype=wp.float32),
    simulation_time: wp.float32,   
    dt: wp.float32,                
    trajectory_type: wp.int32
):
    tid = wp.tid()

    # Time correction at each iteration step
    current_time = simulation_time + wp.float32(tid % number_of_iterations) * dt
    
    # Get reference values from GPU computation
    ref_values = compute_reference(current_time, trajectory_type)
    ref_pos_y = ref_values[0]
    ref_pos_z = ref_values[1]
    ref_velocity = ref_values[2]
    
    # Get state for this step
    pos = position_memory[tid]
    vel = linear_vel_memory[tid]
    ang_vel = angular_vel_memory[tid]
    ori = orientation_memory[tid]
    alpha = alpha_memory[tid]
    beta = beta_memory[tid]

    # Compute individual costs
    altitude_cost = altitude_tracking_cost(pos, vel, ang_vel, ori, ref_pos_z)
    lateral_cost = lateral_tracking_cost(pos, vel, ang_vel, ori, ref_pos_y)
    ground_collision_cost = ground_collision_cost(pos, vel, ang_vel, ori)  
    ang_vel_cost = ang_vel_tracking_cost(pos, vel, ang_vel, ori)
    vel_cost = vel_tracking_cost(pos, vel, ang_vel, ori, ref_velocity)
    high_altitude_cost = excessive_altitude_cost(pos, vel, ang_vel, ori)
    alpha_cost = alpha_tracking_cost(pos, vel, ang_vel, ori, alpha)
    beta_cost = beta_tracking_cost(pos, vel, ang_vel, ori, beta)
    vz_cost = sink_rate_cost(vel, ori)
    
    # Weighted total cost
    total_cost = (4.0 * altitude_cost +      # Track z
                  10.0 * lateral_cost +      # Track y
                  1.0 * ground_collision_cost +   # Avoid ground (z >= 0)
                  3.0 * ang_vel_cost +    # Stay stable
                  4.5 * vel_cost +        # Maintain target velocity
                  0.0 * high_altitude_cost + # Don't go too high
                  1.0 * alpha_cost +      # Penalize alpha if too high
                  0.05 * beta_cost +       # Penalize beta
                  W_VZ * vz_cost)          # Damp slow altitude divergence (0 = historical)

    # A sampled rollout that blows up numerically yields inf/NaN here, and a single NaN poisons
    # the min-cost reduction -> every weight NaN -> uniform weights -> the "plan" is the mean of
    # zero-mean noise, i.e. the previous sequence held constant while the aircraft falls. This is
    # the same guard the attitude kernel below applies; it is gated so the historical (unguarded)
    # behaviour stays the default until the longitudinal results are regenerated.
    if COST_NAN_GUARD != 0:
        safe_cost = float(1.0e9)
        if total_cost < 1.0e9:
            safe_cost = total_cost
        wp.atomic_add(costs, tid // number_of_iterations, safe_cost)
    else:
        wp.atomic_add(costs, tid // number_of_iterations, total_cost)


@wp.kernel
def _evaluate_trajectories_attitude(
    number_of_iterations: wp.int32,
    position_memory: wp.array(dtype=wp.vec3f),
    linear_vel_memory: wp.array(dtype=wp.vec3f),
    angular_vel_memory: wp.array(dtype=wp.vec3f),
    orientation_memory: wp.array(dtype=wp.quatf),
    alpha_memory: wp.array(dtype=wp.float32),
    beta_memory: wp.array(dtype=wp.float32),
    costs: wp.array(dtype=wp.float32),
    step0: wp.int32,
    phi_ref: wp.array(dtype=wp.float32),
    hdot_ref: wp.array(dtype=wp.float32),
    va_ref: wp.array(dtype=wp.float32),
    alpha_limit: wp.float32,
):
    """Attitude-executor cost: track the (phi*, hdot*, Va*) command STREAM instead of a position
    trajectory, so MPPI solves the same task as the RL attitude executor. The stream is indexed by
    absolute sim step (step0 + horizon offset), clamped at the end of the maneuver."""
    tid = wp.tid()

    idx = wp.min(step0 + (tid % number_of_iterations), phi_ref.shape[0] - 1)

    pos = position_memory[tid]
    vel = linear_vel_memory[tid]
    ang_vel = angular_vel_memory[tid]
    ori = orientation_memory[tid]
    alpha = alpha_memory[tid]
    beta = beta_memory[tid]

    phi_c = phi_ref[idx]
    psidot = 9.81 * wp.tan(phi_c) / wp.max(va_ref[idx], 1.0)   # coordinated-turn rate

    bank_cost = bank_tracking_cost(ori, phi_c)
    climb_cost = climb_rate_tracking_cost(vel, ori, hdot_ref[idx])
    speed_cost = airspeed_tracking_cost(vel, va_ref[idx])
    ground_cost = ground_collision_cost(pos, vel, ang_vel, ori)
    alpha_cost = incidence_guard_cost(alpha, alpha_limit)
    beta_cost = sideslip_cost(beta)
    rate_cost = turn_rate_damping_cost(ang_vel, psidot, phi_c)

    total_cost = (20.0 * bank_cost +      # track phi* [rad]
                  3.0 * climb_cost +      # track hdot* [m/s]
                  1.5 * speed_cost +      # track Va* [m/s]
                  1.0 * ground_cost +     # never hit the water
                  1.0 * alpha_cost +      # never stall
                  0.3 * beta_cost +       # keep the turn coordinated
                  2.0 * rate_cost)        # fly the turn, not a bang-bang rate excursion

    # A sampled rollout that blows up numerically yields inf/NaN here, and a single NaN poisons
    # the min-cost reduction -> every weight NaN -> the plan is garbage from then on. Reject it
    # with a large finite cost instead (NaN fails the comparison, so it takes the else branch).
    safe_cost = float(1.0e9)
    if total_cost < 1.0e9:
        safe_cost = total_cost

    wp.atomic_add(costs, tid // number_of_iterations, safe_cost)


# @wp.kernel
# def _compute_weights(costs:wp.array(dtype=float),
#                     min_cost: wp.array(dtype=float),
#                     weights:wp.array(dtype=float),
#                     temperature: wp.float32,
#                     bias_cost_term:wp.array(dtype=wp.float32), 
#                     eta: wp.array(dtype=float)  # Add eta as a parameter?
# ):
#     tid = wp.tid()
#     wp.atomic_min(min_cost, 0, costs[tid])
#     normalised_cost = costs[tid] - min_cost[0]
#     wp.atomic_add(eta, 0, wp.exp(-normalised_cost/temperature))
#     bias_cost_term[tid] = 0.0
#     weights[tid] = (1.0/eta[0]) * wp.exp(-normalised_cost / temperature)

@wp.kernel
def _compute_min_cost(costs: wp.array(dtype=float),
                      min_cost: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_min(min_cost, 0, costs[tid])

@wp.kernel 
def _compute_eta(costs: wp.array(dtype=float),
                 min_cost: wp.array(dtype=float),
                 bias_cost_term: wp.array(dtype=wp.float32),
                 temperature: wp.float32,
                 eta: wp.array(dtype=float)):
    tid = wp.tid()
    normalized_cost = costs[tid] + bias_cost_term[tid] - min_cost[0]
    wp.atomic_add(eta, 0, wp.exp(-normalized_cost / temperature))

@wp.kernel
def _compute_final_weights(costs: wp.array(dtype=float),
                          min_cost: wp.array(dtype=float),
                          bias_cost_term: wp.array(dtype=wp.float32),
                          temperature: wp.float32,
                          eta: wp.array(dtype=float),
                          weights: wp.array(dtype=float)):
    tid = wp.tid()
    normalized_cost = costs[tid] + bias_cost_term[tid] - min_cost[0]
    if eta[0] > 1e-10:  # Prevent division by zero
        weights[tid] = wp.exp(-normalized_cost / temperature) / eta[0]
    else:
        weights[tid] = 1.0 / wp.float32(costs.shape[0])  # Uniform weights if eta is zero


@wp.kernel
def _compute_weighted_average(number_of_iterations:wp.int32,
                              weights:wp.array(dtype=float),
                              sampled_actions:wp.array(dtype=float),
                              optimal_action:wp.array(dtype=float),
):
    tid = wp.tid()
    wp.atomic_add(optimal_action, tid % number_of_iterations, sampled_actions[tid] * weights[tid // number_of_iterations])




@wp.kernel
def _shift_sequence(u1:wp.array(dtype=float),
                    u2:wp.array(dtype=float),
                    u3:wp.array(dtype=float),
                    u4:wp.array(dtype=float),
):
    tid = wp.tid()

    u1_next = u1[tid + 1]
    u1[tid] = u1_next

    u2_next = u2[tid + 1]
    u2[tid] = u2_next

    u3_next = u3[tid + 1]
    u3[tid] = u3_next

    u4_next = u4[tid + 1]
    u4[tid] = u4_next