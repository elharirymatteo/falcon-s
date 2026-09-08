from falcons.aircraft.params import WingParametersStruct, AerodynamicsParametersStruct, PropulsionParametersStruct
import warp as wp

@wp.func
def compute_mu_l(taper_ratio: wp.float32,
                 aspect_ratio: wp.float32,
                 span: wp.float32,
                 h_: wp.float32,
                 cg_z_offset: wp.float32) -> wp.float32:
    h = h_ + cg_z_offset
    return 1.0 + (1.0 - 2.25 * (wp.pow(taper_ratio, 0.00273) - 0.997) *\
                (wp.pow(aspect_ratio, 0.717) + 13.6)) *\
                (288.0 * wp.pow(wp.abs(h/span), 0.787) *\
                wp.exp(-9.14 * (wp.pow(wp.abs(h/span), 0.327)))) / (wp.pow(aspect_ratio, 0.882))

@wp.func
def compute_mu_d(taper_ratio: wp.float32,
                 aspect_ratio: wp.float32,
                 span: wp.float32,
                 h_: wp.float32,
                 cg_z_offset: wp.float32) -> wp.float32:
    h = h_ + cg_z_offset
    term1 = 1.0 - (1.0 - 0.157 * wp.max(0.0, (wp.pow(taper_ratio, 0.775) - 0.373)) *\
                wp.max(0.0, (wp.pow(aspect_ratio, 0.417) - 1.27))) *\
                wp.exp(-4.74 * wp.max(0.0, wp.pow(abs(h/span), 0.814)))
    term2 = wp.abs(h/span) * wp.abs(h/span) * wp.exp(-3.88 * wp.max(0.0, wp.pow(wp.abs(h/span), 0.758)))
    return term1 - term2

@wp.func
def compute_lift_and_drag_forces(Q: wp.float32,
                             WP: WingParametersStruct,
                             C_D: wp.float32,
                             C_Y: wp.float32,
                             C_L: wp.float32,
                             D_tot: wp.float32,
                             Y_tot: wp.float32,
                             L_tot: wp.float32,
                             in_ground_effect: bool,
                             position: wp.vec3f) -> None:

    D_tot = Q * WP.area * C_D
    Y_tot = Q * WP.area * C_Y  
    L_tot = Q * WP.area * C_L
    C_D_ige = C_D
    C_L_ige = C_L

    if in_ground_effect:
        mu_d = compute_mu_d(WP.taper_ratio, WP.aspect_ratio, WP.span, -position[2], WP.cg_z_offset)
        mu_l = compute_mu_l(WP.taper_ratio, WP.aspect_ratio, WP.span, -position[2], WP.cg_z_offset)
        D_tot *= mu_d*wp.pow(mu_l, 2.0)
        L_tot *= mu_l
        C_D_ige *= mu_d*wp.pow(mu_l, 2.0)
        C_L_ige *= mu_l

    return D_tot, Y_tot, L_tot, C_D_ige, C_L_ige

@wp.func
def get_air_density(altitude: wp.float32) -> wp.float32:
    """Get air density as a function of altitude (ISA model up to 11km)"""
    # ISO 2533:1975 standard atmosphere parameters
    T_0 = 288.15                          # Sea level standard temperature [K]
    p_0 = 101325.0                        # Sea level standard pressure [Pa]
    L = 0.0065                            # Temperature lapse rate [K/m]
    g = 9.80665                           # Gravity used in ISA [m/s^2]
    R = 287.05287                         # Specific gas constant for dry air [J/(kg·K)]
    
    # Clamp altitude to >= 0 and <= 11000
    h = wp.clamp(altitude, 0.0, 11000.0)
    
    # Troposphere calculation
    T = T_0 - L * h
    p = p_0 * wp.pow(T / T_0, g / (R * L))
    rho = p / (R * T)
    
    return rho

@wp.func
def compute_aerodynamic_parameters(position: wp.vec3f,
                                   linear_vel: wp.vec3f,
                                   WP: WingParametersStruct,
                                   rho: wp.float32,
                                   Va: wp.float32,
                                   alpha: wp.float32,
                                   beta: wp.float32,
                                   Q: wp.float32,
                                   hr: wp.float32,
                                   wind_linear_gusts: wp.vec3f,
                                   airspeed_vector: wp.vec3f) -> None:
    
    # airspeed_vector = aircraft_velocity - wind_gusts (in body frame)
    airspeed_vector = linear_vel - wind_linear_gusts
    # Air speed
    Va = wp.length(airspeed_vector)
    # Altitude in NED
    h = -position[2]
    # Attack angle
    alpha = wp.atan2(airspeed_vector[2], airspeed_vector[0])
    # Slide angle
    beta = wp.asin(airspeed_vector[1] / (Va + 1e-6))
    # Get air density based on altitude
    rho = get_air_density(h)
    # Air pressure
    Q = 0.5 * rho * Va * Va
    # Height ratio
    hr = -h / WP.span

    return alpha, beta, rho, Q, hr, Va, airspeed_vector


@wp.func
def compute_per_engine_thrust_force(Va: wp.float32,
                                    action: wp.float32,
                                    k_m: wp.float32,
                                    rho: wp.float32,
                                    Sp: wp.float32,
                                    C_p: wp.float32) -> wp.float32:
    Vd = Va + action * (k_m - Va)
    return 0.5 * rho * Sp * C_p * Vd * (Vd - Va)

@wp.func
def compute_per_engine_propeller_moment(action: wp.float32, k_q: wp.float32, k_o: wp.float32) -> wp.float32:
    return -k_q * (k_o * action) * (k_o * action)

@wp.func
def compute_thrust_forces_and_moments(Va: wp.float32,
                                     action_t1: wp.float32,
                                     action_t2: wp.float32,
                                     rho: wp.float32,
                                     PP: PropulsionParametersStruct,
                                     F1_b: wp.vec3f,
                                     F2_b: wp.vec3f,
                                     Fb_thrust: wp.vec3f,
                                     M_prop_left: wp.float32,
                                     M_prop_right: wp.float32,
                                     Mb_thrust: wp.vec3f,) -> None:
    
     

    # Compute thrust force by each engine
    Tp_left = compute_per_engine_thrust_force(Va, action_t1, PP.k_m, rho, PP.Sp, PP.C_p)
    Tp_right = compute_per_engine_thrust_force(Va, action_t2, PP.k_m, rho, PP.Sp, PP.C_p)

    thrust_direction = wp.normalize(PP.tvi)  # Ensure it's normalized
    
    F1_b = Tp_left * thrust_direction
    F2_b = Tp_right * thrust_direction

    Fb_thrust = F1_b + F2_b

    # Compute moment generated by each engine
    M_prop_left = compute_per_engine_propeller_moment(action_t1, PP.k_q, PP.k_o)
    M_prop_right = -compute_per_engine_propeller_moment(action_t2, PP.k_q, PP.k_o)

    Mb_thrust = wp.cross(PP.td1, F1_b) + wp.cross(PP.td2, F2_b) + wp.vec3f(M_prop_left, 0.0, 0.0) + wp.vec3f(M_prop_right, 0.0, 0.0)

    return Fb_thrust, Mb_thrust

@wp.func
def compute_aerodynamic_forces_and_moments(D_tot: wp.float32,
                                           Y_tot: wp.float32,
                                           L_tot: wp.float32,
                                           Cl: wp.float32,
                                           Cm: wp.float32,
                                           Cn: wp.float32,
                                           Q: wp.float32,
                                           WP: WingParametersStruct,
                                           Mac: wp.float32,
                                           angular_vel: wp.vec3f,
                                           Va: wp.float32,
                                           Clp: wp.float32,
                                           Cmq: wp.float32,
                                           Cnr: wp.float32,
                                           Mb_aero: wp.vec3f,
                                           Fw_aero: wp.vec3f) -> None:

    Fw_aero[0] = -D_tot
    Fw_aero[1] = Y_tot
    Fw_aero[2] = -L_tot

    Mb_aero[0] = Cl * Q * WP.area * WP.span
    Mb_aero[1] = Cm * Q * WP.area * Mac
    Mb_aero[2] = Cn * Q * WP.area * WP.span

    # rotary damping (the poly model has none): nondim rate * derivative, added to the body moments.
    # standard form L_damp = Clp*(p b/2V)*Q S b, M_damp = Cmq*(q c/2V)*Q S c, N_damp = Cnr*(r b/2V)*Q S b
    Vc = wp.max(Va, 1.0)
    Mb_aero[0] += Clp * (angular_vel[0] * WP.span / (2.0 * Vc)) * Q * WP.area * WP.span
    Mb_aero[1] += Cmq * (angular_vel[1] * Mac / (2.0 * Vc)) * Q * WP.area * Mac
    Mb_aero[2] += Cnr * (angular_vel[2] * WP.span / (2.0 * Vc)) * Q * WP.area * WP.span

    return Fw_aero, Mb_aero

@wp.func
def compute_forces_applied_to_body(alpha: wp.float32,
                               beta: wp.float32,
                               orientation: wp.quatf,
                               Fw_aero: wp.vec3f,
                               Mb_aero: wp.vec3f,
                               Fb_thrust: wp.vec3f,
                               Mb_thrust: wp.vec3f,
                               g: wp.float32,
                               m: wp.float32,
                               q_aero: wp.quatf,
                               Fb_aero: wp.vec3f,
                               gravity_vector_body: wp.vec3f,
                               Fb_g: wp.vec3f,
                               Fb: wp.vec3f,
                               Mb: wp.vec3f,) -> None:
     
    # -beta, not +beta: the inverse of quat_rpy(0, alpha, beta) rotates the wind-axis force as
    # though the sideslip were the other way round, which would have drag push along the sideslip
    # instead of against it. See the CPU plant's R_wind_to_body for the long version.
    q_aero = wp.quat_rpy(0.0, alpha, -beta)
    Fb_aero = wp.quat_rotate_inv(q_aero, Fw_aero)
    
    gravity_vector = wp.vec3f(0.0, 0.0, g)
    gravity_vector_body = wp.quat_rotate_inv(orientation, gravity_vector)
    Fb_g = m * gravity_vector_body

    Fb = Fb_aero + Fb_g + Fb_thrust 
    Mb = Mb_aero + Mb_thrust

    return Fb, Mb, Fb_g, Fb_aero


@wp.func
def mat3_vec3_mul(J: wp.mat33, v: wp.vec3f) -> wp.vec3f:
    return wp.vec3f(
        J[0,0]*v[0] + J[0,1]*v[1] + J[0,2]*v[2],
        J[1,0]*v[0] + J[1,1]*v[1] + J[1,2]*v[2],
        J[2,0]*v[0] + J[2,1]*v[1] + J[2,2]*v[2]
    )


@wp.func
def compute_derivatives(linear_vel: wp.vec3f,
                        angular_vel: wp.vec3f,
                        orientation:wp.quatf,
                        Fb: wp.vec3f,
                        m: wp.float32,
                        J:wp.mat33,
                        Mb: wp.vec3f)-> None:
     
    v = linear_vel
    w = angular_vel
    q = orientation
    f_b = Fb
    m_b = Mb

    # Normalize quaternion
    q = wp.normalize(q)

    # dot_vb = -w x v + Fb / m
    dot_vb = f_b / m - wp.cross(w, v)

    # dot_wb = inv(J) * (Mb - w x (J * w))
    invJ = wp.inverse(J)

    Jw = mat3_vec3_mul(J, w)

    torque_term = m_b - wp.cross(w, Jw)
    dot_wb = mat3_vec3_mul(invJ, torque_term)

    dot_pos = wp.quat_rotate(q, v)
    dot_orientation = 0.5 * wp.mul(q, wp.quatf(w[0], w[1], w[2], 0.0))   # body-rate form

    return dot_pos, dot_vb, dot_wb, dot_orientation



@wp.func
def euler_update(position: wp.vec3f,
                 linear_vel: wp.vec3f,
                 angular_vel: wp.vec3f,
                 orientation: wp.quatf,
                 dt:wp.float32,
                 Fb: wp.vec3f,
                 Mb: wp.vec3f,
                 m: wp.float32,
                 J: wp.mat33)-> None:

    temp_dot_pos, temp_dot_vb, temp_dot_wb, temp_dot_orientation = \
        compute_derivatives(
            linear_vel, angular_vel, orientation, Fb, m, J, Mb,
        )

    dot_pos = temp_dot_pos
    dot_vb = temp_dot_vb
    dot_wb = temp_dot_wb
    dot_orientation = temp_dot_orientation

    position += dt * dot_pos
    linear_vel += dt * dot_vb
    angular_vel += dt * dot_wb
    orientation += dt * dot_orientation

    return position, linear_vel, angular_vel, orientation

@wp.func
def rk45_update(
                position: wp.vec3f,
                linear_vel: wp.vec3f,
                angular_vel: wp.vec3f,
                orientation: wp.quatf,
                dt: wp.float32,
                Fb: wp.vec3f,
                Mb: wp.vec3f,
                m: wp.float32,
                J: wp.mat33,
) -> None:
    """
    RK45 (Dormand-Prince) integration for aircraft dynamics.
    State: [position, linear_vel, angular_vel, orientation]
    """
    # RK45 Dormand-Prince coefficients
    a21 = 1.0/5.0
    a31, a32 = 3.0/40.0, 9.0/40.0
    a41, a42, a43 = 44.0/45.0, -56.0/15.0, 32.0/9.0
    a51, a52, a53, a54 = 19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0
    a61, a62, a63, a64, a65 = 9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0
    
    # 5th-order weights
    b1, b3, b4, b5, b6 = 35.0/384.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0
    
    # State aliases
    p0 = position
    v0 = linear_vel
    w0 = angular_vel
    q0 = wp.normalize(orientation)
    
    # --- Stage 1 ---
    dp1, dv1, dw1, dq1 = compute_derivatives(v0, w0, q0, Fb, m, J, Mb)
    k1_p = dt * dp1
    k1_v = dt * dv1
    k1_w = dt * dw1
    k1_q = dt * dq1
    
    # --- Stage 2 ---
    # p2 = p0 + a21 * k1_p
    v2 = v0 + a21 * k1_v
    w2 = w0 + a21 * k1_w
    q2 = wp.normalize(q0 + a21 * k1_q)
    
    dp2, dv2, dw2, dq2 = compute_derivatives(v2, w2, q2, Fb, m, J, Mb)
    # k2_p = dt * dp2
    k2_v = dt * dv2
    k2_w = dt * dw2
    k2_q = dt * dq2
    
    # --- Stage 3 ---
    # p3 = p0 + a31 * k1_p + a32 * k2_p
    v3 = v0 + a31 * k1_v + a32 * k2_v
    w3 = w0 + a31 * k1_w + a32 * k2_w
    q3 = wp.normalize(q0 + a31 * k1_q + a32 * k2_q)
    
    dp3, dv3, dw3, dq3 = compute_derivatives(v3, w3, q3, Fb, m, J, Mb)
    k3_p = dt * dp3
    k3_v = dt * dv3
    k3_w = dt * dw3
    k3_q = dt * dq3
    
    # --- Stage 4 ---
    # p4 = p0 + a41 * k1_p + a42 * k2_p + a43 * k3_p
    v4 = v0 + a41 * k1_v + a42 * k2_v + a43 * k3_v
    w4 = w0 + a41 * k1_w + a42 * k2_w + a43 * k3_w
    q4 = wp.normalize(q0 + a41 * k1_q + a42 * k2_q + a43 * k3_q)
    
    dp4, dv4, dw4, dq4 = compute_derivatives(v4, w4, q4, Fb, m, J, Mb)
    k4_p = dt * dp4
    k4_v = dt * dv4
    k4_w = dt * dw4
    k4_q = dt * dq4
    
    # --- Stage 5 ---
    # p5 = p0 + a51 * k1_p + a52 * k2_p + a53 * k3_p + a54 * k4_p
    v5 = v0 + a51 * k1_v + a52 * k2_v + a53 * k3_v + a54 * k4_v
    w5 = w0 + a51 * k1_w + a52 * k2_w + a53 * k3_w + a54 * k4_w
    q5 = wp.normalize(q0 + a51 * k1_q + a52 * k2_q + a53 * k3_q + a54 * k4_q)
    
    dp5, dv5, dw5, dq5 = compute_derivatives(v5, w5, q5, Fb, m, J, Mb)
    k5_p = dt * dp5
    k5_v = dt * dv5
    k5_w = dt * dw5
    k5_q = dt * dq5
    
    # --- Stage 6 ---
    # p6 = p0 + a61 * k1_p + a62 * k2_p + a63 * k3_p + a64 * k4_p + a65 * k5_p
    v6 = v0 + a61 * k1_v + a62 * k2_v + a63 * k3_v + a64 * k4_v + a65 * k5_v
    w6 = w0 + a61 * k1_w + a62 * k2_w + a63 * k3_w + a64 * k4_w + a65 * k5_w
    q6 = wp.normalize(q0 + a61 * k1_q + a62 * k2_q + a63 * k3_q + a64 * k4_q + a65 * k5_q)
    
    dp6, dv6, dw6, dq6 = compute_derivatives(v6, w6, q6, Fb, m, J, Mb)
    k6_p = dt * dp6
    k6_v = dt * dv6
    k6_w = dt * dw6
    k6_q = dt * dq6
    
    # --- Final 5th-order solution ---
    position = p0 + b1*k1_p + b3*k3_p + b4*k4_p + b5*k5_p + b6*k6_p
    linear_vel = v0 + b1*k1_v + b3*k3_v + b4*k4_v + b5*k5_v + b6*k6_v
    angular_vel = w0 + b1*k1_w + b3*k3_w + b4*k4_w + b5*k5_w + b6*k6_w
    orientation = wp.normalize(q0 + b1*k1_q + b3*k3_q + b4*k4_q + b5*k5_q + b6*k6_q)
    
    return position, linear_vel, angular_vel, orientation


@wp.func
def compute_all_coeffs(alpha: wp.float32,
                        beta: wp.float32,
                        elevator: wp.float32,
                        aileron: wp.float32,
                        rudder: wp.float32,
                        alpha_exp: wp.array(dtype=wp.float32),
                        beta_exp: wp.array(dtype=wp.float32),
                        elevator_exp: wp.array(dtype=wp.float32),
                        aileron_exp: wp.array(dtype=wp.float32),
                        rudder_exp: wp.array(dtype=wp.float32),
                        CD_coefs: wp.array(dtype=wp.float32),
                        CL_coefs: wp.array(dtype=wp.float32),
                        CY_coefs: wp.array(dtype=wp.float32),
                        CMx_coefs: wp.array(dtype=wp.float32),
                        CMy_coefs: wp.array(dtype=wp.float32),
                        CMz_coefs: wp.array(dtype=wp.float32),
                        C_L: wp.float32,
                        C_D: wp.float32,
                        C_Y: wp.float32,
                        Cl: wp.float32,
                        Cm: wp.float32,
                        Cn: wp.float32,
                    )-> None:
     
    
    for i in range(30):
        coef = wp.pow(alpha * 180.0 / wp.PI, alpha_exp[i]) * wp.pow(beta * 180.0 / wp.PI, beta_exp[i]) * wp.pow(elevator, elevator_exp[i]) * wp.pow(aileron, aileron_exp[i]) * wp.pow(rudder, rudder_exp[i])
        C_D += CD_coefs[i] * coef 
        C_Y += CY_coefs[i] * coef 
        C_L += CL_coefs[i] * coef 
        Cl += CMx_coefs[i] * coef 
        Cm += CMy_coefs[i] * coef 
        Cn += CMz_coefs[i] * coef 
    return C_D, C_Y, C_L, Cl, Cm, Cn