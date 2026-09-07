import warp as wp
import numpy as np
from typing import Tuple
import random

@wp.func
def euler_step_lti_1d(
    x: wp.float32, 
    u: wp.float32, 
    dt: wp.float32, 
    a: wp.float32, 
    b: wp.float32, 
    c: wp.float32, 
    d: wp.float32
) -> wp.vec2f:
    """
    Euler integration step for 1D LTI system
    Returns [x_next, y]
    """
    # State derivative: x_dot = a*x + b*u
    x_dot = a * x + b * u
    
    # Euler integration: x_next = x + dt * x_dot
    x_next = x + dt * x_dot
    
    # Output: y = c*x + d*u
    y = c * x + d * u
    
    return wp.vec2f(x_next, y)

@wp.func
def euler_step_lti_2d(
    x1: wp.float32,
    x2: wp.float32,
    u: wp.float32,
    dt: wp.float32,
    a11: wp.float32, a12: wp.float32,
    a21: wp.float32, a22: wp.float32,
    b1: wp.float32, b2: wp.float32,
    c1: wp.float32, c2: wp.float32,
    d: wp.float32
) -> wp.vec3f:
    """
    Euler integration step for 2D LTI system
    Returns [x1_next, x2_next, y]
    """
    # State derivatives
    x1_dot = a11 * x1 + a12 * x2 + b1 * u
    x2_dot = a21 * x1 + a22 * x2 + b2 * u
    
    # Euler integration
    x1_next = x1 + dt * x1_dot
    x2_next = x2 + dt * x2_dot
    
    # Output
    y = c1 * x1 + c2 * x2 + d * u
    
    return wp.vec3f(x1_next, x2_next, y)

@wp.func
def euler_step_lti_3d(
    x1: wp.float32, x2: wp.float32, x3: wp.float32,
    u: wp.float32,
    dt: wp.float32,
    a11: wp.float32, a12: wp.float32, a13: wp.float32,
    a21: wp.float32, a22: wp.float32, a23: wp.float32,
    a31: wp.float32, a32: wp.float32, a33: wp.float32,
    b1: wp.float32, b2: wp.float32, b3: wp.float32,
    c1: wp.float32, c2: wp.float32, c3: wp.float32,
    d: wp.float32
) -> wp.vec4f:
    """
    Euler integration step for 3D LTI system
    Returns [x1_next, x2_next, x3_next, y]
    """
    # State derivatives
    x1_dot = a11 * x1 + a12 * x2 + a13 * x3 + b1 * u
    x2_dot = a21 * x1 + a22 * x2 + a23 * x3 + b2 * u
    x3_dot = a31 * x1 + a32 * x2 + a33 * x3 + b3 * u
    
    # Euler integration
    x1_next = x1 + dt * x1_dot
    x2_next = x2 + dt * x2_dot
    x3_next = x3 + dt * x3_dot
    
    # Output
    y = c1 * x1 + c2 * x2 + c3 * x3 + d * u
    
    return wp.vec4f(x1_next, x2_next, x3_next, y)

@wp.func
def generate_gaussian_noise(seed: wp.int32, offset: wp.int32) -> wp.float32:
    """Generate Gaussian white noise using Box-Muller transform"""
    state = wp.rand_init(seed, offset)
    u1 = wp.randf(state)
    u2 = wp.randf(state)
    
    # Box-Muller transform
    return wp.sqrt(-2.0 * wp.log(u1)) * wp.cos(2.0 * wp.PI * u2)

@wp.kernel
def update_dryden_turbulence(
    # Current states (input/output)
    u_gust_states: wp.array(dtype=wp.float32),
    v_gust_states: wp.array(dtype=wp.vec2f),
    w_gust_states: wp.array(dtype=wp.vec2f),
    p_gust_states: wp.array(dtype=wp.float32),
    q_gust_states: wp.array(dtype=wp.vec3f),
    r_gust_states: wp.array(dtype=wp.vec3f),
    
    # Aircraft parameters
    V_a: wp.array(dtype=wp.float32),  # Airspeed for each environment

    position: wp.array(dtype=wp.vec3f),    # Position for each environment

    # Model parameters
    dt: wp.float32,
    b: wp.float32,  # Wingspan
    W_20_ms: wp.float32,
    
    # Random seed
    seed: wp.int32,
    
    # Outputs
    linear_gusts: wp.array(dtype=wp.vec3f),
    angular_gusts: wp.array(dtype=wp.vec3f)
):
    """Update Dryden turbulence model for all environments in parallel"""
    tid = wp.tid()

    # Conversion factors
    m_to_ft = 1.0 / 0.3048
    ft_to_m = 0.3048
    
    # Altitude in NED
    h = -position[tid][2]

    h_ft = h * m_to_ft

    # Turbulence intensities (sigma) in m/s
    sigma_w = 0.1 * W_20_ms
    sigma_u = sigma_w / wp.pow(0.177 + 0.000823 * h_ft, 0.4) 
    sigma_v = sigma_u
    
    # Turbulence scale lengths (L) in meters
    if h_ft >= 1000:
        L_u = h_ft * ft_to_m
        L_v = L_u
        L_w = L_u
    else:
        L_u = (h_ft / wp.pow(0.177 + 0.000823 * h_ft, 1.2)) * ft_to_m
        L_v = L_u
        L_w = h

    # Get current airspeed (ensure minimum value)
    Va = wp.max(V_a[tid], 0.1)
    
    # Generate 6 independent Gaussian noise values
    base_seed = seed*2 + 12345
    noise_u = generate_gaussian_noise(base_seed, tid * 6 + 0) / wp.sqrt(dt)
    noise_v = generate_gaussian_noise(base_seed, tid * 6 + 1) / wp.sqrt(dt)
    noise_w = generate_gaussian_noise(base_seed, tid * 6 + 2) / wp.sqrt(dt)
    noise_p = generate_gaussian_noise(base_seed, tid * 6 + 3) / wp.sqrt(dt)
    noise_q = generate_gaussian_noise(base_seed, tid * 6 + 4) / wp.sqrt(dt)
    noise_r = generate_gaussian_noise(base_seed, tid * 6 + 5) / wp.sqrt(dt)
    
    # --- Longitudinal Gust (u_w) ---
    K_u = wp.sqrt(2.0 * L_u / (wp.PI * Va))
    T_u = L_u / Va
    a_u = -1.0 / T_u
    b_u = 1.0
    c_u = sigma_u * K_u / T_u
    d_u = 0.0
    
    result_u = euler_step_lti_1d(u_gust_states[tid], noise_u, dt, a_u, b_u, c_u, d_u)
    u_gust_states[tid] = result_u[0]
    u_w = result_u[1]
    
    # --- Lateral Gust (v_w) ---
    K_v = wp.sqrt(L_v / (wp.PI * Va))
    T_v = L_v / Va
    a11_v = -2.0 / T_v
    a12_v = -1.0 / (T_v * T_v)
    a21_v = 1.0
    a22_v = 0.0
    b1_v = 1.0
    b2_v = 0.0
    c1_v = sigma_v * K_v * wp.sqrt(3.0) * T_v / (T_v * T_v)
    c2_v = sigma_v * K_v / (T_v * T_v)
    d_v = 0.0
    
    v_state = v_gust_states[tid]
    result_v = euler_step_lti_2d(v_state[0], v_state[1], noise_v, dt, 
                                a11_v, a12_v, a21_v, a22_v, b1_v, b2_v, c1_v, c2_v, d_v)
    v_gust_states[tid] = wp.vec2f(result_v[0], result_v[1])
    v_w = result_v[2]
    
    # --- Vertical Gust (w_w) ---
    K_w = wp.sqrt(L_w / (wp.PI * Va))
    T_w = L_w / Va
    a11_w = -2.0 / T_w
    a12_w = -1.0 / (T_w * T_w)
    a21_w = 1.0
    a22_w = 0.0
    b1_w = 1.0
    b2_w = 0.0
    c1_w = sigma_w * K_w * wp.sqrt(3.0) * T_w / (T_w * T_w)
    c2_w = sigma_w * K_w / (T_w * T_w)
    d_w = 0.0
    
    w_state = w_gust_states[tid]
    result_w = euler_step_lti_2d(w_state[0], w_state[1], noise_w, dt,
                                a11_w, a12_w, a21_w, a22_w, b1_w, b2_w, c1_w, c2_w, d_w)
    w_gust_states[tid] = wp.vec2f(result_w[0], result_w[1])
    w_w = result_w[2]
    
    # --- Roll Angular Gust (p_w) ---
    K_p = wp.sqrt(0.8 * wp.pow(wp.PI * L_w / (4.0 * b), 1.0/3.0) / (L_w * Va))
    T_p = 4.0 * b / (wp.PI * Va)
    a_p = -1.0 / T_p
    b_p = 1.0
    c_p = sigma_w * K_p / T_p
    d_p = 0.0
    
    result_p = euler_step_lti_1d(p_gust_states[tid], noise_p, dt, a_p, b_p, c_p, d_p)
    p_gust_states[tid] = result_p[0]
    p_w = result_p[1]
    
    # --- Pitch Angular Gust (q_w) ---
    T_q = 4.0 * b / (wp.PI * Va)
    a_aux = -sigma_w * K_w * wp.sqrt(3.0) * T_w / Va
    b_aux = -sigma_w * K_w / Va
    c_aux = T_q * T_w * T_w
    d_aux = 2.0 * T_q * T_w + T_w * T_w
    e_aux = 2.0 * T_w + T_q
    
    a11_q = -d_aux / c_aux
    a12_q = -e_aux / c_aux
    a13_q = -1.0 / c_aux
    a21_q = 1.0
    a22_q = 0.0
    a23_q = 0.0
    a31_q = 0.0
    a32_q = 1.0
    a33_q = 0.0
    b1_q = 1.0
    b2_q = 0.0
    b3_q = 0.0
    c1_q = a_aux / c_aux
    c2_q = b_aux / c_aux
    c3_q = 0.0
    d_q = 0.0
    
    q_state = q_gust_states[tid]
    result_q = euler_step_lti_3d(q_state[0], q_state[1], q_state[2], noise_q, dt,
                                a11_q, a12_q, a13_q, a21_q, a22_q, a23_q, 
                                a31_q, a32_q, a33_q, b1_q, b2_q, b3_q, 
                                c1_q, c2_q, c3_q, d_q)
    q_gust_states[tid] = wp.vec3f(result_q[0], result_q[1], result_q[2])
    q_w = result_q[3]
    
    # --- Yaw Angular Gust (r_w) ---
    T_r = 3.0 * b / (wp.PI * Va)
    a_aux_r = sigma_v * K_v * wp.sqrt(3.0) * T_v / Va
    b_aux_r = sigma_v * K_v / Va
    c_aux_r = T_r * T_v * T_v
    d_aux_r = 2.0 * T_r * T_v + T_v * T_v
    e_aux_r = 2.0 * T_v + T_r
    
    a11_r = -d_aux_r / c_aux_r
    a12_r = -e_aux_r / c_aux_r
    a13_r = -1.0 / c_aux_r
    a21_r = 1.0
    a22_r = 0.0
    a23_r = 0.0
    a31_r = 0.0
    a32_r = 1.0
    a33_r = 0.0
    b1_r = 1.0
    b2_r = 0.0
    b3_r = 0.0
    c1_r = a_aux_r / c_aux_r
    c2_r = b_aux_r / c_aux_r
    c3_r = 0.0
    d_r = 0.0
    
    r_state = r_gust_states[tid]
    result_r = euler_step_lti_3d(r_state[0], r_state[1], r_state[2], noise_r, dt,
                                a11_r, a12_r, a13_r, a21_r, a22_r, a23_r,
                                a31_r, a32_r, a33_r, b1_r, b2_r, b3_r,
                                c1_r, c2_r, c3_r, d_r)
    r_gust_states[tid] = wp.vec3f(result_r[0], result_r[1], result_r[2])
    r_w = result_r[3]
    
    # Store outputs
    linear_gusts[tid] = wp.vec3f(u_w, v_w, w_w)
    angular_gusts[tid] = wp.vec3f(p_w, q_w, r_w)

class DrydenTurbulenceModelWARP:
    """
    WARP implementation of the Dryden Turbulence Model for parallel simulation
    """
    
    def __init__(self, num_envs: int, device: str, dt: float, b: float, 
                intensity: str = 'light'):
        """
        Initialize WARP Dryden turbulence model
        
        Args:
            num_envs: Number of parallel environments
            device: WARP device ("cuda" or "cpu")
            dt: Simulation time step (s)
            b: Aircraft wingspan (m)
            h: Altitude (m)
            intensity: Turbulence intensity ('light', 'moderate', 'severe')
        """
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.b = b
        self.time_step = 0
        
        # Initialize turbulence parameters
        self._initialize_parameters(intensity)
        
        # Create WARP arrays for states
        self.u_gust_states = wp.zeros((num_envs,), dtype=wp.float32, device=device)
        self.v_gust_states = wp.zeros((num_envs,), dtype=wp.vec2f, device=device)
        self.w_gust_states = wp.zeros((num_envs,), dtype=wp.vec2f, device=device)
        self.p_gust_states = wp.zeros((num_envs,), dtype=wp.float32, device=device)
        self.q_gust_states = wp.zeros((num_envs,), dtype=wp.vec3f, device=device)
        self.r_gust_states = wp.zeros((num_envs,), dtype=wp.vec3f, device=device)
        
        # Output arrays
        self.linear_gusts = wp.zeros((num_envs,), dtype=wp.vec3f, device=device)
        self.angular_gusts = wp.zeros((num_envs,), dtype=wp.vec3f, device=device)
        
        # Initialize random seed
        self.warp_seed = None
        
        print(f"WARP Dryden Turbulence Model Initialized. Intensity: {intensity}")
    
    def _initialize_parameters(self, intensity: str):
        """Set turbulence parameters based on altitude and intensity"""
        # Conversion factors
        knots_to_ms = 1.852 / 3.6  
        
        # Wind speed at 20 ft based on intensity
        W_20_knots = {'very light': 2, 'light': 15, 'moderate': 30, 'severe': 45}[intensity]
        self.W_20_ms = W_20_knots * knots_to_ms
    
    def seed(self, seed=None):
        """Seed the turbulence model"""
        if seed is not None:
            self.warp_seed = wp.int32(seed)
        else:
            random_seed = random.randint(0, 2**29)
            self.warp_seed = wp.int32(random_seed)

    def update_dryden(self, V_a: wp.array, position: wp.array):
        """
        Update turbulence model for all environments
        
        Args:
            V_a: Array of airspeeds for each environment
            
        Returns:
            Tuple of (linear_gusts, angular_gusts) as WARP arrays
        """
        # Use deterministic time step for seeding if warp_seed is set
        current_time_seed = self.warp_seed + self.time_step

        wp.launch(
            kernel=update_dryden_turbulence,
            dim=self.num_envs,
            inputs=[
                self.u_gust_states,
                self.v_gust_states,
                self.w_gust_states,
                self.p_gust_states,
                self.q_gust_states,
                self.r_gust_states,
                V_a,
                position,
                self.dt,
                self.b,
                self.W_20_ms,
                current_time_seed,  
                self.linear_gusts,
                self.angular_gusts
            ],
            device=self.device
        )
        
        self.time_step += 1
        return self.linear_gusts, self.angular_gusts
    
    def reset(self):
        """Reset all turbulence states to zero"""
        self.u_gust_states.zero_()
        self.v_gust_states.zero_()
        self.w_gust_states.zero_()
        self.p_gust_states.zero_()
        self.q_gust_states.zero_()
        self.r_gust_states.zero_()
        self.time_step = 0
    
    def get_gusts_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current gust values as numpy arrays"""
        return self.linear_gusts.numpy(), self.angular_gusts.numpy()