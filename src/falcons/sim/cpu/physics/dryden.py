import numpy as np
from typing import Tuple

def euler_step_lti(x: np.ndarray, u: float, dt: float, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Performs a single Euler integration step for a continuous-time LTI system.

    Args:
        x (np.ndarray): Current state vector.
        u (float): Current input value.
        dt (float): Time step for the integration.
        A (np.ndarray): System matrix.
        B (np.ndarray): Input matrix.
        C (np.ndarray): Output matrix.
        D (np.ndarray): Feedthrough matrix.

    Returns:
        A tuple containing:
        - x_next (np.ndarray): The state vector at the next time step.
        - y (float): The output at the current time step.
    """
    # State derivative: x_dot = A*x + B*u
    x_dot = A @ x + (B @ np.array([u])).flatten()

    # Euler integration: x_next = x + dt * x_dot
    x_next = x + dt * x_dot

    # Output equation: y = C*x + D*u
    y = (C @ x + (D @ np.array([u]))).item()

    return x_next, y


class DrydenTurbulenceModel:
    """
    Python realization of the continuous Dryden Turbulence Model (MIL-F-8785C),
    adapted for step-by-step simulation.
    
    This model uses digital filters to generate realistic, time-correlated wind gusts
    from white noise, suitable for integration into a real-time physics loop.
    """
    def __init__(self, dt: float, b: float, intensity: str = 'light'):
        """
        Initializes the Dryden turbulence model.
        
        Args:
            dt: Simulation time step (s).
            V_a: Initial aircraft airspeed (m/s).
            h: Initial altitude (m, positive up).
            b: Aircraft wingspan (m).
            intensity: Pre-defined level ('light', 'moderate', 'severe').
        """
        self.dt = dt
        self.b = b # wingspan in meters
        self._initialize_parameters(intensity)
        
        # Initialize internal states for the digital filters
        # u_gust_state is now a 1-element array to be consistent with other states
        self.u_gust_state = np.zeros(1)
        self.v_gust_state = np.zeros(2)
        self.w_gust_state = np.zeros(2)
        self.p_gust_state = np.zeros(1)
        self.q_gust_state = np.zeros(3)
        self.r_gust_state = np.zeros(3)
        
        print(f"Dryden Turbulence Model Initialized. Intensity: {intensity}")

    def _initialize_parameters(self, intensity: str):
        """Set turbulence scale lengths and sigmas based on altitude and intensity."""
        # Conversion factors
        knots_to_ms = 1.852 / 3.6
        
        # Select wind speed at 20 ft based on intensity
        W_20_knots = {'very light': 2, 'light': 15, 'moderate': 30, 'severe': 45}[intensity]
        self.W_20_ms = W_20_knots * knots_to_ms # Wind speed in m/s

    def update(self, V_a: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Steps the model forward using the reusable Euler integration function.
        
        Args:
            V_a: Current aircraft airspeed (m/s).
            h: Current altitude (m).
            
        Returns:
            A tuple containing:
            - linear_gusts (np.ndarray): [u_w, v_w, w_w] in the body frame.
            - angular_gusts (np.ndarray): [p_w, q_w, r_w] in the body frame.
        """
        # Conversion factors
        m_to_ft = 1.0 / 0.3048
        ft_to_m = 0.3048

        # Altitude and airspeed in feet for MIL-SPEC tables
        h_ft = h * m_to_ft

        # Turbulence scale lengths (L) in meters
        # High altitude model
        if h_ft >= 1000:
            self.L_u = h_ft * ft_to_m
            self.L_v = self.L_u
            self.L_w = self.L_u
        # Low altitude model
        else:
            self.L_u = (h_ft / (0.177 + 0.000823 * h_ft)**1.2) * ft_to_m # Correct, h in ft as per the MIL-SPEC
            self.L_v = self.L_u
            self.L_w = h

        # Turbulence intensities (sigma) in m/s
        self.sigma_w = 0.1 * self.W_20_ms
        self.sigma_u = self.sigma_w / (0.177 + 0.000823 * h_ft)**0.4 # Correct, h in ft as per the MIL-SPEC
        self.sigma_v = self.sigma_u    

        # Ensure airspeed is not zero to avoid division errors
        V_a = max(V_a, 0.1)
        
        # Generate Gaussian white noise, scaled correctly for digital simulation
        noise = np.random.randn(6) * (1 / np.sqrt(self.dt)) # np.sqrt(np.pi/self.dt)

        # --- Longitudinal Gust (u_w) ---
        K_u = np.sqrt(2 * self.L_u / (np.pi * V_a))
        T_u = self.L_u / V_a
        A_u = np.array([[-1/T_u]])
        B_u = np.array([[1]])
        C_u = np.array([[self.sigma_u * K_u / T_u]])
        D_u = np.array([[0]])
        self.u_gust_state, u_w = euler_step_lti(self.u_gust_state, noise[0], self.dt, A_u, B_u, C_u, D_u)

        # --- Lateral Gust (v_w) ---
        K_v = np.sqrt(self.L_v / (np.pi * V_a))
        T_v = self.L_v / V_a
        A_v = np.array([[-2/T_v, -1/T_v**2], [1, 0]])
        B_v = np.array([[1], [0]])
        C_v = np.array([[self.sigma_v * K_v * 3**(1/2) * T_v/ T_v**2, self.sigma_v * K_v / T_v**2]])
        D_v = np.array([[0]])
        self.v_gust_state, v_w = euler_step_lti(self.v_gust_state, noise[1], self.dt, A_v, B_v, C_v, D_v)

        # --- Vertical Gust (w_w) ---
        K_w = np.sqrt(self.L_w / (np.pi * V_a))
        T_w = self.L_w / V_a
        A_w = np.array([[-2/T_w, -1/T_w**2], [1, 0]])
        B_w = np.array([[1], [0]])
        C_w = np.array([[self.sigma_w * K_w * 3**(1/2) * T_w/ T_w**2, self.sigma_w * K_w / T_w**2]])
        D_w = np.array([[0]])
        self.w_gust_state, w_w = euler_step_lti(self.w_gust_state, noise[2], self.dt, A_w, B_w, C_w, D_w)

        # --- Longitudinal angular Gust p_w ---
        K_p = np.sqrt(0.8 * (np.pi*self.L_w/(4*self.b))**(1/3) / (self.L_w*V_a))
        T_p = 4*self.b / (np.pi*V_a)
        A_p = np.array([[-1/T_p]])
        B_p = np.array([[1]])
        C_p = np.array([[self.sigma_w * K_p / T_p]])
        D_p = np.array([[0]])
        self.p_gust_state, p_w = euler_step_lti(self.p_gust_state, noise[3], self.dt, A_p, B_p, C_p, D_p)

        # --- Lateral angular Gust q_w ---
        T_q = 4*self.b / (np.pi*V_a)
        a_aux = -self.sigma_w*K_w*3**(1/2)*T_w/V_a
        b_aux = -self.sigma_w * K_w / V_a
        c_aux = T_q*T_w**2
        d_aux = 2*T_q*T_w + T_w**2
        e_aux = 2*T_w + T_q
        A_q = np.array([[-d_aux/c_aux, -e_aux/c_aux, -1/c_aux], [1, 0, 0], [0, 1, 0]])
        B_q = np.array([[1], [0], [0]])
        C_q = np.array([[a_aux/c_aux, b_aux/c_aux, 0]])
        D_q = np.array([[0]])
        self.q_gust_state, q_w = euler_step_lti(self.q_gust_state, noise[4], self.dt, A_q, B_q, C_q, D_q)

        # --- Vertical angular Gust r_w ---
        T_r = 3*self.b / (np.pi*V_a)
        a_aux = self.sigma_v*K_v*3**(1/2)*T_v/V_a
        b_aux = self.sigma_v * K_v / V_a
        c_aux = T_r*T_v**2
        d_aux = 2*T_r*T_v + T_v**2
        e_aux = 2*T_v + T_r
        A_r = np.array([[-d_aux/c_aux, -e_aux/c_aux, -1/c_aux], [1, 0, 0], [0, 1, 0]])
        B_r = np.array([[1], [0], [0]])
        C_r = np.array([[a_aux/c_aux, b_aux/c_aux, 0]])
        D_r = np.array([[0]])
        self.r_gust_state, r_w = euler_step_lti(self.r_gust_state, noise[5], self.dt, A_r, B_r, C_r, D_r)

        linear_gusts = np.array([u_w, v_w, w_w])
        angular_gusts = np.array([p_w, q_w, r_w])
        
        return linear_gusts, angular_gusts