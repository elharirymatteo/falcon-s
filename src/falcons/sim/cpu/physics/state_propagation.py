import numpy as np
from scipy.integrate import solve_ivp, odeint
from typing import Dict, Any, Callable
from abc import ABC, abstractmethod
from .quaternion_math import quat_normalize, quat_rotate_vector
from .forces_moments import ForceMomentCalculatorMixin
from .wind_model import WindModelMixin


class PhysicsIntegrator():
    """Handles numerical integration for aircraft state propagation"""
    
    def __init__(self, solver_type: str = 'RK45', atol: float = 1e-10, rtol: float = 1e-8):
        self.solver_type = solver_type
        self.atol = atol
        self.rtol = rtol
        self._validate_solver_type()
    
    def _validate_solver_type(self):
        """Validate that the solver type is supported"""
        valid_solvers = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA', 'euler', 'odeint']
        if self.solver_type not in valid_solvers:
            raise ValueError(f"Unsupported solver type: {self.solver_type}. Valid types: {valid_solvers}")
    
    def integrate(self, derivatives_func: Callable, initial_state: np.ndarray, dt: float) -> np.ndarray:
        """
        Integrate the state forward by dt using the specified solver
        
        Args:
            derivatives_func: Function that computes state derivatives
            initial_state: Initial state vector
            dt: Time step
            
        Returns:
            Final state after integration
        """
        if self.solver_type == 'odeint':
            return self._integrate_odeint(derivatives_func, initial_state, dt)
        elif self.solver_type in ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']:
            return self._integrate_solve_ivp(derivatives_func, initial_state, dt)
        elif self.solver_type == 'euler':
            return self._integrate_euler(derivatives_func, initial_state, dt)
        else:
            raise ValueError(f"Unsupported solver type: {self.solver_type}")
    
    def _integrate_odeint(self, derivatives_func: Callable, x0: np.ndarray, dt: float) -> np.ndarray:
        """Integrate using odeint (order: x, t)"""
        t = np.linspace(0, dt, 2)
        sol = odeint(derivatives_func, x0, t)
        return sol[-1]
    
    def _integrate_solve_ivp(self, derivatives_func: Callable, x0: np.ndarray, dt: float) -> np.ndarray:
        """Integrate using solve_ivp (order: t, x)"""
        sol = solve_ivp(
            fun=lambda t, x: derivatives_func(t, x),
            t_span=(0, dt),
            max_step=dt,
            y0=x0,
            atol=self.atol,
            rtol=self.rtol,
            method=self.solver_type,
            vectorized=False
        )
        return sol.y[:, -1]
    
    def _integrate_euler(self, derivatives_func: Callable, x0: np.ndarray, dt: float) -> np.ndarray:
        """Simple Euler integration"""
        dx = derivatives_func(0, x0)
        return x0 + dx * dt


class RigidBodyDynamics:
    """Implements rigid body dynamics calculations common to all aircraft"""
    
    @staticmethod
    def position_derivative(linear_vel: np.ndarray, q_body: np.ndarray) -> np.ndarray:
        """
        Compute position derivative by rotating velocity from body to inertial frame
        
        Args:
            linear_vel: Linear velocity in body frame [u, v, w]
            q_body: Quaternion [w, x, y, z] representing body orientation
            
        Returns:
            position_dot: Position derivative in inertial frame
        """
        # Position derivative (body to inertial frame transformation)
        position_dot = quat_rotate_vector(q_body, linear_vel, i_to_b=False)
        return position_dot
    
    @staticmethod
    def linear_vel_derivative(
        linear_vel: np.ndarray,
        angular_vel: np.ndarray, 
        total_forces: np.ndarray,
        mass: float,
    ) -> np.ndarray:
        """
        Compute body frame velocity derivatives using Newton-Euler equations
        
        Args:
            linear_vel: Linear velocity in body frame [u, v, w]
            angular_vel: Angular velocity in body frame [p, q, r]
            total_forces: Total forces in body frame [Fx, Fy, Fz]
            mass: Aircraft mass
            
        Returns:
            linear_vel_dot: Linear velocity derivative
        """
        # Linear velocity derivative in body frame
        linear_vel_dot = total_forces / mass - np.cross(angular_vel, linear_vel)
        
        return linear_vel_dot
    
    @staticmethod
    def angular_vel_derivative(
        angular_vel: np.ndarray, 
        total_moments: np.ndarray,
        inertia_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute angular velocity derivative
        
        Args:
            angular_vel: Angular velocity in body frame [p, q, r]
            total_moments: Total moments in body frame [Mx, My, Mz]
            inertia_matrix: Inertia matrix
            
        Returns:
            angular_vel_dot: Angular velocity derivative
        """
        # Angular velocity derivative (Euler's equation)
        inertia_inv = np.linalg.inv(inertia_matrix)
        angular_vel_dot = inertia_inv @ (total_moments - np.cross(angular_vel, inertia_matrix @ angular_vel))
        
        # Safety check for numerical stability
        if np.linalg.norm(angular_vel) > 1e6:
            raise ValueError(f"Angular velocity is too high: {angular_vel}")
        
        return angular_vel_dot
    
    @staticmethod
    def quaternion_derivative(q: np.ndarray, ang_vel: np.ndarray) -> np.ndarray:
        """
        Compute quaternion time derivative
        
        Args:
            q: Quaternion [w, x, y, z]
            angular_vel: Angular velocity [p, q, r] in body frame
            
        Returns:
            Quaternion derivative
        """
        w_matrix = np.array([[0, -ang_vel[0], -ang_vel[1], -ang_vel[2]],
                            [ang_vel[0], 0, ang_vel[2], -ang_vel[1]],
                            [ang_vel[1], -ang_vel[2], 0, ang_vel[0]],
                            [ang_vel[2], ang_vel[1], -ang_vel[0], 0]])
        q_dot = 0.5 * w_matrix @ q
        return q_dot


class AircraftStateManager:
    """Manages aircraft state structure and conversions"""
    
    @staticmethod
    def state_to_vector(state: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Convert state dictionary to vector for integration
        
        Args:
            state: State dictionary with keys: 'position', 'linear_vel', 'angular_vel', 'orientation'
            
        Returns:
            State vector [pos(3), vel(3), omega(3), quat(4)]
        """
        return np.concatenate([
            state['position'],
            state['linear_vel'], 
            state['angular_vel'],
            state['orientation']
        ])
    
    @staticmethod
    def vector_to_state(state_vector: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convert state vector back to dictionary
        
        Args:
            state_vector: State vector [pos(3), vel(3), omega(3), quat(4)]
            
        Returns:
            State dictionary
        """
        # Normalize quaternion for numerical stability
        quaternion = state_vector[9:13]
        quaternion = quat_normalize(quaternion)
        
        return {
            'position': state_vector[:3],
            'linear_vel': state_vector[3:6],
            'angular_vel': state_vector[6:9],
            'orientation': quaternion,
        }


class StatePropagationMixin(ForceMomentCalculatorMixin, WindModelMixin):
    """
    Mixin class providing state propagation functionality to aircraft models.
    This is the central point for applying all physics, including turbulence.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize physics components
        solver_type = self.config.get('environment_params', {}).get('solver_type', 'RK45')
        atol = self.config.get('environment_params', {}).get('integration_atol', 1e-10)
        rtol = self.config.get('environment_params', {}).get('integration_rtol', 1e-8)
        
        self.integrator = PhysicsIntegrator(solver_type, atol, rtol)
        self.rigid_body_dynamics = RigidBodyDynamics()
        self.state_manager = AircraftStateManager()
    
    def propagate_state(self, dt: float):
        """
        Propagate aircraft state forward by dt using physics integration
        
        Args:
            dt: Time step
            
        Returns:
            Updated state dictionary
        """
        # Update the turbulence model for the next step (gets new gust vectors)
        self.update_wind()

        # Update aerodynamic parameters
        self.set_aerodynamic_parameters()
        
        # Compute forces and moments
        self.get_forces_moments_applied_to_body()
        
        # Convert state to vector for integration
        initial_state_vector = self.state_manager.state_to_vector(self.state)
        
        # Create derivatives function with correct signature for the solver
        def derivatives_wrapper(*args):
            return self._compute_derivatives(*args)
        
        # Integrate
        final_state_vector = self.integrator.integrate(
            derivatives_wrapper, 
            initial_state_vector, 
            dt
        )
        
        # Convert back to state dictionary
        self.state = self.state_manager.vector_to_state(final_state_vector)
        
        return self.state
    
    def _compute_derivatives(self, *args):
        """
        Compute derivatives of the state vector for integration, including turbulence effects.
        
        Args depend on solver type:
        - odeint: (state_vector, time)
        - solve_ivp: (time, state_vector) 
        - euler: (time, state_vector)
        """
        # Handle different argument orders for different solvers
        if self.integrator.solver_type == 'odeint':
            state_vector, t = args
        else:  # solve_ivp methods and euler
            t, state_vector = args
        
        # Extract state components
        position = state_vector[:3]
        linear_vel = state_vector[3:6]
        angular_vel = state_vector[6:9]
        orientation = state_vector[9:13]
        
        # Normalize quaternion
        orientation = quat_normalize(orientation)

        # Position derivative (body to inertial frame transformation)
        position_dot = self.rigid_body_dynamics.position_derivative(
            linear_vel, orientation
        )
        
        # Compute derivatives using rigid body dynamics
        linear_vel_dot = self.rigid_body_dynamics.linear_vel_derivative(
            linear_vel=linear_vel,
            angular_vel=angular_vel,
            total_forces=self.Fb,
            mass=self.get_mass()
        )
        
        angular_vel_dot = self.rigid_body_dynamics.angular_vel_derivative(
            angular_vel=angular_vel,
            total_moments=self.Mb,
            inertia_matrix=self.get_inertia_matrix()
        )
        
        # Quaternion derivative
        orientation_dot = self.rigid_body_dynamics.quaternion_derivative(orientation, angular_vel)
        
        return np.hstack([position_dot, linear_vel_dot, angular_vel_dot, orientation_dot])