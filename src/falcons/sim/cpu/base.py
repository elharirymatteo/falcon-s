from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from falcons.sim.cpu.physics.state_propagation import StatePropagationMixin

class AircraftBase(StatePropagationMixin, ABC):
    """Abstract base class for all aircraft models - pure dynamics"""

    def __init__(self, config: Dict[str, Any], save_history: bool = False):
        self.config = config
        self.save_history = save_history
        
        # Common state structure for all aircraft
        self.state: Dict[str, np.ndarray] = {}
        
        # Forces and moments in body frame
        self.Fb = np.zeros(3, dtype=float)  # total forces
        self.Mb = np.zeros(3, dtype=float)  # total moments
        
        # Integration solver configuration
        self.solver_type: str = 'RK45'
        self.integration_atol: float = 1e-10
        self.integration_rtol: float = 1e-8
        
        # Initialize mixins (call parent constructors)
        super().__init__()

        # Initialize actuators, sensors and estimators
        self.setup_systems()
        
        # Initialize history storage if needed
        if self.save_history:
            self._initialize_history()
    
    # ==================== Abstract Methods - Must be implemented by subclasses ====================
    
    @abstractmethod
    def get_aero_action_size(self) -> int:
        """Return the size of the aerodynamic action vector for this aircraft"""
        pass
    
    @abstractmethod
    def get_motor_action_size(self) -> int:
        """Return the size of the motor action vector for this aircraft"""
        pass
    
    @abstractmethod
    def default_initial_state(self) -> Dict[str, np.ndarray]:
        """Return default initial state for this aircraft"""
        pass
    
    @abstractmethod
    def get_mass(self) -> float:
        """Return aircraft mass"""
        pass
    
    @abstractmethod 
    def get_inertia_matrix(self) -> np.ndarray:
        """Return aircraft inertia matrix"""
        pass
    
    @abstractmethod
    def create_actuator_system(self):
        """Create actuator system specific to this aircraft"""
        pass

    @abstractmethod
    def create_sensor_system(self):
        """Create sensor system specific to this aircraft"""
        pass

    @abstractmethod
    def create_estimator_system(self):
        """Create estimator system specific to this aircraft"""
        pass

    @abstractmethod
    def save_forces_history(self):
        """Save forces to history - implement in subclasses with aircraft-specific force components"""
        pass

    @abstractmethod 
    def save_aero_params_history(self):
        """Save aerodynamic parameters to history - implement in subclasses with aircraft-specific parameters"""
        pass
    
    @abstractmethod
    def setup_systems(self):
        """Create and configure all systems (actuators, sensors, estimators)"""
        pass
    
    # ==================== Template Methods - Same for all aircraft ====================
    
    def reset(self, init_state=None, init_aero_action=None, init_motor_action=None):
        """Reset aircraft to initial conditions"""
        self.state = init_state if init_state is not None else self.default_initial_state()
        self.aero_action = init_aero_action if init_aero_action is not None else np.zeros(self.get_aero_action_size(), dtype=float)
        self.motor_action = init_motor_action if init_motor_action is not None else np.zeros(self.get_motor_action_size(), dtype=float)
        
        # Reset time
        self.current_time = 0.0

        # Reset actuator system
        if hasattr(self, 'actuator_system'):
            self.actuator_system.reset_all()
        
        # Clear history if saving
        if self.save_history:
            self._initialize_history()
        
        return self.state
    
    def step(self, aero_action: np.ndarray, motor_action: np.ndarray):
        """Step the aircraft simulation forward one time step"""
        self._validate_aero_action(aero_action)
        self._validate_motor_action(motor_action)
        
        self.aero_action = self.actuator_system.apply_dynamics('aero_surfaces', aero_action)
        self.motor_action = self.actuator_system.apply_dynamics('motors', motor_action)
        
        dt = self.get_time_step()
        self.propagate_state(dt)

        # Update current time
        self.current_time += dt
        
        return self.state
    
    def get_config_seed(self) -> Optional[int]:
        """Extract seed from configuration"""
        env_params = self.config.get('environment_params', {})
        return env_params.get('seed')
    
    def seed(self, seed=None):
        """Seed random number generator for both actions and sensors"""
        if seed is not None:
            # Seed all random sources
            np.random.seed(seed)  # Global NumPy seed
            self.np_random = np.random.RandomState(seed)
            
            # Seed sensor system if it exists
            if hasattr(self, 'sensor_system') and self.sensor_system is not None:
                self.sensor_system.seed(seed + 1000)  # Offset for sensors
                
            # Seed estimator system if it exists
            if hasattr(self, 'estimator_system') and self.estimator_system is not None:
                self.estimator_system.seed(seed + 2000)  # Offset for estimators
        else:
            self.np_random = np.random.RandomState()
            if hasattr(self, 'sensor_system') and self.sensor_system is not None:
                self.sensor_system.seed()
    
    def get_random_aero_action(self) -> np.ndarray:
        """Get a random aerodynamic action within valid bounds"""
        if not hasattr(self, 'np_random') or self.np_random is None:
            self.seed()
        return self.np_random.uniform(-1, 1, self.get_aero_action_size())
    
    def get_random_motor_action(self) -> np.ndarray:
        """Get a random motor action within valid bounds"""
        if not hasattr(self, 'np_random') or self.np_random is None:
            self.seed()
        return self.np_random.uniform(-1, 1, self.get_motor_action_size())
    
    def get_time_step(self) -> float:
        """Get time step from configuration"""
        env_params = self.config.get('environment_params', {})
        return env_params.get('dt', 0.01)
    
    # ==================== Protected/Private Methods ====================
    
    def _validate_aero_action(self, action: np.ndarray) -> None:
        """Validate aerodynamic action input"""
        expected_size = self.get_aero_action_size()
        if len(action) != expected_size:
            raise ValueError(f"Aerodynamic action must have shape ({expected_size},) but got {action.shape}")
        if np.any(np.abs(action) > 1):
            raise ValueError(f"Aerodynamic action values must be between -1 and 1 but got {action}")
    
    def _validate_motor_action(self, action: np.ndarray) -> None:
        """Validate motor action input"""
        expected_size = self.get_motor_action_size()
        if len(action) != expected_size:
            raise ValueError(f"Motor action must have shape ({expected_size},) but got {action.shape}")
        if np.any(np.abs(action) > 1):
            raise ValueError(f"Motor action values must be between -1 and 1 but got {action}")
    
    def _initialize_history(self) -> None:
        """Initialize history storage"""
        self.forces = {
            'Fb': [], 'Mb': [], 'F_aero': [], 'M_aero': [], 
            'F_thrust': [], 'M_thrust': [], 'Fb_g': []
        }
        self.aero_params = {
            'alpha': [], 'beta': [], 'Va': [], 'height ratio': [], 
            'Coef_f': [], 'Coef_m': [], 'oge_ige': []
        }