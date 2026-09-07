"""CPU actuator models: limits and the abstract base, the concrete servo / throttle actuators,
then the type registry and the per-aircraft ActuatorSystem that the CPU plant builds from JSON.
The warp twin of this file is falcons/sim/warp/actuators.py."""
import numpy as np
import dataclasses
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from scipy.integrate import solve_ivp

@dataclasses.dataclass
class ActuatorLimits:
    """Base class for actuator limits"""
    min_value: float
    max_value: float
    
    def clip(self, value: float) -> float:
        """Clip value to limits"""
        return np.clip(value, self.min_value, self.max_value)
    
    def scale_from_normalized(self, normalized_value: float) -> float:
        """Scale from normalized [-1, 1] to actual limits"""
        return (normalized_value + 1) / 2 * (self.max_value - self.min_value) + self.min_value

@dataclasses.dataclass
class ActuatorConfig:
    """Configuration for a single actuator"""
    limits: ActuatorLimits
    dynamics_params: Dict[str, Any]
    initial_state: Optional[np.ndarray] = None
    name: str = "unnamed_actuator"

class BaseActuator(ABC):
    """Abstract base class for all actuators"""
    
    def __init__(self, config: ActuatorConfig, dt: float):
        self.config = config
        self.dt = dt
        self.name = config.name
        self.limits = config.limits
        
        # Initialize internal state
        self.state = self._initialize_state()
        
        # Dynamics parameters
        self.dynamics_params = config.dynamics_params
        
    @abstractmethod
    def _initialize_state(self) -> np.ndarray:
        """Initialize internal state for this actuator type"""
        pass
    
    @abstractmethod
    def _compute_dynamics(self, commanded_input: float) -> np.ndarray:
        """Compute actuator dynamics for one time step"""
        pass
    
    @abstractmethod
    def get_output(self) -> float:
        """Get current actuator output"""
        pass
    
    def apply_dynamics(self, commanded_input: float) -> float:
        """
        Apply actuator dynamics to commanded input
        
        Args:
            commanded_input: Commanded input (normalized [-1, 1])
            
        Returns:
            Actual actuator output (clipped to limits)
        """
        # Update internal state
        self.state = self._compute_dynamics(commanded_input)
        
        # Get output and apply limits
        output = self.get_output()
        return self.limits.clip(output)
    
    def reset(self):
        """Reset actuator to initial state"""
        self.state = self._initialize_state()
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get information about current actuator state"""
        return {
            'name': self.name,
            'state': self.state.copy(),
            'output': self.get_output(),
            'limits': self.limits
        }

class ActuatorGroup:
    """Group of actuators that can be controlled together"""
    
    def __init__(self, actuators: List[BaseActuator], group_name: str = "actuator_group"):
        self.actuators = actuators
        self.group_name = group_name
        self.n_actuators = len(actuators)
    
    def apply_dynamics(self, commanded_inputs: np.ndarray) -> np.ndarray:
        """
        Apply dynamics to all actuators in the group
        
        Args:
            commanded_inputs: Array of commanded inputs for each actuator (normalized [-1, 1])
            
        Returns:
            Array of actual actuator outputs
        """
        if len(commanded_inputs) != self.n_actuators:
            raise ValueError(f"Expected {self.n_actuators} inputs, got {len(commanded_inputs)}")
        
        outputs = np.zeros(self.n_actuators)
        for i, actuator in enumerate(self.actuators):
            outputs[i] = actuator.apply_dynamics(commanded_inputs[i])
        
        return outputs
    
    def reset(self):
        """Reset all actuators in the group"""
        for actuator in self.actuators:
            actuator.reset()
    
    def get_group_info(self) -> Dict[str, Any]:
        """Get information about all actuators in the group"""
        return {
            'group_name': self.group_name,
            'n_actuators': self.n_actuators,
            'actuators': [actuator.get_state_info() for actuator in self.actuators]
        }

class ODEIntegrator:
    """Utility class for ODE integration in actuator dynamics"""
    
    @staticmethod
    def solve_ode(dynamics_func, initial_state, dt, commanded_input, 
                  method='RK45', rtol=1e-6, atol=1e-9):
        """
        Solve ODE using scipy's solve_ivp
        
        Args:
            dynamics_func: Function defining the ODE dynamics
            initial_state: Initial state
            dt: Time step
            commanded_input: Commanded input
            method: Integration method
            rtol: Relative tolerance
            atol: Absolute tolerance
            
        Returns:
            Final state after time step dt
        """
        result = solve_ivp(
            fun=dynamics_func,
            t_span=[0, dt],
            y0=initial_state,
            args=(commanded_input,),
            method=method,
            rtol=rtol,
            atol=atol,
            t_eval=[dt]
        )
        
        return result.y[:, -1]

class ServoActuator(BaseActuator):
    """Servo actuator with second-order dynamics - equivalent to actuators_custom"""
    
    def __init__(self, config: ActuatorConfig, dt: float):
        super().__init__(config, dt)
        
        # Servo parameters (matching actuators_custom defaults)
        self.omega_0 = self.dynamics_params.get('omega_0', 10.0)  # Natural frequency (rad/s)
        self.zeta = self.dynamics_params.get('zeta', 1/np.sqrt(2))  # Damping ratio
        self.use_rk45_integration = self.dynamics_params.get('use_rk45_integration', True)
        
    def _initialize_state(self) -> np.ndarray:
        """Initialize servo state: [position, velocity]"""
        if self.config.initial_state is not None:
            return self.config.initial_state.copy()
        return np.array([0.0, 0.0])  # Start at neutral position with zero velocity
    
    def _compute_dynamics(self, commanded_input: float) -> np.ndarray:
        """Compute servo dynamics"""
        # Scale commanded input to actual deflection
        commanded_deflection = self.limits.scale_from_normalized(commanded_input)
        
        # Compute dynamics
        if self.use_rk45_integration:
            return self._compute_dynamics_ode(commanded_deflection)
        else:
            return self._compute_dynamics_euler(commanded_deflection)
    
    def _compute_dynamics_euler(self, commanded_deflection: float) -> np.ndarray:
        """Compute dynamics using discrete-time approximation"""
        # Current state
        position, velocity = self.state
        
        # Second-order system: x'' + 2*zeta*omega_0*x' + omega_0^2*x = omega_0^2*u
        acceleration = self.omega_0**2 * (commanded_deflection - position) - 2 * self.zeta * self.omega_0 * velocity
        
        # Euler integration
        new_velocity = velocity + acceleration * self.dt
        new_position = position + new_velocity * self.dt
        
        return np.array([new_position, new_velocity])
    
    def _compute_dynamics_ode(self, commanded_deflection: float) -> np.ndarray:
        """Compute servo dynamics with ODE integration (matching actuators_custom)"""
        def dynamics(t, state, u):
            """Second-order servo dynamics"""
            y, dy = state
            ddy = self.omega_0**2 * (u - y) - 2 * self.zeta * self.omega_0 * dy
            return [dy, ddy]
        
        return ODEIntegrator.solve_ode(
            dynamics, self.state, self.dt, commanded_deflection
        )
    
    def get_output(self) -> float:
        """Get current servo deflection"""
        return self.state[0]
    
    def get_servo_info(self) -> Dict[str, Any]:
        """Get detailed servo information"""
        info = self.get_state_info()
        info.update({
            'omega_0': self.omega_0,
            'zeta': self.zeta,
            'use_rk45_integration': self.use_rk45_integration
        })
        return info

def create_servo(surface_type: str, dt: float, **kwargs) -> BaseActuator:
    """
    Factory function to create servo actuators
    
    Args:
        surface_type: Type of control surface ('elevator', 'aileron', 'rudder', 'custom')
        dt: Time step
        **kwargs: Additional parameters
        
    Returns:
        Servo actuator instance
    """
    # Default parameters for different surface types (matching actuators_custom)
    surface_defaults = {
        'elevator': {
            'min_deflection': -20.0, 
            'max_deflection': 20.0, 
            'omega_0': 10.0,
            'zeta': 1/np.sqrt(2)
        },
        'aileron': {
            'min_deflection': -15.0, 
            'max_deflection': 15.0, 
            'omega_0': 10.0,
            'zeta': 1/np.sqrt(2)
        },
        'rudder': {
            'min_deflection': -15.0, 
            'max_deflection': 15.0, 
            'omega_0': 10.0,
            'zeta': 1/np.sqrt(2)
        },
        'custom': {
            'min_deflection': -15.0, 
            'max_deflection': 15.0, 
            'omega_0': 10.0,
            'zeta': 1/np.sqrt(2)
        }
    }
    
    # Get defaults for this surface type
    default_params = surface_defaults.get(surface_type, surface_defaults['custom'])
    default_params.update({
        'use_rk45_integration': True,
        'name': f'{surface_type}_servo'
    })
    default_params.update(kwargs)
    
    # Create limits and config
    limits = ActuatorLimits(
        min_value=default_params['min_deflection'], 
        max_value=default_params['max_deflection']
    )
    config = ActuatorConfig(
        limits=limits,
        dynamics_params=default_params,
        name=default_params['name']
    )
    
    # Create servo
    return ServoActuator(config, dt)

class ThrottleActuator(BaseActuator):
    """Throttle actuator with first-order dynamics - equivalent to actuators_custom.

    Electric motors and piston engines are the same lag; they differ only in the time constant
    (T = 0.2 s electric, T = 3 s piston), so `create_electric_motor` and `create_piston_engine`
    below are this class with a different default T.
    """

    def __init__(self, config: ActuatorConfig, dt: float, T: float = 0.2):
        super().__init__(config, dt)

        # Actuator parameters (matching actuators_custom defaults)
        self.T = self.dynamics_params.get('T', T)  # Time constant for throttle dynamics
        self.use_rk45_integration = self.dynamics_params.get('use_rk45_integration', True)

    def _initialize_state(self) -> np.ndarray:
        """Initialize actuator state (current thrust value)"""
        if self.config.initial_state is not None:
            return self.config.initial_state.copy()
        return np.array([0.0])  # Start with zero thrust
    
    def _compute_dynamics(self, commanded_input: float) -> np.ndarray:
        """Compute first-order throttle dynamics"""
        if self.use_rk45_integration:
            return self._compute_dynamics_ode(commanded_input)
        else:
            return self._compute_dynamics_euler(commanded_input)
    
    def _compute_dynamics_euler(self, commanded_input: float) -> np.ndarray:
        """Compute dynamics using discrete-time approximation (matching actuators_custom)"""
        # Scale commanded input to actual thrust units
        commanded_thrust = self.limits.scale_from_normalized(commanded_input)
        
        # First-order exponential response (matching actuators_custom)
        current_thrust = self.state[0]
        new_thrust = current_thrust + (self.dt / self.T) * (commanded_thrust - current_thrust)
        
        return np.array([new_thrust])
    
    def _compute_dynamics_ode(self, commanded_input: float) -> np.ndarray:
        """Compute dynamics using ODE integration (matching actuators_custom)"""
        # Scale commanded input to actual thrust units
        commanded_thrust = self.limits.scale_from_normalized(commanded_input)
        
        def dynamics(t, state, u):
            """First-order dynamics: T * dy/dt + y = u (matching actuators_custom)"""
            dydt = (u - state[0]) / self.T
            return [dydt]
        
        return ODEIntegrator.solve_ode(
            dynamics, self.state, self.dt, commanded_thrust
        )
    
    def get_output(self) -> float:
        """Get current thrust output"""
        return self.state[0]

    def get_actuator_info(self) -> Dict[str, Any]:
        """Get detailed throttle actuator information"""
        info = self.get_state_info()
        info.update({
            'T': self.T,
            'use_rk45_integration': self.use_rk45_integration
        })
        return info

def _create_throttle_actuator(dt: float, T: float, name: str, /, **kwargs) -> BaseActuator:
    """Build a ThrottleActuator with the caller's default time constant and name.
    T and name are positional-only: the JSON config supplies 'T' and 'name' through kwargs.
    """
    # Default parameters (matching actuators_custom)
    default_params = {
        'T': T,  # Time constant for throttle dynamics
        'use_rk45_integration': True,
        'min_throttle': 0.0,
        'max_throttle': 1.0,
        'name': name
    }
    default_params.update(kwargs)
    
    # Create limits and config
    limits = ActuatorLimits(
        min_value=default_params['min_throttle'], 
        max_value=default_params['max_throttle']
    )
    config = ActuatorConfig(
        limits=limits,
        dynamics_params=default_params,
        name=default_params['name']
    )
    
    return ThrottleActuator(config, dt, T)

def create_electric_motor(dt: float, **kwargs) -> BaseActuator:
    """
    Factory function to create electric motor actuators
    
    Args:
        dt: Time step
        **kwargs: Additional parameters
        
    Returns:
        Electric motor actuator instance
    """
    return _create_throttle_actuator(dt, 0.2, 'electric_motor', **kwargs)

def create_piston_engine(dt: float, **kwargs) -> BaseActuator:
    """
    Factory function to create piston engine actuators

    Args:
        dt: Time step
        **kwargs: Additional parameters
        
    Returns:
        Piston engine actuator instance
    """
    return _create_throttle_actuator(dt, 3, 'piston_engine', **kwargs)

# Factory registry
_ACTUATOR_FACTORIES = {
    'electric_motor': create_electric_motor,
    'piston_engine': create_piston_engine,
    'control_surface': create_servo,
}

def register_actuator_factory(actuator_type: str, factory_func):
    """Register a new actuator factory function"""
    _ACTUATOR_FACTORIES[actuator_type] = factory_func

def create_actuator(actuator_type: str, dt: float, **kwargs) -> BaseActuator:
    """
    Factory function to create actuators
    
    Args:
        actuator_type: Type of actuator to create
        dt: Time step
        **kwargs: Actuator configuration parameters
        
    Returns:
        Created actuator instance
    """
    if actuator_type not in _ACTUATOR_FACTORIES:
        raise ValueError(f"Unknown actuator type: {actuator_type}. Available types: {list(_ACTUATOR_FACTORIES.keys())}")
    
    factory_func = _ACTUATOR_FACTORIES[actuator_type]
    
    # Handle special case for control surfaces
    if actuator_type == 'control_surface':
        surface_type = kwargs.pop('surface_type', 'custom')
        return factory_func(surface_type, dt, **kwargs)
    else:
        return factory_func(dt, **kwargs)

def get_available_actuator_types() -> list:
    """Get list of available actuator types"""
    return list(_ACTUATOR_FACTORIES.keys())

class ActuatorSystem:
    """Complete actuator system for an aircraft"""
    
    def __init__(self, dt: float):
        self.dt = dt
        self.actuator_groups: Dict[str, ActuatorGroup] = {}
        self.all_actuators: List = []
    
    def register_actuator_factory(self, actuator_type: str, factory_func):
        """Register a new actuator factory function"""
        register_actuator_factory(actuator_type, factory_func)
    
    def add_actuator_group(self, group_name: str, actuator_configs: Dict[str, Dict[str, Any]]):
        """
        Add a group of actuators to the system
        
        Args:
            group_name: Name of the actuator group (e.g., 'aero_surfaces', 'motors')
            actuator_configs: Dictionary of actuator configurations
        """
        actuators = []
        
        for name, config in actuator_configs.items():
            actuator_type = config.get('type')
            if not actuator_type:
                raise ValueError(f"Actuator '{name}' missing required 'type' field")
            
            # Clean config and add name
            actuator_config = {k: v for k, v in config.items() if k != 'type'}
            actuator_config['name'] = name
            
            # Create actuator using factory
            actuator = create_actuator(actuator_type, self.dt, **actuator_config)
            
            actuators.append(actuator)
            self.all_actuators.append(actuator)
        
        self.actuator_groups[group_name] = ActuatorGroup(actuators, group_name)
    
    def apply_dynamics(self, group_name: str, commanded_inputs: np.ndarray) -> np.ndarray:
        """Apply dynamics to a specific actuator group"""
        if group_name not in self.actuator_groups:
            raise ValueError(f"Unknown actuator group: {group_name}")
        
        return self.actuator_groups[group_name].apply_dynamics(commanded_inputs)
    
    def reset_all(self):
        """Reset all actuators"""
        for actuator in self.all_actuators:
            actuator.reset()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the entire actuator system"""
        return {
            'groups': {name: group.get_group_info() for name, group in self.actuator_groups.items()},
            'total_actuators': len(self.all_actuators)
        }
