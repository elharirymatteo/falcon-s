import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from .quaternion_math import quat_rotate_vector
from .aerodynamics import AerodynamicsCalculatorMixin

class ForcesAndMomentsBase(ABC):
    """Base class for computing forces and moments on aircraft"""
    
    def __init__(self, vehicle_params, aerodynamic_params, environment_params):
        self.VP = vehicle_params
        self.AP = aerodynamic_params  
        self.EP = environment_params
    
    @abstractmethod
    def compute_aerodynamic_forces_from_coeffs(self, coeffs: Dict[str, float],
                                          aerodynamic_vars: Dict[str, float]) -> np.ndarray:
        """Compute aerodynamic forces in body frame"""
        pass
    
    @abstractmethod
    def compute_aerodynamic_moments_from_coeffs(self, coeffs: Dict[str, float],
                                          aerodynamic_vars: Dict[str, float]) -> np.ndarray:
        """Compute aerodynamic moments in body frame"""
        pass
    
    @abstractmethod
    def compute_propulsion_forces_moments(self, motor_action: np.ndarray,
                                        aerodynamic_vars: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute propulsion forces and moments"""
        pass
    
    def compute_gravity_forces(self, state: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute gravity forces in body frame
        
        Args:
            state: Aircraft state dictionary
            
        Returns:
            Gravity forces in body frame [Fx, Fy, Fz]
        """
        # Gravity vector in inertial frame (NED: positive Z is down)
        gravity_inertial = np.array([0, 0, self.EP.get('g', 9.81)])
        
        # Transform to body frame using quaternion rotation
        gravity_body = quat_rotate_vector(
            state['orientation'], gravity_inertial, i_to_b=True
        )
        
        # Apply mass to get force
        return self.VP['mass'] * gravity_body


class AircraftForcesAndMoments(ForcesAndMomentsBase):
    """Forces and moments computation for airship using modular aerodynamics"""
    
    def __init__(self, vehicle_params, aerodynamic_params, environment_params, 
                 aerodynamics_model):
        super().__init__(vehicle_params, aerodynamic_params, environment_params)
        self.aerodynamics = aerodynamics_model
        
        # Initialize storage for component forces/moments (for history)
        self.F_aero = np.zeros(3)
        self.M_aero = np.zeros(3)
        self.F_thrust = np.zeros(3)
        self.M_thrust = np.zeros(3)
        self.F_gravity = np.zeros(3)
        
        # Motor type handlers
        self._motor_handlers = {
            'electric_motor': self._compute_electric_motor_thrust,
            'piston_engine': self._compute_piston_engine_thrust,
            'jet_engine': self._compute_jet_engine_thrust
        }
    
    def compute_aerodynamic_forces_from_coeffs(self, coeffs: Dict[str, float], 
                                         aerodynamic_vars: Dict[str, float]) -> np.ndarray:
        """
        Compute aerodynamic forces from pre-computed coefficients
        
        Args:
            coeffs: Pre-computed aerodynamic coefficients
            aerodynamic_vars: Dict with Va, alpha, beta, Q
            
        Returns:
            Aerodynamic forces in body frame [Fx, Fy, Fz]
        """
        alpha = aerodynamic_vars['alpha']
        beta = aerodynamic_vars['beta']
        Q = aerodynamic_vars['Q']

        # Compute forces in wind frame
        L = Q * self.VP['wing']['area'] * coeffs['CL']   # Lift
        D = Q * self.VP['wing']['area'] * coeffs['CD']   # Drag  
        Y = Q * self.VP['wing']['area'] * coeffs['CY']   # Side force
        
        # Forces in wind frame: [drag, side_force, lift]
        F_wind = np.array([-D, Y, -L])
        
        # Transform from wind frame to body frame
        R_wind_to_body = np.array([
            [np.cos(alpha) * np.cos(beta), np.cos(alpha) * np.sin(beta), -np.sin(alpha)],  
            [-np.sin(beta), np.cos(beta), 0],
            [np.sin(alpha) * np.cos(beta), np.sin(alpha) * np.sin(beta), np.cos(alpha)]
        ])
        
        self.F_aero = R_wind_to_body @ F_wind
        return self.F_aero
    
    def compute_aerodynamic_moments_from_coeffs(self, coeffs: Dict[str, float],
                                          aerodynamic_vars: Dict[str, float]) -> np.ndarray:
        """
        Compute aerodynamic moments from pre-computed coefficients
        
        Args:
            coeffs: Pre-computed aerodynamic coefficients
            aerodynamic_vars: Dict with Va, alpha, beta, Q
            
        Returns:
            Aerodynamic moments in body frame [Mx, My, Mz]
        """
        Q = aerodynamic_vars['Q']

        # Compute moments
        Mx = coeffs['Cl'] * Q * self.VP['wing']['area'] * self.VP['wing']['span']    # Roll moment
        My = coeffs['Cm'] * Q * self.VP['wing']['area'] * self.VP['wing']['mac']   # Pitch moment
        Mz = coeffs['Cn'] * Q * self.VP['wing']['area'] * self.VP['wing']['span']    # Yaw moment
        
        self.M_aero = np.array([Mx, My, Mz])
        return self.M_aero
    
    # ===========================
    # Motor Type Specific Methods
    # ===========================
    
    def _compute_electric_motor_thrust(self, throttle: float, motor_config: dict, 
                                     aero_vars: Dict[str, float]) -> float:
        """
        Compute thrust for electric motor using momentum theory
        
        Args:
            throttle: Throttle setting [0, 1]
            motor_config: Motor configuration dictionary
            aero_vars: Aerodynamic variables dict
            
        Returns:
            Thrust force in Newtons
        """
        Va = aero_vars['Va']
        rho = aero_vars.get('rho', 1.225)
        prop_data = motor_config['motor_propeller_data']
        max_thrust = motor_config['max_thrust']
        
        # Momentum theory: compute discharge velocity
        Vd = Va + throttle * (prop_data['k_m'] - Va)
        
        # Thrust force from momentum theory
        thrust = 0.5 * rho * prop_data['Sp'] * prop_data['C_p'] * Vd * (Vd - Va)
        
        # Apply maximum thrust limit
        return np.clip(thrust, 0, max_thrust)
    
    def _compute_piston_engine_thrust(self, throttle: float, motor_config: dict, 
                                    aero_vars: Dict[str, float]) -> float:
        """
        Compute thrust for piston engine using power-to-thrust conversion
        
        Args:
            throttle: Throttle setting [0, 1]
            motor_config: Motor configuration dictionary
            aero_vars: Aerodynamic variables dict
            
        Returns:
            Thrust force in Newtons
        """
        Va = aero_vars['Va']
        max_power = motor_config['max_power']  # Watts
        prop_data = motor_config['motor_propeller_data']
        prop_efficiency = prop_data.get('propeller_efficiency', 0.85)
        
        # Available power at current throttle setting
        available_power = throttle * max_power
        
        # Thrust from power: T = (P * η) / V
        # Protect against division by zero at very low airspeeds
        thrust = (available_power * prop_efficiency) / max(Va, 0.1)
        
        # Apply maximum thrust limit if specified
        if 'max_thrust' in motor_config:
            thrust = min(thrust, motor_config['max_thrust'])
        
        return max(thrust, 0.0)  # Ensure non-negative thrust
    
    def _compute_jet_engine_thrust(self, throttle: float, motor_config: dict, 
                                 aero_vars: Dict[str, float]) -> float:
        """
        PLACEHOLDER - FUTURE IMPLEMENTATION
        Compute thrust for jet engine
        
        Args:
            throttle: Throttle setting [0, 1]
            motor_config: Motor configuration dictionary
            aero_vars: Aerodynamic variables dict
            
        Returns:
            Thrust force in Newtons
        """
        return throttle
    
    def _compute_propeller_moment(self, throttle: float, direction: str, 
                                prop_data: dict) -> float:
        """
        Compute propeller moment (torque reaction)
        
        Args:
            throttle: Throttle setting [0, 1]
            direction: 'clockwise' or 'counterclockwise'
            prop_data: Propeller data dictionary
            
        Returns:
            Propeller moment (torque reaction)
        """
        # Propeller moment from torque reaction
        M_prop = -prop_data['k_q'] * (prop_data['k_o'] * throttle)**2
        
        # Reverse sign for counterclockwise rotation
        if direction == 'counterclockwise':
            M_prop = -M_prop
        
        return M_prop
    
    # ===========================
    # Motor Force/Moment Computation
    # ===========================
    
    def _compute_single_motor_force_moment(self, motor_config: dict, throttle: float, 
                                         aero_vars: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute force and moment for a single motor
        
        Args:
            motor_config: Motor configuration dictionary
            throttle: Throttle input [0, 1]
            aero_vars: Aerodynamic variables dictionary
            
        Returns:
            (force_vector, moment_vector) in body frame
        """
        # Validate configuration
        self._validate_motor_config(motor_config)
        
        # Saturate throttle to valid range
        throttle = np.clip(throttle, 0.0, 1.0)
        
        # Extract motor properties
        position = np.array(motor_config['position'])
        direction = motor_config['direction']
        thrust_vector = np.array(motor_config['thrust_vector'])
        thrust_vector = thrust_vector / np.linalg.norm(thrust_vector)  # Normalize
        motor_type = motor_config.get('type', 'electric_motor')
        
        # Compute thrust using appropriate motor model
        if motor_type not in self._motor_handlers:
            raise ValueError(f"Unsupported motor type: {motor_type}")
        
        thrust_magnitude = self._motor_handlers[motor_type](throttle, motor_config, aero_vars)
        
        # Compute propeller moment (if applicable)
        propeller_moment = 0.0
        if 'motor_propeller_data' in motor_config:
            prop_data = motor_config['motor_propeller_data']
            propeller_moment = self._compute_propeller_moment(throttle, direction, prop_data)
        
        # Force vector in body frame
        force_motor = thrust_magnitude * thrust_vector
        
        # Moment vector in body frame
        moment_from_thrust_arm = np.cross(position, force_motor)
        moment_from_propeller = propeller_moment * thrust_vector
        total_moment = moment_from_thrust_arm + moment_from_propeller

        return force_motor, total_moment

    def compute_propulsion_forces_moments(self, motor_action: np.ndarray,
                                        aerodynamic_vars: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute total propulsion forces and moments from all motors
        
        Args:
            motor_action: Array of throttle commands [0, 1] for each motor
            aerodynamic_vars: Aerodynamic variables dictionary
            
        Returns:
            (total_force, total_moment) in body frame
        """
        # Initialize total forces and moments
        self.F_thrust = np.zeros(3)
        self.M_thrust = np.zeros(3)
        
        # Get motor configurations
        motor_configs = self.VP.get('actuator_system', {}).get('motors', {})
        
        # Early return if no motors configured
        if not motor_configs:
            return self.F_thrust, self.M_thrust
        
        # Validate input size
        if len(motor_action) != len(motor_configs):
            raise ValueError(
                f"Motor action size ({len(motor_action)}) doesn't match "
                f"configured motors ({len(motor_configs)})"
            )
        
        # Compute forces and moments for each motor
        for i, (motor_name, motor_config) in enumerate(motor_configs.items()):
            try:
                throttle = motor_action[i]
                force, moment = self._compute_single_motor_force_moment(
                    motor_config, throttle, aerodynamic_vars
                )
                
                self.F_thrust += force
                self.M_thrust += moment
                
            except Exception as e:
                raise RuntimeError(f"Error computing forces for motor '{motor_name}': {e}")
        
        return self.F_thrust, self.M_thrust
    
    # ===========================
    # Validation Methods
    # ===========================
    
    def _validate_motor_config(self, motor_config: dict) -> None:
        """
        Validate motor configuration has required fields
        
        Args:
            motor_config: Motor configuration dictionary
            
        Raises:
            ValueError: If required fields are missing
        """
        required_fields = ['position', 'thrust_vector', 'direction']
        motor_type = motor_config.get('type', 'electric_motor')
        
        # Check basic required fields
        for field in required_fields:
            if field not in motor_config:
                raise ValueError(f"Motor config missing required field: '{field}'")
        
        # Type-specific validation
        if motor_type in ['electric_motor', 'piston_engine']:
            if 'motor_propeller_data' not in motor_config:
                raise ValueError(f"Motor type '{motor_type}' requires 'motor_propeller_data'")
        
        if motor_type == 'electric_motor':
            if 'max_thrust' not in motor_config:
                raise ValueError("Electric motor requires 'max_thrust'")
        
        if motor_type == 'piston_engine':
            if 'max_power' not in motor_config:
                raise ValueError("Piston engine requires 'max_power'")
        
        if motor_type == 'jet_engine':
            if 'max_thrust' not in motor_config:
                raise ValueError("Jet engine requires 'max_thrust'")
        
        # Validate direction
        valid_directions = ['clockwise', 'counterclockwise']
        if motor_config['direction'] not in valid_directions:
            raise ValueError(f"Invalid direction: {motor_config['direction']}. Must be one of {valid_directions}")


class ForceMomentCalculatorMixin(AerodynamicsCalculatorMixin):
    """Mixin to add force/moment computation to aircraft classes"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize force/moment calculator after aerodynamics is initialized
        self.force_moment_calculator = None
    
    def _initialize_force_moment_calculator(self):
        """Initialize force/moment calculator"""
        if self.force_moment_calculator is None:
            self.force_moment_calculator = AircraftForcesAndMoments(
                self.VP, self.AP, self.EP, self.aerodynamics
            )
    
    def get_forces_moments_applied_to_body(self):
        """
        Compute total forces and moments applied to aircraft body
        
        This method updates self.Fb and self.Mb with total forces and moments
        """
        # Initialize calculator if needed
        self._initialize_force_moment_calculator()
        
        # Get aerodynamic parameters
        aero_vars = self.get_aerodynamic_vars()
        
        # Compute and store aerodynamic coefficients ONCE on main aircraft object
        self.compute_aerodynamic_coeffs(self.alpha, self.beta, self.aero_action, self.state)

        # Compute aerodynamic forces and moments using pre-computed coefficients
        self.Fb_aero = self.force_moment_calculator.compute_aerodynamic_forces_from_coeffs(
            self.coeffs, aero_vars
        )
        self.Mb_aero = self.force_moment_calculator.compute_aerodynamic_moments_from_coeffs(
            self.coeffs, aero_vars
        )
        
        # Compute propulsion forces and moments
        self.Fb_thrust, self.Mb_thrust = self.force_moment_calculator.compute_propulsion_forces_moments(
            self.motor_action, aero_vars
        )
        
        # Compute gravity forces
        self.Fb_g = self.force_moment_calculator.compute_gravity_forces(self.state)
        
        # Total forces and moments
        self.Fb = self.Fb_aero + self.Fb_thrust + self.Fb_g
        self.Mb = self.Mb_aero + self.Mb_thrust
        
        # Save history if enabled
        if hasattr(self, 'save_history') and self.save_history:
            self.save_forces_history()
            self.save_aero_params_history()