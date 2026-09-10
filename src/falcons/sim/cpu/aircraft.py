import json
import os
from typing import Dict, Any

import numpy as np

from falcons.sim.cpu.base import AircraftBase
from falcons.sim.cpu.actuators import ActuatorSystem
from falcons.sim.cpu.sensors import SensorSystem
from falcons.sim.cpu.estimators import EstimatorSystem
from falcons.aircraft.config import AircraftConfig
from falcons.sim.aero_contract import check_surface_order

class Aircraft(AircraftBase):
    """
    A generic, config-driven aircraft model simulator.
    """
    
    def __init__(self, aircraft_name: str = None, config_path: str = None, save_history: bool = False):
        """
        Initialize aircraft model
        
        Args:
            aircraft_name: Name of aircraft directory (e.g., 'Airship_V7')
            save_history: Whether to save simulation history
        """
        if aircraft_name is None and config_path is None:
            raise ValueError("Must provide 'aircraft_name'")
        
        # Create config manager
        self.config_manager = AircraftConfig(aircraft_name)
        
        # Load configuration
        self.config = self.config_manager.load()

        # Store parameter groups for easier access
        self.VP = self.config['vehicle_params']
        self.AP = self.config['aero_params']
        self.EP = self.config['environment_params']
        self.IP = self.config['default_initial_state']

        # The action vector's meaning comes from the order of the JSON's aero_surfaces block and
        # is re-checked nowhere else, so it is checked here, once, at construction.
        check_surface_order(self.config, aircraft_name)

        # Initialize base class
        super().__init__(self.config, save_history)    

        # Initialize actions
        self.aero_action: np.ndarray = np.zeros(self.get_aero_action_size())
        self.motor_action: np.ndarray = np.zeros(self.get_motor_action_size())

    def setup_systems(self):
        """Create all systems and handle seeding in one clean method"""
        # Create systems
        self.actuator_system = self.create_actuator_system()
        self.sensor_system = self.create_sensor_system()
        self.estimator_system = self.create_estimator_system()
        
        # Apply config seed if available
        config_seed = self.get_config_seed()
        if config_seed is not None:
            self.seed(config_seed)
    
    def create_actuator_system(self) -> ActuatorSystem:
        """Create actuator system from config"""
        system = ActuatorSystem(self.get_time_step())
        actuator_config = self.VP['actuator_system']
        
        for group_name, configs in actuator_config.items():
            system.add_actuator_group(group_name, configs)
            
        return system
    
    def create_sensor_system(self) -> SensorSystem:
        """Create sensor system from config"""
        system = SensorSystem()
        sensor_config = self.VP['sensor_system']

        if sensor_config:
            for group_name, configs in sensor_config.items():
                system.add_sensor_group(group_name, configs)
        else:
            print("No sensor system configured")

        return system
    
    def create_estimator_system(self):
        """Create estimator system from config"""
        system = EstimatorSystem()
        
        # Get estimator configuration
        estimator_config = self.VP.get('estimator_system')
        
        if estimator_config:
            for group_name, configs in estimator_config.items():
                system.add_estimator_group(group_name, configs)
        else:
            print("No estimator system configured")
        
        return system
    
    def get_aero_action_size(self) -> int:
        """Get aero action size from config"""
        return len(self.VP['actuator_system']['aero_surfaces'])
    
    def get_motor_action_size(self) -> int:
        """Get motor action size from config"""
        return len(self.VP['actuator_system']['motors'])
    
    def get_action_keys(self) -> dict:
        """
        Return a dictionary with separate lists for aerodynamic and motor action keys.
        """
        return {
            "control_surfaces": list(self.VP['actuator_system']['aero_surfaces'].keys()),
            "motors": list(self.VP['actuator_system']['motors'].keys())
        }

    def default_initial_state(self) -> Dict[str, np.ndarray]:
        """Default initial state for airship from config"""
        state_config = self.config.get('default_initial_state', {})
        return {key: np.array(value, dtype=float) for key, value in state_config.items()}
    
    def get_mass(self) -> float:
        """Return aircraft mass from config"""
        return self.VP['mass']
    
    def get_inertia_matrix(self) -> np.ndarray:
        """Return aircraft inertia matrix from config"""
        return np.array(self.VP['inertia_matrix'])
    
    def get_constant_actions(self) -> tuple:
        """Get constant actions for testing based on the configured number of actuators."""
        aero_size = self.get_aero_action_size()
        motor_size = self.get_motor_action_size()
        
        # Zero deflection for all aerodynamic surfaces
        aero_action = np.zeros(aero_size)
        
        # Set all motors to 0% throttle
        motor_action = np.ones(motor_size) * -1.0
        
        return aero_action, motor_action
    
    def get_random_actions(self) -> tuple:
        """Get random actions for testing"""
        aero_action = self.get_random_aero_action()
        motor_action = self.get_random_motor_action()
        return aero_action, motor_action
    
    def save_forces_history(self):
        """Save forces to history"""
        if hasattr(self, 'forces'):
            self.forces['Fb'].append(self.Fb.copy())
            self.forces['Mb'].append(self.Mb.copy())
            self.forces['F_aero'].append(self.Fb_aero.copy())
            self.forces['M_aero'].append(self.Mb_aero.copy())
            self.forces['F_thrust'].append(self.Fb_thrust.copy())
            self.forces['M_thrust'].append(self.Mb_thrust.copy())
            self.forces['Fb_g'].append(self.Fb_g.copy())

    def save_aero_params_history(self):
        """Save aerodynamic parameters to history"""
        if hasattr(self, 'aero_params'):
            self.aero_params['alpha'].append(self.alpha)
            self.aero_params['beta'].append(self.beta)
            self.aero_params['Va'].append(self.Va)
            self.aero_params['height ratio'].append(self.hr)
            self.aero_params['Coef_f'].append(np.array([self.coeffs['CL'], self.coeffs['CD']]))
            self.aero_params['Coef_m'].append(np.array([self.coeffs['Cl'], self.coeffs['Cm'], self.coeffs['Cn']]))
            # free-air vs in-ground-effect pair, straight off the measured height sweep
            self.aero_params['oge_ige'].append(np.array([
                self.coeffs['CL_free'], self.coeffs['CD_free'],
                self.coeffs['CL'], self.coeffs['CD']]))

    def check_termination_conditions(self) -> Dict[str, bool]:
        """
        Check if simulation should terminate based on crash or stall conditions.
        
        Returns:
            Dict containing 'crashed', 'stalled' flags and termination 'reason'
        """
        termination = {
            'crashed': False,
            'stalled': False,
            'reason': None
        }

        # Crash: the CG reaching the ground. This was the WING height before the aerodynamic
        # model changed -- the empirical ground-effect correction needed the wing's offset from
        # the CG, and nothing else did, so `cg_offset_vector` went with it.
        if -self.state['position'][2] < 0:
            termination['crashed'] = True
            termination['reason'] = 'CRASHED'

        # Incidence limit. NOT a stall -- the derivative model has no stall and will keep
        # generating lift past this. It is the edge of the envelope the set was fitted in, and
        # crossing it means the coefficients are extrapolation.
        if hasattr(self, 'alpha') and np.abs(np.degrees(self.alpha)) > self.AP['alpha_max_deg']:
            termination['stalled'] = True
            termination['reason'] = 'ALPHA MAX EXCEEDED'

        return termination