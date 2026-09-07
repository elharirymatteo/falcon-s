"""CPU sensor models: limits and the abstract base, each concrete sensor with its factory,
then the per-aircraft SensorSystem that the CPU plant builds from JSON.
The warp twin of this file is falcons/sim/warp/sensors.py."""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class SensorLimits:
    """Sensor limits configuration"""
    min_value: float = -np.inf
    max_value: float = np.inf
    resolution: Optional[float] = None  # Quantization resolution

class SensorBase(ABC):
    """Base class for all sensors"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.noise_std = config.get('noise_std', 0.0)
        
        # Handle bias as scalar or vector
        bias_config = config.get('bias', 0.0)
        if isinstance(bias_config, (list, tuple)):
            self.bias = np.array(bias_config)
        else:
            self.bias = bias_config  # Scalar
    
        self.scale_factor = config.get('scale_factor', 1.0)
        self.limits = SensorLimits(
            min_value=config.get('min_value', -np.inf),
            max_value=config.get('max_value', np.inf),
            resolution=config.get('resolution', None)
        )
        self.enable_noise = config.get('enable_noise', True)
        
        # Sampling rate and delay settings
        self.sampling_rate = config.get('sampling_rate', 100.0)  # Hz, default 100Hz
        self.delay = config.get('delay', 0.0)  # seconds, default no delay
        
        # Internal state for sampling and delay
        self.sampling_period = 1.0 / self.sampling_rate
        self.last_sample_time = 0.0
        self.last_measurement = None
        
        # Delay buffer for storing measurements with timestamps
        self.delay_buffer = []
        self.delay_timestamps = []
        
        self.np_random = None
        self.current_time = 0.0
    
    def seed(self, seed: Optional[int] = None):
        """Seed the random number generator"""
        self.np_random = np.random.RandomState(seed)
    
    def update_time(self, current_time: float):
        """Update the current simulation time"""
        self.current_time = current_time
    
    def should_sample(self) -> bool:
        """Check if it's time to take a new sample based on sampling rate"""
        # Always sample at t=0
        if self.current_time == 0.0:
            return True

        # Check if enough time has passed since the last sample
        time_since_last = self.current_time - self.last_sample_time
        return time_since_last >= (self.sampling_period - 1e-9)  # Small tolerance for floating point
    
    def _add_to_delay_buffer(self, measurement: np.ndarray, timestamp: float):
        """Add measurement to delay buffer"""
        self.delay_buffer.append(measurement.copy())
        self.delay_timestamps.append(timestamp)

    def _get_measurement_at_time(self, target_time: float) -> np.ndarray:
        """Get the measurement closest to target_time from delay buffer"""
        if not self.delay_buffer or not self.delay_timestamps:
            return self.last_measurement
        
        # Find the measurement with timestamp closest to target_time
        best_index = 0
        best_diff = abs(self.delay_timestamps[0] - target_time)
        
        for i, timestamp in enumerate(self.delay_timestamps):
            diff = abs(timestamp - target_time)
            if diff < best_diff:
                best_diff = diff
                best_index = i
        
        return self.delay_buffer[best_index]
    
    @abstractmethod
    def sense(self, true_value: np.ndarray) -> np.ndarray:
        """Apply sensor model to true value"""
        pass
    
    def sense_with_timing(self, true_value: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply sensor model with sampling rate and delay considerations
        Returns None if no new measurement is available
        """
        # Check if it's time to sample
        if not self.should_sample():
            # Return last measurement if no new sample
            return self.last_measurement
        
        # Take new measurement
        new_measurement = self.sense(true_value)
        self.last_sample_time = self.current_time
        
        if self.delay > 0:
            # Add to delay buffer with current timestamp
            self._add_to_delay_buffer(new_measurement, self.current_time)
            
            # Output the measurement that was taken 'delay' seconds ago
            delayed_time = self.current_time - self.delay
            
            if delayed_time <= 0:
                # Not enough time has passed, output the FIRST measurement (t=0 value)
                if self.delay_buffer:
                    self.last_measurement = self.delay_buffer[0]  # First measurement at t=0
                else:
                    self.last_measurement = new_measurement
                return self.last_measurement
            else:
                # Find the measurement closest to delayed_time
                delayed_measurement = self._get_measurement_at_time(delayed_time)
                self.last_measurement = delayed_measurement
                return delayed_measurement
        else:
            # No delay, return measurement immediately
            self.last_measurement = new_measurement
            return new_measurement

    def _apply_noise(self, value: np.ndarray) -> np.ndarray:
        """Apply Gaussian white noise"""
        if not self.enable_noise or self.noise_std == 0:
            return value
            
        if self.np_random is None:
            self.seed()
            
        noise = self.np_random.normal(0, self.noise_std, value.shape)
        return value + noise
    
    def _apply_bias_and_scale(self, value: np.ndarray) -> np.ndarray:
        """Apply bias and scale factor"""
        # Handle both scalar and vector bias
        if isinstance(self.bias, np.ndarray):
            # Vector bias - ensure shapes match
            if self.bias.shape != value.shape:
                raise ValueError(f"Bias shape {self.bias.shape} doesn't match value shape {value.shape}")
            return self.scale_factor * value + self.bias
        else:
            # Scalar bias
            return self.scale_factor * value + self.bias
    
    def _apply_limits(self, value: np.ndarray) -> np.ndarray:
        """Apply sensor limits and quantization"""
        # Clamp to limits
        value = np.clip(value, self.limits.min_value, self.limits.max_value)
        
        # Apply quantization if specified
        if self.limits.resolution is not None:
            value = np.round(value / self.limits.resolution) * self.limits.resolution
            
        return value
    
    def reset(self):
        """Reset sensor state"""
        self.last_sample_time = 0.0
        self.last_measurement = None
        self.current_time = 0.0
        self.delay_buffer.clear()
        self.delay_timestamps.clear()

class SensorGroup:
    """Group of sensors of the same type"""
    
    def __init__(self, group_name: str):
        self.group_name = group_name
        self.sensors: List[SensorBase] = []
    
    def add_sensor(self, sensor: SensorBase):
        """Add a sensor to this group"""
        self.sensors.append(sensor)
    
    def update_time(self, current_time: float):
        """Update time for all sensors in the group"""
        for sensor in self.sensors:
            sensor.update_time(current_time)
    
    def sense_all(self, true_values: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply all sensors in this group (basic version without timing)"""
        measured_values = {}
        for sensor in self.sensors:
            if sensor.name in true_values:
                measured_values[sensor.name] = sensor.sense(true_values[sensor.name])
        return measured_values
    
    def sense_all_with_timing(self, true_values: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply all sensors in this group with timing considerations"""
        measured_values = {}
        for sensor in self.sensors:
            if sensor.name in true_values:
                measurement = sensor.sense_with_timing(true_values[sensor.name])
                if measurement is not None:
                    measured_values[sensor.name] = measurement
        return measured_values
    
    def seed(self, seed: Optional[int] = None):
        """Seed all sensors in this group"""
        for i, sensor in enumerate(self.sensors):
            sensor_seed = None if seed is None else seed + i
            sensor.seed(sensor_seed)
    
    def reset(self):
        """Reset all sensors in this group"""
        for sensor in self.sensors:
            sensor.reset()
    
    def get_group_info(self) -> Dict[str, Any]:
        """Get information about this sensor group"""
        return {
            'group_name': self.group_name,
            'num_sensors': len(self.sensors),
            'sensor_names': [sensor.name for sensor in self.sensors],
            'sampling_rates': [sensor.sampling_rate for sensor in self.sensors],
            'delays': [sensor.delay for sensor in self.sensors]
        }

class PositionSensor(SensorBase):
    """GPS/Position sensor with configurable noise"""
    
    def sense(self, true_position: np.ndarray) -> np.ndarray:
        """Apply position sensor model"""
        measured = self._apply_bias_and_scale(true_position)
        measured = self._apply_noise(measured)
        measured = self._apply_limits(measured)
        return measured

def create_position_sensor(name: str, **kwargs) -> PositionSensor:
    """Factory function to create position sensors"""
    default_config = {
        'noise_std': 0.1,
        'bias': 0.0,
        'scale_factor': 1.0,
        'enable_noise': True,
        'sampling_rate': 100.0,
        'delay': 0.0
    }
    default_config.update(kwargs)
    return PositionSensor(name, default_config)

class VelocitySensor(SensorBase):
    """Velocity sensor (e.g., from GPS or INS)"""
    
    def sense(self, true_velocity: np.ndarray) -> np.ndarray:
        """Apply velocity sensor model"""
        measured = self._apply_bias_and_scale(true_velocity)
        measured = self._apply_noise(measured)
        measured = self._apply_limits(measured)
        return measured

def create_velocity_sensor(name: str, **kwargs) -> VelocitySensor:
    """Factory function to create velocity sensors"""
    default_config = {
        'noise_std': 0.05,
        'bias': 0.0,
        'scale_factor': 1.0,
        'enable_noise': True,
        'sampling_rate': 100.0,
        'delay': 0.0
    }
    default_config.update(kwargs)
    return VelocitySensor(name, default_config)

class AttitudeSensor(SensorBase):
    """IMU/Attitude sensor for orientation"""
    
    def sense(self, true_quaternion: np.ndarray) -> np.ndarray:
        """Apply attitude sensor model"""
        # measured = self._apply_bias_and_scale(true_quaternion)
        measured = self._apply_noise(true_quaternion)
        measured = self._apply_limits(measured)
        
        # Normalize quaternion after noise addition
        measured = measured / np.linalg.norm(measured)
        return measured

def create_attitude_sensor(name: str, **kwargs) -> AttitudeSensor:
    """Factory function to create attitude sensors"""
    default_config = {
        'noise_std': 0.01,
        'bias': 0.0,
        'scale_factor': 1.0,
        'enable_noise': True,
        'sampling_rate': 100.0,
        'delay': 0.0
    }
    default_config.update(kwargs)
    return AttitudeSensor(name, default_config)

class AngularVelocitySensor(SensorBase):
    """Gyroscope sensor for angular velocity"""
    
    def sense(self, true_angular_vel: np.ndarray) -> np.ndarray:
        """Apply gyroscope sensor model"""
        measured = self._apply_bias_and_scale(true_angular_vel)
        measured = self._apply_noise(measured)
        measured = self._apply_limits(measured)
        return measured

def create_angular_velocity_sensor(name: str, **kwargs) -> AngularVelocitySensor:
    """Factory function to create angular velocity sensors"""
    default_config = {
        'noise_std': 0.02,
        'bias': 0.001,
        'scale_factor': 1.0,
        'enable_noise': True,
        'sampling_rate': 100.0,
        'delay': 0.0
    }
    default_config.update(kwargs)
    return AngularVelocitySensor(name, default_config)

class SensorSystem:
    """Complete sensor system for aircraft"""
    
    def __init__(self):
        self.sensor_groups: Dict[str, SensorGroup] = {}
        self._sensor_factories = {
            'position': create_position_sensor,
            'velocity': create_velocity_sensor,
            'attitude': create_attitude_sensor,
            'angular_velocity': create_angular_velocity_sensor
        }
        self.use_noise = False  # Control flag
    
    def register_sensor_factory(self, sensor_type: str, factory_func):
        """Register a new sensor factory function"""
        self._sensor_factories[sensor_type] = factory_func
    
    def add_sensor_group(self, group_name: str, sensor_configs: Dict[str, Dict[str, Any]]):
        """Add a group of sensors from configuration"""
        group = SensorGroup(group_name)
        
        for sensor_name, config in sensor_configs.items():
            sensor_type = config.get('type', group_name)  # Default to group name
            
            if sensor_type not in self._sensor_factories:
                raise ValueError(f"Unknown sensor type: {sensor_type}")
            
            # Remove 'type' from config before passing to factory
            sensor_config = {k: v for k, v in config.items() if k != 'type'}
            
            # Create sensor using factory
            factory_func = self._sensor_factories[sensor_type]
            sensor = factory_func(sensor_name, **sensor_config)
            group.add_sensor(sensor)
        
        self.sensor_groups[group_name] = group
    
    def update_time(self, current_time: float):
        """Update time for all sensor groups"""
        for group in self.sensor_groups.values():
            group.update_time(current_time)
    
    def sense_state(self, true_state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply all sensors to the true state (basic version)"""
        measured_state = {}
        
        for group_name, group in self.sensor_groups.items():
            group_measurements = group.sense_all(true_state)
            measured_state.update(group_measurements)
        
        return measured_state
    
    def sense_state_with_timing(self, true_state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply all sensors to the true state with timing considerations"""
        measured_state = {}
        
        for group_name, group in self.sensor_groups.items():
            group_measurements = group.sense_all_with_timing(true_state)
            measured_state.update(group_measurements)
        
        return measured_state
    
    def seed(self, seed: Optional[int] = None):
        """Seed all sensor groups"""
        for i, group in enumerate(self.sensor_groups.values()):
            group_seed = None if seed is None else seed + i * 100
            group.seed(group_seed)
    
    def reset_all(self):
        """Reset all sensors"""
        for group in self.sensor_groups.values():
            group.reset()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the entire sensor system"""
        return {
            'groups': {name: group.get_group_info() for name, group in self.sensor_groups.items()},
            'total_sensors': sum(len(group.sensors) for group in self.sensor_groups.values())
        }
    
    def enable_noise(self, enable: bool = True):
        """Enable or disable sensor noise for all sensors"""
        self.use_noise = enable
        # Optionally propagate to individual sensors
        for group in self.sensor_groups.values():
            for sensor in group.sensors:
                sensor.enable_noise = enable
    
    def get_measured_state(self, true_state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Get sensor measurements with noise control
        
        This is the interface method that respects the use_noise flag
        """
        if not self.use_noise:
            return true_state.copy()  # Return true state when noise disabled
        
        # Use timing-aware sensing when noise is enabled
        return self.sense_state_with_timing(true_state)
