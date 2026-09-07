"""CPU state estimators: config and the abstract base, the concrete Extended Kalman Filter and
low-pass filter, then the type registry and the per-aircraft EstimatorSystem that the CPU plant
builds from JSON."""
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field

@dataclass
class EstimatorConfig:
    """Configuration for state estimators"""
    process_noise: Union[float, Dict[str, float]] = 0.01
    measurement_noise: Union[float, Dict[str, float]] = 0.1
    initial_covariance: float = 1.0
    enable_estimation: bool = True

    def get_process_noise(self, state_component: str, default: float = 0.01) -> float:
        """Get process noise for specific state component"""
        if isinstance(self.process_noise, dict):
            return self.process_noise.get(state_component, default)
        return self.process_noise

    def get_measurement_noise(self, state_component: str, default: float = 0.1) -> float:
        """Get measurement noise for specific state component"""
        if isinstance(self.measurement_noise, dict):
            return self.measurement_noise.get(state_component, default)
        return self.measurement_noise

class BaseEstimator(ABC):
    """Abstract base class for all state estimators"""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

        # Handle both old (float) and new (dict) noise configurations
        process_noise = config.get('process_noise', 0.01)
        measurement_noise = config.get('measurement_noise', 0.1)

        self.estimator_config = EstimatorConfig(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            initial_covariance=config.get('initial_covariance', 1.0),
            enable_estimation=config.get('enable_estimation', True)
        )

        # State and covariance
        self.state_estimate = None
        self.covariance = None
        self.is_initialized = False

        # Timing
        self.dt = config.get('dt', 0.01)
        self.last_update_time = 0.0

        # Random number generator
        self.np_random = None

    @abstractmethod
    def predict(self, dt: float, control_input: Optional[np.ndarray] = None) -> np.ndarray:
        """Prediction step"""
        pass

    @abstractmethod
    def update(self, measurement: Dict[str, np.ndarray]) -> np.ndarray:
        """Update step with measurements"""
        pass

    @abstractmethod
    def initialize(self, initial_measurement: Dict[str, np.ndarray]) -> None:
        """Initialize estimator with first measurement"""
        pass

    def estimate(self, measurement: Dict[str, np.ndarray],
                control_input: Optional[np.ndarray] = None,
                current_time: float = 0.0) -> Dict[str, np.ndarray]:
        """Main estimation function"""
        if not self.estimator_config.enable_estimation:
            return measurement  # Pass through if disabled

        if not self.is_initialized:
            self.initialize(initial_measurement=measurement)
            self.last_update_time = current_time
            return self._state_to_dict()

        # Calculate time step
        dt = current_time - self.last_update_time if current_time > self.last_update_time else self.dt

        # Prediction step
        self.predict(dt, control_input)

        # Update step with measurements
        self.update(measurement)

        self.last_update_time = current_time
        return self._state_to_dict()

    @abstractmethod
    def _state_to_dict(self) -> Dict[str, np.ndarray]:
        """Convert internal state to output dictionary"""
        pass

    def seed(self, seed: Optional[int] = None):
        """Seed the random number generator"""
        self.np_random = np.random.RandomState(seed)

    def reset(self):
        """Reset estimator to uninitialized state"""
        self.state_estimate = None
        self.covariance = None
        self.is_initialized = False
        self.last_update_time = 0.0

class EstimatorGroup:
    """Group of estimators"""

    def __init__(self, group_name: str):
        self.group_name = group_name
        self.estimators: List[BaseEstimator] = []

    def add_estimator(self, estimator: BaseEstimator):
        """Add estimator to group"""
        self.estimators.append(estimator)

    def estimate_all(self, measurements, control_input, current_time):
        estimates = {}
        for estimator in self.estimators:
            estimate = estimator.estimate(measurements, control_input, current_time)
            estimates[estimator.name] = estimate  # Store by estimator name
        return estimates

    def seed(self, seed: Optional[int] = None):
        """Seed all estimators in this group"""
        for i, estimator in enumerate(self.estimators):
            estimator_seed = None if seed is None else seed + i
            estimator.seed(estimator_seed)

    def reset_all(self):
        """Reset all estimators in group"""
        for estimator in self.estimators:
            estimator.reset()

class ExtendedKalmanFilter(BaseEstimator):
    """Extended Kalman Filter for aircraft state estimation"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)

        # State dimension (position, velocity, orientation quaternion, angular velocity)
        # [px, py, pz, vx, vy, vz, p, q, r, q0, q1, q2, q3] = 13 states
        self.state_dim = 13
        self.measurement_dim = 13  # Same as state for full state measurement

        # Process and measurement noise matrices
        self.Q = self._build_process_noise_matrix()
        self.R = self._build_measurement_noise_matrix()

        # State and covariance initialization
        self.state_estimate = np.zeros(self.state_dim)
        self.state_estimate[9] = 1.0

        self.covariance = np.eye(self.state_dim) * self.estimator_config.initial_covariance

    def _build_process_noise_matrix(self) -> np.ndarray:
        """Build process noise covariance matrix Q with per-state configuration"""

        # Get noise values for each state component
        pos_noise = self.estimator_config.get_process_noise('position', 0.1)
        vel_noise = self.estimator_config.get_process_noise('velocity', 0.05)
        angvel_noise = self.estimator_config.get_process_noise('angular_velocity', 0.02)
        quat_noise = self.estimator_config.get_process_noise('orientation', 0.01)

        # Build diagonal Q matrix
        Q_diag = np.array([
            pos_noise, pos_noise, pos_noise,        # position
            vel_noise, vel_noise, vel_noise,        # velocity
            angvel_noise, angvel_noise, angvel_noise, # angular velocity
            quat_noise, quat_noise, quat_noise, quat_noise  # quaternion
        ])

        return np.diag(Q_diag ** 2)

    def _build_measurement_noise_matrix(self) -> np.ndarray:
        """Build measurement noise covariance matrix R with per-state configuration"""

        # Get measurement noise values for each sensor/state component
        pos_noise = self.estimator_config.get_measurement_noise('gps_position', 0.1)
        vel_noise = self.estimator_config.get_measurement_noise('gps_velocity', 0.05)
        angvel_noise = self.estimator_config.get_measurement_noise('gyroscope', 0.01)
        quat_noise = self.estimator_config.get_measurement_noise('attitude_sensor', 0.005)

        R_diag = np.array([
            pos_noise, pos_noise, pos_noise,        # position measurement
            vel_noise, vel_noise, vel_noise,        # velocity measurement
            angvel_noise, angvel_noise, angvel_noise, # angular velocity measurement
            quat_noise, quat_noise, quat_noise, quat_noise  # quaternion measurement
        ])

        return np.diag(R_diag ** 2)

    def _state_dynamics(self, state: np.ndarray, dt: float,
                       control_input: Optional[np.ndarray] = None) -> np.ndarray:
        """Nonlinear state dynamics model f(x, u)"""
        # Extract state components
        position = state[0:3]
        velocity = state[3:6]
        angular_vel = state[6:9]
        quaternion = state[9:13]

        # Ensure quaternion is normalized
        quaternion = quaternion / np.linalg.norm(quaternion)

        # Position dynamics: p_new = p + v * dt
        new_position = position + velocity * dt

        # Velocity dynamics (simplified - no acceleration input for now)
        # In a full implementation we can include control forces here
        new_velocity = velocity  # Constant velocity model

        # Angular velocity dynamics (simplified)
        new_angular_vel = angular_vel  # Constant angular velocity model

        # Quaternion dynamics: integrate angular velocity
        new_quaternion = self._integrate_quaternion(quaternion, angular_vel, dt)

        return np.concatenate([
            new_position,
            new_velocity,
            new_angular_vel,
            new_quaternion
        ])

    def _integrate_quaternion(self, q: np.ndarray, ang_vel: np.ndarray, dt: float) -> np.ndarray:
        """Integrate quaternion with angular velocity"""

        w_matrix = np.array([[0, -ang_vel[0], -ang_vel[1], -ang_vel[2]],
                            [ang_vel[0], 0, ang_vel[2], -ang_vel[1]],
                            [ang_vel[1], -ang_vel[2], 0, ang_vel[0]],
                            [ang_vel[2], ang_vel[1], -ang_vel[0], 0]])
        q_dot = 0.5 * w_matrix @ q

        # Quaternion multiplication: q_new = dq * q
        new_q = q + q_dot * dt
        return new_q / np.linalg.norm(new_q)

    def _jacobian_f(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Compute Jacobian of state dynamics"""
        F = np.eye(self.state_dim)

        pos_i = slice(0, 3)
        vel_i = slice(3, 6)
        omega_i = slice(6, 9)   # 3 elements
        quat_i = slice(9, 13)   # 4 elements

        # Position derivatives w.r.t velocity
        F[pos_i, vel_i] = np.eye(3) * dt

        # Quaternion terms
        q = state[quat_i]  # q0, q1, q2, q3 (scalar-first)
        q0, q1, q2, q3 = q
        wx, wy, wz = state[omega_i]

        # Build G(q) (4x3)
        G = np.array([
            [-q1, -q2, -q3],
            [ q0, -q3,  q2],
            [ q3,  q0, -q1],
            [-q2,  q1,  q0]
        ], dtype=np.float64)

        # Build Ω(omega) (4x4)
        Omega = np.array([
            [0,   -wx, -wy, -wz],
            [wx,   0,   wz, -wy],
            [wy,  -wz,  0,   wx],
            [wz,   wy, -wx,  0 ]
        ], dtype=np.float64)

        # Blocks
        F[quat_i, omega_i] = 0.5 * dt * G         # dq/dω
        F[quat_i, quat_i]  = np.eye(4) + 0.5 * dt * Omega  # dq/dq

        return F

    def _measurement_model(self, state: np.ndarray) -> np.ndarray:
        """Measurement model h(x) - maps state to expected measurements"""
        # Direct measurement of all states (full state feedback)
        return state.copy()

    def _jacobian_h(self, state: np.ndarray) -> np.ndarray:
        """Compute Jacobian of measurement model"""
        # Identity matrix for direct state measurement
        return np.eye(self.measurement_dim, self.state_dim)

    def predict(self, dt: float, control_input: Optional[np.ndarray] = None) -> np.ndarray:
        """EKF Prediction step"""
        # Predict state
        self.state_estimate = self._state_dynamics(self.state_estimate, dt, control_input)

        # Ensure quaternion normalization
        self.state_estimate[9:13] = self.state_estimate[9:13] / np.linalg.norm(self.state_estimate[9:13])

        # Predict covariance
        F = self._jacobian_f(self.state_estimate, dt)
        self.covariance = F @ self.covariance @ F.T + self.Q

        return self.state_estimate

    def update(self, measurement: Dict[str, np.ndarray]) -> np.ndarray:
        """EKF Update step"""
        # Convert measurement dict to vector
        z = self._measurement_dict_to_vector(measurement)

        if z is None:
            return self.state_estimate  # No valid measurement

        # Predicted measurement
        z_pred = self._measurement_model(self.state_estimate)

        # Measurement Jacobian
        H = self._jacobian_h(self.state_estimate)

        # Innovation
        y = z - z_pred

        # Normalize quaternion part of innovation if needed
        if len(y) >= 13:
            # Ensure quaternion innovation is reasonable
            quat_innovation = y[9:13]
            if np.linalg.norm(quat_innovation) > 1.0:
                y[9:13] = quat_innovation / np.linalg.norm(quat_innovation)

        # Innovation covariance
        S = H @ self.covariance @ H.T + self.R

        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.state_estimate = self.state_estimate + K @ y
        I_KH = np.eye(self.state_dim) - K @ H
        # self.covariance = I_KH @ self.covariance
        # Joseph-form covariance protection
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ self.R @ K.T
        self.covariance = (self.covariance + self.covariance.T) * 0.5  # Ensure symmetry

        # Ensure quaternion normalization after update
        self.state_estimate[9:13] = self.state_estimate[9:13] / np.linalg.norm(self.state_estimate[9:13])

        return self.state_estimate

    def _measurement_dict_to_vector(self, measurement: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Convert measurement dictionary to vector"""
        z = []

        # Position (GPS)
        if 'gps_position' in measurement:
            z.extend(measurement['gps_position'])
        else:
            return None  # Need position measurement

        # Velocity (GPS)
        if 'gps_velocity' in measurement:
            z.extend(measurement['gps_velocity'])
        else:
            z.extend([0.0, 0.0, 0.0])  # Zero velocity if not available

        # Angular velocity (IMU)
        if 'gyroscope' in measurement:
            z.extend(measurement['gyroscope'])
        else:
            z.extend([0.0, 0.0, 0.0])  # Zero angular velocity

        # Attitude (IMU)
        if 'attitude_sensor' in measurement:
            quat = measurement['attitude_sensor']
            # Ensure quaternion is normalized
            quat = quat / np.linalg.norm(quat)
            z.extend(quat)
        else:
            z.extend([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

        return np.array(z)

    def initialize(self, initial_measurement: Dict[str, np.ndarray]) -> None:
        """Initialize estimator with first measurement"""
        z = self._measurement_dict_to_vector(initial_measurement)

        if z is not None:
            self.state_estimate = z.copy()
            # Ensure quaternion is normalized
            self.state_estimate[9:13] = self.state_estimate[9:13] / np.linalg.norm(self.state_estimate[9:13])

            self.is_initialized = True

    def _state_to_dict(self) -> Dict[str, np.ndarray]:
        """Convert internal state to output dictionary"""
        return {
            'position': self.state_estimate[0:3].copy(),
            'linear_vel': self.state_estimate[3:6].copy(),
            'angular_vel': self.state_estimate[6:9].copy(),
            'orientation': self.state_estimate[9:13].copy(),
        }

    def get_position_uncertainty(self) -> np.ndarray:
        """Get position uncertainty (standard deviation)"""
        return np.sqrt(np.diag(self.covariance[0:3, 0:3]))

    def get_velocity_uncertainty(self) -> np.ndarray:
        """Get velocity uncertainty (standard deviation)"""
        return np.sqrt(np.diag(self.covariance[3:6, 3:6]))

class LowPassFilter(BaseEstimator):
    """Simple low-pass filter for state estimation"""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)

        # Low-pass filter parameters
        self.cutoff_frequency = config.get('cutoff_frequency', 10.0)  # Hz
        self.dt = config.get('dt', 0.01)  # Time step

        # Calculate filter coefficient
        # alpha = dt / (1/(2*pi*fc) + dt)
        tau = 1.0 / (2.0 * np.pi * self.cutoff_frequency)
        self.alpha = self.dt / (tau + self.dt)

        # State storage
        self.filtered_state = None
        self.is_initialized = False

    def initialize(self, initial_measurement: Dict[str, np.ndarray]):
        """Initialize filter with first measurement"""
        # Convert measurements to state format
        self.filtered_state = {
            'position': initial_measurement.get('gps_position', np.zeros(3)).copy(),
            'linear_vel': initial_measurement.get('gps_velocity', np.zeros(3)).copy(),
            'angular_vel': initial_measurement.get('gyroscope', np.zeros(3)).copy(),
            'orientation': initial_measurement.get('attitude_sensor', np.array([1.0, 0.0, 0.0, 0.0])).copy()
        }

        # Normalize quaternion
        if 'orientation' in self.filtered_state:
            self.filtered_state['orientation'] = self.filtered_state['orientation'] / np.linalg.norm(self.filtered_state['orientation'])

        self.is_initialized = True

    def predict(self, dt: float, control_input: Optional[np.ndarray] = None):
        """Prediction step - not used in simple low-pass filter"""
        pass

    def update(self, measurement: Dict[str, np.ndarray]) -> np.ndarray:
        """Update filter with new measurements"""
        if not self.is_initialized:
            self.initialize(measurement)
            return self._state_to_dict()

        # Convert measurements to consistent format
        current_measurement = {
            'position': measurement.get('gps_position', self.filtered_state['position']),
            'linear_vel': measurement.get('gps_velocity', self.filtered_state['linear_vel']),
            'angular_vel': measurement.get('gyroscope', self.filtered_state['angular_vel']),
            'orientation': measurement.get('attitude_sensor', self.filtered_state['orientation'])
        }

        # Apply low-pass filter: x_filtered = alpha * x_new + (1 - alpha) * x_old
        for key in self.filtered_state.keys():
            if key in current_measurement:
                if key == 'orientation':
                    # Special handling for quaternions
                    self.filtered_state[key] = self._filter_quaternion(
                        current_measurement[key],
                        self.filtered_state[key]
                    )
                else:
                    # Standard low-pass filtering
                    self.filtered_state[key] = (
                        self.alpha * current_measurement[key] +
                        (1.0 - self.alpha) * self.filtered_state[key]
                    )

        return self._state_to_dict()

    def _filter_quaternion(self, new_quat: np.ndarray, old_quat: np.ndarray) -> np.ndarray:
        """Apply low-pass filter to quaternion with proper handling"""
        # Normalize input quaternion
        new_quat = new_quat / np.linalg.norm(new_quat)

        # Check if quaternions are on same hemisphere (avoid long path)
        if np.dot(new_quat, old_quat) < 0:
            new_quat = -new_quat

        # Apply low-pass filter
        filtered_quat = self.alpha * new_quat + (1.0 - self.alpha) * old_quat

        # Normalize result
        return filtered_quat / np.linalg.norm(filtered_quat)

    def _state_to_dict(self) -> Dict[str, np.ndarray]:
        """Convert internal state to output dictionary"""
        if not self.is_initialized or self.filtered_state is None:
            return {}

        return {
            'position': self.filtered_state['position'].copy(),
            'linear_vel': self.filtered_state['linear_vel'].copy(),
            'angular_vel': self.filtered_state['angular_vel'].copy(),
            'orientation': self.filtered_state['orientation'].copy()
        }

    def get_state_uncertainty(self) -> Dict[str, Any]:
        """Get state uncertainty (not applicable for simple low-pass filter)"""
        return {
            'position_std': np.full(3, 0.1),
            'velocity_std': np.full(3, 0.05),
            'angular_velocity_std': np.full(3, 0.01),
            'orientation_std': np.full(4, 0.01)
        }

    def reset(self):
        """Reset filter state"""
        self.filtered_state = None
        self.is_initialized = False

    def set_cutoff_frequency(self, frequency: float):
        """Update cutoff frequency and recalculate filter coefficient"""
        self.cutoff_frequency = frequency
        tau = 1.0 / (2.0 * np.pi * self.cutoff_frequency)
        self.alpha = self.dt / (tau + self.dt)

# Factory registry
_ESTIMATOR_FACTORIES = {
    'extended_kalman': ExtendedKalmanFilter,
    'low_pass_filter': LowPassFilter
}

def register_estimator_factory(estimator_type: str, estimator_class):
    """Register a new estimator factory"""
    _ESTIMATOR_FACTORIES[estimator_type] = estimator_class

def create_estimator(estimator_type: str, name: str, **kwargs) -> BaseEstimator:
    """
    Factory function to create estimators

    Args:
        estimator_type: Type of estimator to create
        name: Name of the estimator instance
        **kwargs: Estimator configuration parameters

    Returns:
        Created estimator instance
    """
    if estimator_type not in _ESTIMATOR_FACTORIES:
        raise ValueError(f"Unknown estimator type: {estimator_type}. Available: {list(_ESTIMATOR_FACTORIES.keys())}")

    estimator_class = _ESTIMATOR_FACTORIES[estimator_type]
    return estimator_class(name, kwargs)

def get_available_estimator_types() -> list:
    """Get list of available estimator types"""
    return list(_ESTIMATOR_FACTORIES.keys())

class EstimatorSystem:
    """Complete estimator system for aircraft"""

    def __init__(self):
        self.estimator_groups: Dict[str, EstimatorGroup] = {}
        self.active_estimator = None
        self.use_estimation = False  # Control flag

    def register_estimator_factory(self, estimator_type: str, estimator_class):
        """Register a new estimator factory"""
        register_estimator_factory(estimator_type, estimator_class)

    def add_estimator_group(self, group_name: str, estimator_configs: Dict[str, Dict[str, Any]]):
        """Add a group of estimators"""
        group = EstimatorGroup(group_name)

        for estimator_name, config in estimator_configs.items():
            estimator_type = config.get('type')
            if not estimator_type:
                raise ValueError(f"Estimator '{estimator_name}' missing required 'type' field")

            # Clean config
            estimator_config = {k: v for k, v in config.items() if k != 'type'}

            # Create estimator
            estimator = create_estimator(estimator_type, estimator_name, **estimator_config)
            group.add_estimator(estimator)

            # Set as active estimator if it's the first one or marked as primary
            if self.active_estimator is None or config.get('primary', False):
                self.active_estimator = estimator

        self.estimator_groups[group_name] = group

    def estimate_state(self, measurements: Dict[str, np.ndarray],
                      control_input: Optional[np.ndarray] = None,
                      current_time: float = 0.0) -> Dict[str, np.ndarray]:
        """Get state estimate from primary estimator"""
        if self.active_estimator is None:
            return measurements  # Fallback to measurements

        return self.active_estimator.estimate(measurements, control_input, current_time)

    def estimate_all(self, measurements: Dict[str, np.ndarray],
                    control_input: Optional[np.ndarray] = None,
                    current_time: float = 0.0) -> Dict[str, Dict[str, np.ndarray]]:
        """Run all estimator groups"""
        all_estimates = {}

        for group_name, group in self.estimator_groups.items():
            group_estimates = group.estimate_all(measurements, control_input, current_time)
            all_estimates[group_name] = group_estimates

        return all_estimates

    def reset_all(self):
        """Reset all estimators"""
        for group in self.estimator_groups.values():
            group.reset_all()

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'groups': {name: len(group.estimators) for name, group in self.estimator_groups.items()},
            'active_estimator': self.active_estimator.name if self.active_estimator else None,
            'total_estimators': sum(len(group.estimators) for group in self.estimator_groups.values())
        }

    def get_state_uncertainty(self) -> Optional[Dict[str, np.ndarray]]:
        """Get uncertainty estimates from active estimator"""
        # Use duck typing instead of isinstance() - consistent with sensor system approach
        if hasattr(self.active_estimator, 'get_position_uncertainty'):
            try:
                return {
                    'position_std': self.active_estimator.get_position_uncertainty(),
                    'velocity_std': self.active_estimator.get_velocity_uncertainty(),
                    'covariance': self.active_estimator.get_state_covariance()
                }
            except AttributeError:
                # If any method is missing, return None
                pass
        return None

    def seed(self, seed: Optional[int] = None):
        """Seed all estimator groups"""
        for i, (group_name, group) in enumerate(self.estimator_groups.items()):
            group_seed = None if seed is None else seed + i * 100
            group.seed(group_seed)

    def enable_estimation(self, enable: bool = True):
        """Enable or disable state estimation"""
        self.use_estimation = enable

    def get_estimated_state(self, measurements: Dict[str, np.ndarray],
                           control_input: Optional[np.ndarray] = None,
                           current_time: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Get estimated state with estimation control

        This is the interface method that respects the use_estimation flag
        """
        if not self.use_estimation or self.active_estimator is None:
            return measurements.copy()  # Return measurements when estimation disabled

        return self.estimate_state(measurements, control_input, current_time)
