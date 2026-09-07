import numpy as np
import os
from falcons.envs.cpu import CpuEnv

from scipy.integrate import solve_ivp
from falcons.controllers.lqr.reference import compute_reference

class LQR_Control(CpuEnv):
    """
    LQR Position Control Environment for Aircraft. This environment uses a LQR controller to maintain the
    aircraft at a target y and z positions by adjusting the motor and control surfaces deflections. The environment provides feedback on the state
    errors, control actions, and stability conditions to ensure smooth flight.
    """
    def __init__(self, aircraft_name: str = None, 
                 targets=None, target_range=None, init_vel_range=(25, 35), 
                 spawning_distance=4, target_airspeed=28, 
                 k_i=np.array([0, -1, -1.5]), save_history=False, 
                 use_sensor_noise=True, use_estimator=True):
        """
        Initialize LQR controller
        
        Args:
            aircraft_name: Name of aircraft (e.g., 'Airship_V7')
            targets: Trajectory targets configuration
            target_range: Range for target positioning
            init_vel_range: Initial velocity range
            spawning_distance: Distance for spawning the aircraft
            target_airspeed: Target airspeed for the aircraft
            k_i: Integral gain for the controller
            save_history: Flag to save history of states and actions
            use_sensor_noise: Flag to enable sensor noise
            use_estimator: Flag to use state estimator
        """
        # Pass to parent (which creates the aircraft model)
        super().__init__(aircraft_name=aircraft_name, save_history=save_history)
        
        # Initialize trajectory configuration
        if targets is None:
            targets = {'trajectory_type': 'constant_altitude'}
        self.targets = targets
        self.trajectory_type = targets.get('trajectory_type', 'constant_altitude')
        
        self.target_range = target_range
        self.spawning_distance = spawning_distance
        self.init_vel_range = init_vel_range
        
        # Load LQR gains using config manager. LQR_GAINS_PATH overrides the aircraft's stored
        # matrix, so an experiment can evaluate a different tuning without mutating the repo's
        # committed gains.
        lqr_path = os.environ.get("LQR_GAINS_PATH") or self.airship.config_manager.lqr_gains_path
        self.K_LQR = np.loadtxt(lqr_path, delimiter=',')  # LQR gain
        
        self.k_i = k_i  # Integral gain
        self.dt = self.airship.EP['dt']  # Simulation time step
        
        # Get aircraft configuration for adaptive dimensions
        self._setup_adaptive_dimensions()

        # Use aircraft-specific target airspeed from JSON
        initial_conditions = self.airship.IP
        init_speed = np.linalg.norm(np.array(initial_conditions.get('linear_vel', [target_airspeed, 0, 0])))
        self.target_airspeed = init_speed if init_speed > 0 else target_airspeed
        self.target_y = 0

        # Load aircraft-specific actuator dynamics per actuator
        self._load_actuator_dynamics()

        total_actuators = self.n_motors + self.n_aero_surfaces

        # Action/observation widths as plain arrays: CpuEnv is not a reinforcement-learning
        # Env subclass, and the only thing it reads off these is action_space.shape[0].
        self.action_space = np.zeros(total_actuators, dtype=np.float64)
        self.observation_space = np.zeros(7, dtype=np.float64)

        # Default actuator dynamics
        self.zeta = 2**(1/2)/2  # Damping ratio for the second-order system
        self.omega_n = 10.0  # Natural frequency for the second-order system
        self.T_s = 0.2  # Time constant for the first-order system
        
        # Initialize state vectors with adaptive dimensions
        self.motor_states_est = np.zeros(self.n_motors) # State vector for motor dynamics (1 per motor)
        self.actuator_states_est = np.zeros(self.n_aero_states)  # State vector for actuator dynamics (2 per surface)
        self.previous_lqr_action = np.zeros(total_actuators)
        self.integral_error = np.zeros(3)  # Integral error for the controller
        self.t = 0  # Time variable
        
        # Sensor and Estimator configuration
        self.use_sensor_noise = use_sensor_noise
        self.use_estimator = use_estimator
        
        # Enable sensor noise if sensor system exists
        if hasattr(self.airship, 'sensor_system') and self.airship.sensor_system is not None:
            self.airship.sensor_system.enable_noise(use_sensor_noise)
        
        # Setup estimator if enabled and configured in JSON
        if self.use_estimator:
            self._setup_estimator_from_config()

    def _setup_estimator_from_config(self):
        """Setup state estimator from JSON configuration"""
        # Check if estimator system exists and is configured
        if not hasattr(self.airship, 'estimator_system') or self.airship.estimator_system is None:
            print("Warning: State estimator requested but no estimator_system found in JSON config. Disabling state estimator.")
            self.use_estimator = False
            return
        
        # Get system info
        system_info = self.airship.estimator_system.get_system_info()
        if system_info['total_estimators'] == 0:
            print("Warning: State estimator requested but no estimators configured. Disabling state estimator.")
            self.use_estimator = False
            return
        
        # Enable estimation in the estimator system
        self.airship.estimator_system.enable_estimation(True)
        print(f"State estimator initialized: {system_info['active_estimator']}")

    def _setup_adaptive_dimensions(self):
        """Setup adaptive dimensions based on aircraft configuration"""
        # Get actuator configurations
        aero_group = self.airship.actuator_system.actuator_groups['aero_surfaces']
        motor_group = self.airship.actuator_system.actuator_groups['motors']
        
        # Store actuator information
        self.aero_actuator_names = [actuator.name for actuator in aero_group.actuators]
        self.motor_names = [actuator.name for actuator in motor_group.actuators]
        
        # Calculate dimensions
        self.n_aero_surfaces = len(self.aero_actuator_names)
        self.n_motors = len(self.motor_names)
        self.n_aero_states = 2 * self.n_aero_surfaces  # 2 states per control surface
        
        print(f"Aircraft configuration:")
        print(f"Control surfaces: {self.aero_actuator_names} ({self.n_aero_surfaces} surfaces, {self.n_aero_states} states)")
        print(f"Motors: {self.motor_names} ({self.n_motors} motors)")
        
    def reset(self, seed=None, **kwargs):
        """
        Resets the environment and the LQR controller state.
        
        :param seed: Seed for the random number generator.
        :return: Initial state of the environment.
        """
        state = super().reset(seed=seed, **kwargs)

        # Reset LQR controller state with adaptive dimensions
        self.motor_states_est = np.zeros(self.n_motors)  # Reset motor states
        self.actuator_states_est = np.zeros(self.n_aero_states)  # Reset actuator states
        self.previous_lqr_action = np.zeros(self.n_motors + self.n_aero_surfaces)  # Reset previous LQR action
        self.integral_error = np.zeros(3)  # Reset integral error

        return state, self.info
    
    def initial_state(self):
        """
        Generate initial state using aircraft-specific parameters from JSON config
        """
        # Get initial conditions from vehicle parameters
        initial_conditions = self.airship.IP
        
        # Extract each component with fallbacks
        position = np.array(initial_conditions.get('position', [0, 0, -1]))
        linear_vel = np.array(initial_conditions.get('linear_vel', [28, 0, 0]))
        angular_vel = np.array(initial_conditions.get('angular_vel', [0, 0, 0]))
        orientation = np.array(initial_conditions.get('orientation', [1, 0, 0, 0]))
        
        return {
            'position': position,
            'linear_vel': linear_vel,
            'angular_vel': angular_vel,
            'orientation': orientation
        }
    
    def get_current_state(self):
        """Get current state from sensors with optional state estimation"""
    
        # Get measurements from sensors if they exist
        if self.airship.sensor_system is not None:
            # Update sensor system time
            self.airship.sensor_system.update_time(self.t)
            
            # Get true state for sensors
            true_state = {
                'gps_position': self.airship.state['position'],
                'gps_velocity': self.airship.state['linear_vel'],
                'attitude_sensor': self.airship.state['orientation'],
                'gyroscope': self.airship.state['angular_vel']
            }
            
            # Get measurements (with or without noise based on flag)
            if self.use_sensor_noise:
                measured_state = self.airship.sensor_system.sense_state_with_timing(true_state)
            else:
                # No noise - return perfect measurements
                measured_state = {
                    'gps_position': true_state['gps_position'],
                    'gps_velocity': true_state['gps_velocity'], 
                    'attitude_sensor': true_state['attitude_sensor'],
                    'gyroscope': true_state['gyroscope']
                }

            # Apply state estimation if enabled (works with both noisy and perfect measurements)
            if self.use_estimator and self.airship.estimator_system is not None:
                # Use state estimator to filter the noisy measurements
                try:
                    filtered_state = self.airship.estimator_system.estimate_state(measured_state, current_time=self.t)
                    
                    test = {
                        'position': filtered_state.get('position', measured_state['gps_position']),
                        'linear_vel': filtered_state.get('linear_vel', measured_state['gps_velocity']),
                        'orientation': filtered_state.get('orientation', measured_state['attitude_sensor']),
                        'angular_vel': filtered_state.get('angular_vel', measured_state['gyroscope'])
                    }
                    return test
                except Exception as e:
                    print(f"Warning: state estimation failed at t={self.t:.2f}: {e}")
                    print("Falling back to raw sensor measurements.")
                    # Fall through to use raw measurements

            # Use raw measurements (either noisy or perfect)
            return {
                'position': measured_state.get('gps_position', self.airship.state['position']),
                'linear_vel': measured_state.get('gps_velocity', self.airship.state['linear_vel']),
                'orientation': measured_state.get('attitude_sensor', self.airship.state['orientation']),
                'angular_vel': measured_state.get('gyroscope', self.airship.state['angular_vel'])
            }
        else:
            # No sensor system - return true state
            return self.airship.state.copy()

    def _extract_state(self, full_state, action):
        """Extract state for observation (can use either true or measured state)"""
        # Use measured state for consistency with controller
        measured_state = self.get_current_state()
        
        current_altitude = measured_state['position'][2]
        
        target_position = self.target_ref()
        error_h = target_position[2] - current_altitude

        current_velocity = measured_state['linear_vel']
        current_angular_velocity = measured_state['angular_vel']
        current_quaternion = measured_state['orientation'][1:]
        current_position = measured_state['position']

        state = [
            float(error_h),
            float(current_velocity[0]), float(current_velocity[1]), float(current_velocity[2]),
            float(current_angular_velocity[0]), float(current_angular_velocity[1]), float(current_angular_velocity[2]),
            float(current_quaternion[0]), float(current_quaternion[1]), float(current_quaternion[2]),
            float(current_position[0]), float(current_position[1]), float(current_position[2]), *action
        ]

        return state
    
    def target_ref(self): 
        """ 
        Generate the target reference position for the aircraft based on the current time. 

        :param self: Total time elapsed.

        :return: Target reference position array. 
        """ 
        # Get reference from trajectory system
        ref = compute_reference(self.t, self.trajectory_type)
        
        # ref returns [y_pos, z_pos, velocity] - we only need position
        return ref # [pos_x, pos_y, pos_z]

    def reference_state(self):
        """
        Generate the reference state for the LQR controller.
        
        :return: Reference state array containing desired velocity, angular velocity, orientation, and position.
        """
        
        # reference_velocity = np.array([ref[2], 0, 0])  # [forward_velocity, 0, 0]
        reference_velocity = np.array([self.target_airspeed, 0, 0])  # [forward_velocity, 0, 0]
        reference_angular_velocity = np.zeros(3)  # Desired angular velocity (no rotation)
        reference_quaternion = np.array([1, 0, 0, 0])  # Desired orientation (no rotation)
        reference_position = self.target_ref()   # [x_pos, y_pos, z_pos]

        return np.concatenate([reference_velocity, reference_angular_velocity, reference_quaternion, reference_position])
    
    def _load_actuator_dynamics(self):
        """Load actuator dynamics parameters per actuator from aircraft JSON configuration"""
        
        actuator_config = self.airship.VP.get('actuator_system', {})
        
        # Load control surface dynamics per surface
        aero_config = actuator_config.get('aero_surfaces', {})
        self.surface_dynamics = {}
        
        for surface_name in self.aero_actuator_names:
            surface_config = aero_config.get(surface_name, {})
            self.surface_dynamics[surface_name] = {
                'zeta': surface_config.get('zeta', 2**(1/2)/2),
                'omega_n': surface_config.get('omega_0', 10.0)
            }
        
        # Load motor dynamics per motor
        motor_config = actuator_config.get('motors', {})
        self.motor_dynamics = {}
        
        for motor_name in self.motor_names:
            motor_config_item = motor_config.get(motor_name, {})
            self.motor_dynamics[motor_name] = {
                'T_s': motor_config_item.get('T', 0.2)
            }
        
        print(f"Per-actuator dynamics loaded:")
        for name, params in self.surface_dynamics.items():
            print(f"  {name}: zeta={params['zeta']:.3f}, omega_n={params['omega_n']:.1f}")
        for name, params in self.motor_dynamics.items():
            print(f"  {name}: T_s={params['T_s']:.3f}")

    def get_quaternion_error(self, current_quat, reference_quat):
        """
        Calculate the quaternion error between the current and desired orientation.
        
        :param current_quat: Current orientation quaternion
               reference_quat: Desired orientation quaternion

        :return: Quaternion error (array of size 4)
        """   
        
        # Calculate quaternion error
        q_err = np.array([
            reference_quat[0]*current_quat[0] + reference_quat[1]*current_quat[1] + reference_quat[2]*current_quat[2] + reference_quat[3]*current_quat[3],
            -reference_quat[1]*current_quat[0] + reference_quat[0]*current_quat[1] + reference_quat[3]*current_quat[2] - reference_quat[2]*current_quat[3],
            -reference_quat[2]*current_quat[0] - reference_quat[3]*current_quat[1] + reference_quat[0]*current_quat[2] + reference_quat[1]*current_quat[3],
            -reference_quat[3]*current_quat[0] + reference_quat[2]*current_quat[1] - reference_quat[1]*current_quat[2] + reference_quat[0]*current_quat[3]
        ])
        
        return q_err
    
    def first_order_actuator_dynamics(self, u_previous, dt):
        """
        Implements a first-order dynamic model for motor dynamics.

        Args:
            u_previous: Motor control inputs (array of size n_motors)
            dt (float): Time step.

        Updates:
            self.motor_states_est: Motor state vector (size n_motors)
        """
        def dynamics(t, y, u, T_s):
            return -1/T_s * y + u

        x_next = np.zeros_like(self.motor_states_est)
        for i in range(self.n_motors):
            motor_name = self.motor_names[i]
            T_s = self.motor_dynamics[motor_name]['T_s']
            
            y = self.motor_states_est[i]
            u = u_previous[i] if len(u_previous) > i else 0.0
            result = solve_ivp(
                fun=dynamics,
                t_span=[0, dt],
                y0=[y],
                args=(u, T_s),
                method='RK45',
                rtol=1e-6,
                atol=1e-9,
                t_eval=[dt]
            )
            x_next[i] = result.y[0, -1]
        self.motor_states_est = x_next
    
    def second_order_actuator_dynamics(self, u_previous, dt):
        """
        Implements a second-order dynamic model for control surface dynamics.

        Args:
            u_previous: Control surface inputs (array of size n_aero_surfaces)
            dt (float): Time step.

        Updates:
            self.actuator_states_est: Control surface state vector (size 2*n_aero_surfaces)
        """
        def dynamics(t, y, u, wn, zeta):
            x_dot = y[0]
            x     = y[1]
            dxdt = x_dot
            dx_dotdt = -2 * zeta * wn * x_dot - wn**2 * x + u
            return [dx_dotdt, dxdt]

        x_next = np.zeros_like(self.actuator_states_est)
        for i in range(self.n_aero_surfaces):
            surface_name = self.aero_actuator_names[i]
            omega_n = self.surface_dynamics[surface_name]['omega_n']
            zeta = self.surface_dynamics[surface_name]['zeta']
            
            state_idx = i * 2
            y0 = [self.actuator_states_est[state_idx],     
                  self.actuator_states_est[state_idx + 1]] 
            u = u_previous[i] if len(u_previous) > i else 0.0
            result = solve_ivp(
                fun=dynamics,
                t_span=[0, dt],
                y0=y0,
                args=(u, omega_n, zeta),
                method='RK45',
                rtol=1e-6,
                atol=1e-9,
                t_eval=[dt]
            )
            x_next[state_idx] = result.y[0, -1]      # x_dot
            x_next[state_idx + 1] = result.y[1, -1]  # x
        self.actuator_states_est = x_next

    def LQR_action(self, measured_state, x_integral):
        """
        Use the LQR controller to adjust actuators based on the MEASURED state.

        :param measured_state: The current measured state of the system.
        :param x_integral: Integral states.
        :return: An array containing the LQR control action.
        """

        # Get MEASURED states from sensors
        current_velocity = measured_state['linear_vel']
        current_angular_velocity = measured_state['angular_vel']
        current_quaternion = measured_state['orientation']
        current_altitude = measured_state['position']

        quaternion_error = self.get_quaternion_error(current_quaternion, self.reference_state()[6:10])
        current_quaternion = quaternion_error[1:]

        current_state = np.concatenate([
                        self.actuator_states_est,
                        self.motor_states_est,
                        current_velocity,
                        current_angular_velocity,
                        current_quaternion,
                        current_altitude,
                        x_integral
                    ])
        
        # Adaptive reference with correct dimensions
        reference_actuator_motor = np.zeros(self.n_aero_states + self.n_motors)
        reference = np.concatenate([
                        reference_actuator_motor,
                        self.reference_state()[0:6],
                        np.zeros(3),
                        self.reference_state()[10:13],
                        np.zeros(3)
                    ])
        
        # Calculate the control action using the LQR controller
        lqr_control_action = -self.K_LQR @ (current_state - reference)

        return lqr_control_action
    
    def integral_action(self, error):
        """
        Calculate the integral action for the LQR controller.

        :param error: Current error in position (array of size 3).
        :return: Integral action based on the error (array of size 3).
        """
        def dynamics(t, y, error):
            # dy/dt = error (element-wise for each axis)
            return error

        # Integrate the integral error for each element using RK45
        integral_error_new = np.zeros_like(self.integral_error)
        for i in range(3):
            result = solve_ivp(
                fun=dynamics,
                t_span=[0, self.dt],
                y0=[self.integral_error[i]],
                args=([error[i]],),
                method='RK45',
                rtol=1e-6,
                atol=1e-9,
                t_eval=[self.dt]
            )
            integral_error_new[i] = result.y[0, -1]

        self.integral_error = integral_error_new

        integral_action = self.k_i * self.integral_error

        return integral_action
    
    def normalize_actuator_commands(self, lqr_action):
        """
        Normalize LQR control commands to actuator input ranges for any number of motors and control surfaces.
        
        Args:
            lqr_action: Raw LQR control action [motors..., aero_surfaces...]
            
        Returns:
            combined_action: Normalized action array for the aircraft actuators
        """
        # Get actuator groups
        aero_group = self.airship.actuator_system.actuator_groups['aero_surfaces']
        motor_group = self.airship.actuator_system.actuator_groups['motors']

        # Extract LQR action
        motor_commands = lqr_action[:self.n_motors]
        
        # Control surfaces: [aileron, elevator, rudder] 
        aileron_cmd = lqr_action[self.n_motors]
        elevator_cmd = lqr_action[self.n_motors + 1] 
        rudder_cmd = lqr_action[self.n_motors + 2]
        
        # Match LQR actuator expected order
        aircraft_action = np.concatenate([
            motor_commands,                              
            [elevator_cmd, aileron_cmd, rudder_cmd]      # Order: elevator, aileron, rudder
        ])
        
        normalized_action = np.zeros(len(aircraft_action))
        
        # Normalize motor actions
        for i in range(self.n_motors):
            motor_limits = motor_group.actuators[i].limits
            normalized_action[i] = np.clip(
                2 * (aircraft_action[i] - motor_limits.min_value) / 
                (motor_limits.max_value - motor_limits.min_value) - 1, 
                -1, 1
            )
        
        # Normalize control surface actions
        for i in range(self.n_aero_surfaces):
            aero_limits = aero_group.actuators[i].limits
            action_idx = self.n_motors + i
            
            # Convert from radians to degrees
            surface_deg = aircraft_action[action_idx] * (180/np.pi)
            normalized_action[action_idx] = np.clip(
                2 * (surface_deg - aero_limits.min_value) / 
                (aero_limits.max_value - aero_limits.min_value) - 1, 
                -1, 1
            )

        return normalized_action

    def compute_action(self):
        """
        Compute the control action using MEASURED state from sensors.
        
        :return: Formatted action array for the aircraft.
        """
        # Get MEASURED position from sensors instead of true position
        measured_state = self.get_current_state()
        current_pos = measured_state['position']
        
        # Calculate the error in position using measured position
        error = self.reference_state()[10:13] - current_pos

        # Update the actuator states using second-order dynamics (control surfaces)
        if self.n_aero_surfaces > 0:
            aero_actions = self.previous_lqr_action[self.n_motors:self.n_motors + self.n_aero_surfaces]
            self.second_order_actuator_dynamics(aero_actions, self.dt)

        # Update the motor states using first-order dynamics
        if self.n_motors > 0:
            motor_actions = self.previous_lqr_action[0:self.n_motors]
            self.first_order_actuator_dynamics(motor_actions, self.dt)
        
        # Compute the integral action based on the position error
        integral_action = self.integral_action(error)
        
        # Compute the LQR control action
        lqr_action = self.LQR_action(measured_state, integral_action)

        # Update previous LQR action for next step
        self.previous_lqr_action = lqr_action.copy()

        # Normalize the LQR action to actuator commands
        combined_action = self.normalize_actuator_commands(lqr_action)
        
        return combined_action

    def step(self):
        """
        Take a step in the environment, using the LQR controller to adjust actuators based on the current state.
        
        :return: A tuple containing the next state, reward, done flag, truncation flag, and info dictionary.
        """
        # Step the environment with the computed action
        action = self.compute_action()  # Compute the control action using LQR

        self.t += self.dt  # Increment time step   

        self.airship.current_time = self.t 
        
        aero_action = action[self.n_motors:self.n_motors + self.n_aero_surfaces]  # Aero actions
        motor_action = action[0:self.n_motors]                                   # Motor actions
        
        next_state, reward, done, truncated, info = super().step(aero_action, motor_action)
        return next_state, reward, done, truncated, info, action

    def close(self):
        """
        Closes the environment.
        """
        return super().close()
