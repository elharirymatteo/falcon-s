import numpy as np
from falcons.aircraft.params import load_params

import warp as wp
import time 
import sys
import random

from falcons.sim.warp.physics import aircraft_simulation_step
from falcons.sim.warp.sensors import SensorSystemWARP
from falcons.sim.warp.wind import WindModelWARP
from falcons.sim.warp.history import save_history_kernel
from falcons.sim.warp.termination import check_termination_kernel


class Aircraft: 

    def __init__(self, num_envs: int, device: str, config, save_history: bool = False, env_to_save: list = None, max_steps: int = 10000, in_ground_effect: bool = True):
        """
        Initialize the model
        """
        self._num_envs = num_envs
        self._device = device
        self._ge_enabled = in_ground_effect      # toggle GE lift/drag correction (WIG ablation)

        self._save_history = save_history
        self._env_to_save = env_to_save if env_to_save is not None else [0]
        self._max_steps = max_steps
        self.current_step = 0

        # Unpack config
        self.AP = config['AP']
        self.VP = config['VP']
        self.CL = config['CL']
        self.EP = config['EP']

        # Build all state and temporary buffers
        self.build_all_buffers()

        self.solver_type = 0

        # Termination check buffers (GPU-side flags)
        self._crashed_flag = wp.zeros(1, device=self._device, dtype=wp.int32)
        self._stalled_flag = wp.zeros(1, device=self._device, dtype=wp.int32)

        if self._save_history:
            # Pre-allocate GPU history buffers
            self.history_alpha = wp.zeros(max_steps, device=self._device, dtype=wp.float32)
            self.history_beta = wp.zeros(max_steps, device=self._device, dtype=wp.float32)
            self.history_Va = wp.zeros(max_steps, device=self._device, dtype=wp.float32)
            self.history_hr = wp.zeros(max_steps, device=self._device, dtype=wp.float32)
            
            self.history_Fb = wp.zeros((max_steps, 3), device=self._device, dtype=wp.float32)
            self.history_Mb = wp.zeros((max_steps, 3), device=self._device, dtype=wp.float32)
            self.history_Fb_aero = wp.zeros((max_steps, 3), device=self._device, dtype=wp.float32)
            self.history_Mb_aero = wp.zeros((max_steps, 3), device=self._device, dtype=wp.float32)
            self.history_Fb_thrust = wp.zeros((max_steps, 3), device=self._device, dtype=wp.float32)
            self.history_Mb_thrust = wp.zeros((max_steps, 3), device=self._device, dtype=wp.float32)
            self.history_Fb_g = wp.zeros((max_steps, 3), device=self._device, dtype=wp.float32)
            
            self.history_C_L = wp.zeros(max_steps, device=self._device, dtype=wp.float32)
            self.history_C_D = wp.zeros(max_steps, device=self._device, dtype=wp.float32)
            self.history_C_L_free = wp.zeros(max_steps, device=self._device, dtype=wp.float32)
            self.history_C_D_free = wp.zeros(max_steps, device=self._device, dtype=wp.float32)
            self.history_Cl = wp.zeros(max_steps, device=self._device, dtype=wp.float32)
            self.history_Cm = wp.zeros(max_steps, device=self._device, dtype=wp.float32)
            self.history_Cn = wp.zeros(max_steps, device=self._device, dtype=wp.float32)
            
            self.history_elevator = wp.zeros(max_steps, device=self._device, dtype=wp.float32)
            self.history_aileron = wp.zeros(max_steps, device=self._device, dtype=wp.float32)
            self.history_rudder = wp.zeros(max_steps, device=self._device, dtype=wp.float32)
            self.history_throttle_left = wp.zeros(max_steps, device=self._device, dtype=wp.float32)
            self.history_throttle_right = wp.zeros(max_steps, device=self._device, dtype=wp.float32)
            
            self.history_position = wp.zeros((max_steps, 3), device=self._device, dtype=wp.float32)
            self.history_orientation = wp.zeros((max_steps, 4), device=self._device, dtype=wp.float32)
            self.history_linear_vel = wp.zeros((max_steps, 3), device=self._device, dtype=wp.float32)
            self.history_angular_vel = wp.zeros((max_steps, 3), device=self._device, dtype=wp.float32)

            # Add CPU-side attributes for compatibility with legacy code (MPPI, etc.)
            self.aero_params = {'alpha': [], 'beta': [], 'Va': [], 'height ratio': [], 
                                'Coef_f': [], 'Coef_m': [], 'oge_ige': []}
            self.forces = {'Fb': [], 'Mb': [], 'F_aero': [], 'M_aero': [], 
                           'F_thrust': [], 'M_thrust': [], 'Fb_g': []}
            self.actions = np.zeros(5)

        # Initialize sensor system
        if hasattr(self.VP, 'sensor_system') and self.VP.sensor_system is not None:
            self.sensor_system = SensorSystemWARP(num_envs, device, self.VP.sensor_system, self.EP.dt)
            print(f"Sensor system initialized ({num_envs} environments)")
        else:
            self.sensor_system = None
            print("No sensor system configuration found")

        # Create wind model
        self.wind_model = WindModelWARP(num_envs, device, self.EP, self.VP.WP.span)

        # Apply config seed if available
        if hasattr(self.EP, 'seed') and self.EP.seed is not None:
            self.seed(self.EP.seed)
        else:
            self.seed()


    def build_all_buffers(self):
        """Build all required buffers for the fused kernel"""
        # State buffers
        self._state = {}
        self._state["position"] = wp.zeros((self._num_envs), device=self._device, dtype=wp.vec3f)
        self._state["orientation"] = wp.zeros((self._num_envs), device=self._device, dtype=wp.quatf)
        self._state["orientation"].fill_([0, 0, 0, 1])
        self._state["linear_vel"] = wp.zeros((self._num_envs), device=self._device, dtype=wp.vec3f)
        self._state["angular_vel"] = wp.zeros((self._num_envs), device=self._device, dtype=wp.vec3f)

        # Action command buffers
        self._action_command = {}
        self._action_command["elevator"] = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._action_command["aileron"] = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._action_command["rudder"] = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._action_command["throttle_left"] = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._action_command["throttle_right"] = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)

        # Actuator state buffers
        self._actuator_states = {}
        self._actuator_states["elevator"] = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._actuator_states["aileron"] = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._actuator_states["rudder"] = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._actuator_states["throttle_left"] = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._actuator_states["throttle_right"] = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._actuator_states["elevator_dot"] = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._actuator_states["aileron_dot"] = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._actuator_states["rudder_dot"] = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)

        # Aerodynamic parameter buffers
        self._Va = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._alpha = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._beta = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._rho = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._Q = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._hr = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._airspeed_vector = wp.zeros((self._num_envs), device=self._device, dtype=wp.vec3f)

        # Coefficient buffers (temporary)
        self._C_L = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._C_Y = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._C_D = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._Cl = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._Cm = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._Cn = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._C_D_free = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._C_L_free = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)

        # Force/moment intermediate buffers
        self._D_tot = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._L_tot = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._Y_tot = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._F1_b = wp.zeros((self._num_envs), device=self._device, dtype=wp.vec3f)
        self._F2_b = wp.zeros((self._num_envs), device=self._device, dtype=wp.vec3f)
        self._Fb_thrust = wp.zeros((self._num_envs), device=self._device, dtype=wp.vec3f)
        self._M_prop_left = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._M_prop_right = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._Mb_thrust = wp.zeros((self._num_envs), device=self._device, dtype=wp.vec3f)
        self._Fw_aero = wp.zeros((self._num_envs), device=self._device, dtype=wp.vec3f)
        self._Fb_aero = wp.zeros((self._num_envs), device=self._device, dtype=wp.vec3f)
        self._Mb_aero = wp.zeros((self._num_envs), device=self._device, dtype=wp.vec3f)
        self._q_aero = wp.zeros((self._num_envs), device=self._device, dtype=wp.quatf)
        self._q_aero.fill_([0, 0, 0, 1])
        self._gravity_vector_body = wp.zeros((self._num_envs), device=self._device, dtype=wp.vec3f)
        self._Fb_g = wp.zeros((self._num_envs), device=self._device, dtype=wp.vec3f)
        self._Fb = wp.zeros((self._num_envs), device=self._device, dtype=wp.vec3f)
        self._Mb = wp.zeros((self._num_envs), device=self._device, dtype=wp.vec3f)

        # Additional required buffers
        self._in_ground_effect = wp.zeros((self._num_envs), device=self._device, dtype=bool)

    def reset(self, init_state: dict=None, init_action: dict=None) -> dict:
        """Reset the model to the initial state and set the initial action."""
        
        # Reset states
        self._reset_states(init_state)
        
        # Reset actions
        self._reset_actions(init_action)

        # Reset subsystems and history
        self._reset_subsystems()
        
        return self._state

    def _reset_states(self, init_state: dict):
        """State initialization with automatic shape detection"""
        if init_state is None:
            # Default initialization
            self._state["position"].fill_([0, 0, 0])
            self._state["linear_vel"].fill_([0, 0, 0])
            self._state["angular_vel"].fill_([0, 0, 0])
            self._state["orientation"].fill_(wp.quat_identity())
            return

        # State configuration mapping
        state_config = {
            "position": (self._state["position"], wp.vec3f),
            "linear_vel": (self._state["linear_vel"], wp.vec3f),
            "angular_vel": (self._state["angular_vel"], wp.vec3f),
            "orientation": (self._state["orientation"], wp.quatf)
        }
        
        for key, (buffer, dtype) in state_config.items():
            if key in init_state:
                self._set_buffer_data(buffer, init_state[key], dtype)

    def _reset_actions(self, init_action: dict):
        """Action initialization with automatic shape detection"""
        action_keys = ["elevator", "aileron", "rudder", "throttle_left", "throttle_right", 
                    "elevator_dot", "aileron_dot", "rudder_dot"]
        
        if init_action is None:
            # Zero all actuator states
            for key in action_keys:
                if key in self._actuator_states:
                    self._actuator_states[key].fill_(0.0)
            return
        
        for key in action_keys:
            if key in init_action:
                self._set_buffer_data(self._actuator_states[key], init_action[key], wp.float32)

    def _set_buffer_data(self, buffer, data, dtype):
        """Universal buffer setter - handles both single values and arrays"""
        data_array = np.asarray(data)
        
        if data_array.ndim == 0 or (data_array.ndim == 1 and len(data_array) <= 4):
            # Single value or small vector (position/orientation) - broadcast to all environments
            if dtype == wp.vec3f:
                buffer.fill_(wp.vec3f(*data_array))
            elif dtype == wp.quatf:
                buffer.fill_(wp.quatf(*data_array))
            elif dtype == wp.float32:
                buffer.fill_(float(data_array))
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
        else:
            # Array of values - one per environment
            temp_array = wp.array(data, device=self._device, dtype=dtype)
            wp.copy(buffer, temp_array)

    def _reset_subsystems(self):
        """Reset history and subsystems"""
        self.current_step = 0
        
        # Reset GPU history buffers if history saving is enabled
        if self._save_history:
            self.history_alpha.zero_()
            self.history_beta.zero_()
            self.history_Va.zero_()
            self.history_hr.zero_()
            self.history_Fb.zero_()
            self.history_Mb.zero_()
            self.history_Fb_aero.zero_()
            self.history_Mb_aero.zero_()
            self.history_Fb_thrust.zero_()
            self.history_Mb_thrust.zero_()
            self.history_Fb_g.zero_()
            self.history_C_L.zero_()
            self.history_C_D.zero_()
            self.history_C_L_free.zero_()
            self.history_C_D_free.zero_()
            self.history_Cl.zero_()
            self.history_Cm.zero_()
            self.history_Cn.zero_()
            self.history_elevator.zero_()
            self.history_aileron.zero_()
            self.history_rudder.zero_()
            self.history_throttle_left.zero_()
            self.history_throttle_right.zero_()
            self.history_position.zero_()
            self.history_orientation.zero_()
            self.history_linear_vel.zero_()
            self.history_angular_vel.zero_()

        if self.sensor_system is not None:
            self.sensor_system.reset()

        if self.wind_model is not None:
            self.wind_model.reset()

    def get_sensor_measurements(self):
        """Get sensor measurements from the sensor system"""
        if self.sensor_system is None:
            return None
        
        # Sense all sensors with current state
        self.sensor_system.sense_all_sensors(self._state, self.EP.dt)
        
        # Get the measurements
        return self.sensor_system.get_sensor_measurements()

    def step(self, action=None, return_sensors: bool = False):
        """
        Step the model using single fused kernel
        """
        if action is not None:
            for key in ["elevator", "aileron", "rudder", "throttle_left", "throttle_right"]:
                if key in action:
                    if isinstance(action[key], (int, float, np.integer, np.floating)):
                        # Single scalar value - replicate for all environments
                        action_value = float(action[key])
                        self._action_command[key].fill_(action_value)
                    else:
                        # Array of values - one per environment
                        action_array = wp.array(action[key], device=self._device, dtype=wp.float32)
                        wp.copy(self._action_command[key], action_array)
        else:
            # Set constant action
            self._action_command["elevator"].fill_(0.0)
            self._action_command["aileron"].fill_(0.0)
            self._action_command["rudder"].fill_(0.0)
            self._action_command["throttle_left"].fill_(-1.0)
            self._action_command["throttle_right"].fill_(-1.0)

        # Update wind model
        if self.wind_model is not None:
            self.wind_model.update_wind(self._state["orientation"], self._Va, self._state["position"])

        # Single fused kernel launch for entire simulation step
        wp.launch(
            kernel=aircraft_simulation_step,
            dim=self._num_envs,
            inputs=[
                # State arrays
                self._state["position"],
                self._state["linear_vel"],
                self._state["angular_vel"],
                self._state["orientation"],
                
                # Action commands
                self._action_command["elevator"],
                self._action_command["aileron"],
                self._action_command["rudder"],
                self._action_command["throttle_left"],
                self._action_command["throttle_right"],
                
                # Actuator states
                self._actuator_states["elevator"],
                self._actuator_states["aileron"],
                self._actuator_states["rudder"],
                self._actuator_states["throttle_left"],
                self._actuator_states["throttle_right"],
                self._actuator_states["elevator_dot"],
                self._actuator_states["aileron_dot"],
                self._actuator_states["rudder_dot"],
                
                # Aerodynamic parameters
                self._Va,
                self._alpha,
                self._beta,
                self._rho,
                self._Q,
                self._hr,
                self._airspeed_vector,
                
                # Coefficient arrays (temporary)
                self._C_D,
                self._C_Y,
                self._C_L,
                self._Cl,
                self._Cm,
                self._Cn,
                self._C_D_free,
                self._C_L_free,

                # Force/moment arrays
                self._D_tot,
                self._Y_tot,
                self._L_tot,
                self._F1_b,
                self._F2_b,
                self._Fb_thrust,
                self._M_prop_left,
                self._M_prop_right,
                self._Mb_thrust,
                self._Fw_aero,
                self._Fb_aero,
                self._Mb_aero,
                self._q_aero,
                self._gravity_vector_body,
                self._Fb_g,
                self._Fb,
                self._Mb,
                
                # Wind effects
                self.wind_model.total_linear_gusts if self.wind_model else wp.zeros((self._num_envs), device=self._device, dtype=wp.vec3f),
                
                # Configuration parameters
                self.VP.Mac,
                self.EP.dt,
                self.EP.g,
                self.VP.m,
                self._ge_enabled,  # in_ground_effect
                
                # Configuration structs
                self.AP,
                self.VP.WP,
                self.VP.J,
                self.VP.PP,
                
                # Actuator parameters
                self.VP.T_s,
                self.VP.zeta,
                self.VP.omega_0,
                self.CL.elevator_limits,
                self.CL.aileron_limits,
                self.CL.rudder_limits,
                self.CL.throttle_limits,
                
                # Solver type
                wp.int32(self.solver_type)
            ],
            device=self._device
        )

        # Save history on GPU (no CPU transfer)
        if self._save_history and self.current_step < self._max_steps:
            env_idx = self._env_to_save[0] if isinstance(self._env_to_save, list) else self._env_to_save
            
            wp.launch(
                kernel=save_history_kernel,
                dim=1,
                inputs=[
                    self.current_step,
                    env_idx,
                    self._alpha, self._beta, self._Va, self._hr,
                    self._Fb, self._Mb, self._Fb_aero, self._Mb_aero,
                    self._Fb_thrust, self._Mb_thrust, self._Fb_g,
                    self._C_L, self._C_D, self._C_L_free, self._C_D_free,
                    self._Cl, self._Cm, self._Cn,
                    self._actuator_states["elevator"],
                    self._actuator_states["aileron"],
                    self._actuator_states["rudder"],
                    self._actuator_states["throttle_left"],
                    self._actuator_states["throttle_right"],
                    self._state["position"],
                    self._state["orientation"],
                    self._state["linear_vel"],
                    self._state["angular_vel"],
                    self.history_alpha, self.history_beta, self.history_Va, self.history_hr,
                    self.history_Fb, self.history_Mb,
                    self.history_Fb_aero, self.history_Mb_aero,
                    self.history_Fb_thrust, self.history_Mb_thrust, self.history_Fb_g,
                    self.history_C_L, self.history_C_D, self.history_C_L_free, self.history_C_D_free,
                    self.history_Cl, self.history_Cm, self.history_Cn,
                    self.history_elevator, self.history_aileron, self.history_rudder,
                    self.history_throttle_left, self.history_throttle_right,
                    self.history_position, self.history_orientation,
                    self.history_linear_vel, self.history_angular_vel
                ],
                device=self._device
            )
        
        self.current_step += 1

        # Handle sensors
        if return_sensors and self.sensor_system is not None:
            sensor_measurements = self.get_sensor_measurements()
            return self._state, sensor_measurements
        else:
            return self._state, None
        
    def seed(self, seed=None):
        """Seed random number generator for all WARP systems"""
        if seed is not None:
            self.warp_seed = wp.int32(seed)
            np.random.seed(seed)
            self.np_random = np.random.RandomState(seed)
        else:
            random_seed = random.randint(0, 2**29)
            self.warp_seed = wp.int32(random_seed)
            np.random.seed(random_seed)
            self.np_random = np.random.RandomState(random_seed)

        # Seed subsystems
        if hasattr(self, 'sensor_system') and self.sensor_system is not None:
            self.sensor_system.seed((self.warp_seed.value if seed is not None else random_seed) + 1000)
            
        if hasattr(self, 'wind_model') and self.wind_model is not None:
            self.wind_model.seed((self.warp_seed.value if seed is not None else random_seed) + 2000)    

    def check_termination(self, alpha_max: float):
        """
        Check termination conditions on GPU and return flags.
        Returns: (crashed: bool, stalled: bool)
        Only transfers 2 integers (8 bytes) from GPU.
        """
        env_idx = self._env_to_save[0] if isinstance(self._env_to_save, list) else self._env_to_save
        
        # Reset flags to 0
        self._crashed_flag.fill_(0)
        self._stalled_flag.fill_(0)
        
        # Run termination check kernel
        wp.launch(
            kernel=check_termination_kernel,
            dim=1,
            inputs=[
                self._state["position"],
                self._alpha,
                env_idx,
                alpha_max,
                self._crashed_flag,
                self._stalled_flag
            ],
            device=self._device
        )
        
        # Transfer only the flags (8 bytes total)
        crashed = bool(self._crashed_flag.numpy()[0])
        stalled = bool(self._stalled_flag.numpy()[0])
        
        return crashed, stalled

    def get_history_data(self):
        """Transfer ALL history at once (call after simulation ends)"""
        if not self._save_history:
            return None
        
        n = min(self.current_step, self._max_steps)
        
        # Single batch transfer
        aero_params = {
            'alpha': self.history_alpha.numpy()[:n],
            'beta': self.history_beta.numpy()[:n],
            'Va': self.history_Va.numpy()[:n],
            'height ratio': self.history_hr.numpy()[:n],
            'Coef_f': np.column_stack([
                self.history_C_L_free.numpy()[:n],
                self.history_C_D_free.numpy()[:n]
            ]),
            'Coef_m': np.column_stack([
                self.history_Cl.numpy()[:n],
                self.history_Cm.numpy()[:n],
                self.history_Cn.numpy()[:n]
            ]),
            'oge_ige': np.column_stack([
                self.history_C_L.numpy()[:n],
                self.history_C_D.numpy()[:n],
                self.history_C_L_free.numpy()[:n],
                self.history_C_D_free.numpy()[:n]
            ])
        }
        
        forces = {
            'Fb': self.history_Fb.numpy()[:n],
            'Mb': self.history_Mb.numpy()[:n],
            'F_aero': self.history_Fb_aero.numpy()[:n],
            'M_aero': self.history_Mb_aero.numpy()[:n],
            'F_thrust': self.history_Fb_thrust.numpy()[:n],
            'M_thrust': self.history_Mb_thrust.numpy()[:n],
            'Fb_g': self.history_Fb_g.numpy()[:n]
        }
        
        actions = {
            'elevator': self.history_elevator.numpy()[:n],
            'aileron': self.history_aileron.numpy()[:n],
            'rudder': self.history_rudder.numpy()[:n],
            'throttle_left': self.history_throttle_left.numpy()[:n],
            'throttle_right': self.history_throttle_right.numpy()[:n]
        }
        
        states = {
            'position': self.history_position.numpy()[:n],
            'orientation': self.history_orientation.numpy()[:n],
            'linear_vel': self.history_linear_vel.numpy()[:n],
            'angular_vel': self.history_angular_vel.numpy()[:n]
        }
        
        return {'aero_params': aero_params, 'forces': forces, 'actions': actions, 'states': states}