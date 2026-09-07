import numpy as np
import warp as wp
from falcons.sim.warp.aircraft import Aircraft

from falcons.controllers.mppi.kernels import (
    _sample_inputs_sequences, 
    _convert_inputs_sequences_into_trajectories, 
    _evaluate_trajectories, 
    # _compute_weights,
    _compute_min_cost, 
    _compute_eta,
    _compute_final_weights,
    _compute_weighted_average, 
    _shift_sequence
)
from falcons.controllers.mppi.reference import get_trajectory_type_id




class MPPIAltitudeControl(Aircraft):

    def __init__(self, MPPI_specs, targets, config, save_history=False):
        self.extract_MPPI_specs(MPPI_specs)

        super().__init__(self.number_of_sampled_trajectories, 
                         "cuda", 
                         config,
                         save_history=save_history)

        self.targets = targets
        self.dt = self.EP.dt  

        # Unpack config
        self.AP = config['AP']
        self.VP = config['VP']
        self.CL = config['CL']
        self.EP = config['EP']

        # Initialize the state of the MPPI controller
        self.build_MPPI_sequences_buffers()
        self.build_MPPI_samples_data_buffers()
        self.build_MPPI_material_buffers()
        
        # Initialize the reference generator
        self.trajectory_type_id = get_trajectory_type_id(targets.get('trajectory_type', 'constant_altitude'))
        self.simulation_time = 0.0

    def extract_MPPI_specs(self, MPPI_specs):
        self.number_of_sampled_trajectories = MPPI_specs["number_of_sampled_trajectories"]
        self.number_of_iterations_per_sample = MPPI_specs["number_of_iterations_per_sample"]
        self.noise = {
            "elevator": MPPI_specs["noise_variance_elevator"],           
            "aileron": MPPI_specs["noise_variance_aileron"],           
            "rudder": MPPI_specs["noise_variance_rudder"],               
            "throttle": MPPI_specs["noise_variance_throttle_both"]
        }
        self.temperature = MPPI_specs["temperature"]

    def build_MPPI_sequences_buffers(self):
        self.optimal_sequence = {}
        self.optimal_sequence["elevator"] = wp.zeros((self.number_of_iterations_per_sample), device=self._device, dtype=wp.float32)
        self.optimal_sequence["aileron"] = wp.zeros((self.number_of_iterations_per_sample), device=self._device, dtype=wp.float32)
        self.optimal_sequence["rudder"] = wp.zeros((self.number_of_iterations_per_sample), device=self._device, dtype=wp.float32)
        self.optimal_sequence["throttle"] = wp.zeros((self.number_of_iterations_per_sample), device=self._device, dtype=wp.float32)

        self.sampled_sequences = {}
        self.sampled_sequences["elevator"] = wp.zeros((self.number_of_sampled_trajectories * self.number_of_iterations_per_sample), device=self._device, dtype=wp.float32)
        self.sampled_sequences["aileron"] = wp.zeros((self.number_of_sampled_trajectories * self.number_of_iterations_per_sample), device=self._device, dtype=wp.float32)
        self.sampled_sequences["rudder"] = wp.zeros((self.number_of_sampled_trajectories * self.number_of_iterations_per_sample), device=self._device, dtype=wp.float32)
        self.sampled_sequences["throttle"] = wp.zeros((self.number_of_sampled_trajectories * self.number_of_iterations_per_sample), device=self._device, dtype=wp.float32)

    def build_MPPI_samples_data_buffers(self):
        self.position_memory = wp.zeros((self.number_of_sampled_trajectories * self.number_of_iterations_per_sample), device=self._device, dtype=wp.vec3f)
        self.linear_vel_memory = wp.zeros((self.number_of_sampled_trajectories * self.number_of_iterations_per_sample), device=self._device, dtype=wp.vec3f)
        self.angular_vel_memory = wp.zeros((self.number_of_sampled_trajectories * self.number_of_iterations_per_sample), device=self._device, dtype=wp.vec3f)
        self.orientation_memory= wp.zeros((self.number_of_sampled_trajectories * self.number_of_iterations_per_sample), device=self._device, dtype=wp.quatf)
        self.alpha_memory = wp.zeros((self.number_of_sampled_trajectories * self.number_of_iterations_per_sample), device=self._device, dtype=wp.float32)
        self.beta_memory = wp.zeros((self.number_of_sampled_trajectories * self.number_of_iterations_per_sample), device=self._device, dtype=wp.float32)

    def build_MPPI_material_buffers(self):
        self.costs = wp.zeros((self.number_of_sampled_trajectories), device=self._device, dtype=wp.float32)
        self.bias_cost_term = wp.zeros((self.number_of_sampled_trajectories), device=self._device, dtype=wp.float32)
        self.min_cost = wp.array([np.inf], dtype=float, device=self._device)
        self.weights = wp.zeros(self.number_of_sampled_trajectories, dtype=float, device=self._device)
        self.eta = wp.zeros(1, dtype=float, device=self._device) 

    def reset(self, init_state: dict = None, init_action: dict = None):
        """Reset MPPI controller with new initial state and action"""
        # Call parent reset first to set the actuator states
        super().reset(init_state, init_action)

        # Initialize optimal sequence with normalized init_action values
        if init_action is not None:
            # Helper function to normalize physical values to [-1, 1] range
            def normalize_to_command(physical_value, limits_vec2):
                lower = limits_vec2.x
                upper = limits_vec2.y
                return 2.0 * (physical_value - lower) / (upper - lower) - 1.0
            
            # Normalize the initial action values to [-1, 1] range
            elevator_normalized = normalize_to_command(init_action["elevator"], self.CL.elevator_limits)
            aileron_normalized = normalize_to_command(init_action["aileron"], self.CL.aileron_limits)
            rudder_normalized = normalize_to_command(init_action["rudder"], self.CL.rudder_limits)
            throttle_normalized = normalize_to_command(init_action["throttle_left"], self.CL.throttle_limits)
            
            # Fill optimal sequence with normalized values
            self.optimal_sequence["elevator"].fill_(elevator_normalized)
            self.optimal_sequence["aileron"].fill_(aileron_normalized)
            self.optimal_sequence["rudder"].fill_(rudder_normalized)
            self.optimal_sequence["throttle"].fill_(throttle_normalized)

        # Reset MPPI material buffers
        self.reset_MPPI_material_buffers()


    def reset_MPPI_material_buffers(self):
        self.costs.zero_()
        self.bias_cost_term.zero_()
        self.min_cost = wp.array([np.inf], dtype=float, device=self._device)
        self.weights.zero_()
        self.eta.zero_()

    def sample_inputs_sequences(self):

        for (key1, previous_action), (key2, noise_std_dev), (key3, noisy_action) in zip(self.optimal_sequence.items(),
                                                                                        self.noise.items(),
                                                                                        self.sampled_sequences.items()):   
            wp.launch(
                kernel=_sample_inputs_sequences,
                dim=self.number_of_sampled_trajectories*self.number_of_iterations_per_sample,
                inputs=[
                    self.number_of_iterations_per_sample,
                    int(self.np_random.uniform(0, 1e5)),
                    previous_action,
                    noise_std_dev,
                    noisy_action,
                    self.bias_cost_term,
                    self.temperature
                ],
                device=self._device
            )




    def convert_inputs_sequences_into_trajectories(self):

        wp.launch(
            kernel=_convert_inputs_sequences_into_trajectories,
            dim=self.number_of_sampled_trajectories,
            inputs=[
                self.number_of_iterations_per_sample,
                self._state["position"],
                self._state["linear_vel"],
                self._state["angular_vel"],
                self._state["orientation"],
                self.position_memory,
                self.linear_vel_memory,
                self.angular_vel_memory,
                self.orientation_memory,
                self.alpha_memory,
                self.beta_memory,
                self.VP.Mac,
                self._rho,
                self._Va,
                self._alpha,
                self._beta,
                self._Q,
                self._hr,
                self.sampled_sequences["elevator"],
                self.sampled_sequences["aileron"],
                self.sampled_sequences["rudder"],
                self.AP.alpha_exp,
                self.AP.beta_exp,
                self.AP.elevator_exp,
                self.AP.aileron_exp,
                self.AP.rudder_exp,
                self.AP.CD_coefs,
                self.AP.CL_coefs,
                self._C_L,
                self._C_D,
                self.VP.WP,
                self._D_tot,
                self._L_tot,
                self._in_ground_effect,
                self.AP.CY_coefs,
                self._C_Y,
                self._Y_tot,
                self.AP.CMx_coefs,
                self.AP.CMy_coefs,
                self.AP.CMz_coefs,
                self._Cl,
                self._Cm,
                self._Cn,
                self._Mb_aero,
                self._Fw_aero,
                self._Fb_thrust,
                self._Mb_thrust,
                self.EP.g,
                self.VP.m,
                self._q_aero,
                self._Fb_aero,
                self._gravity_vector_body,
                self._Fb_g,
                self._Fb,
                self.VP.J,
                self._Mb,
                self.dt,
                self.sampled_sequences["throttle"],
                self.VP.PP,
                self._F1_b,
                self._F2_b,
                self._M_prop_left,
                self._M_prop_right,
                self.VP.T_s,
                self.VP.zeta,
                self.VP.omega_0,
                self.CL.elevator_limits,
                self.CL.aileron_limits,
                self.CL.rudder_limits,
                self.CL.throttle_limits,
                self._actuator_states["elevator"],
                self._actuator_states["aileron"],
                self._actuator_states["rudder"],
                self._actuator_states["throttle_left"],
                self._actuator_states["throttle_right"],
                self._actuator_states["elevator_dot"],
                self._actuator_states["aileron_dot"],
                self._actuator_states["rudder_dot"],
                self.AP.Clp,
                self.AP.Cmq,
                self.AP.Cnr,
            ],
            device=self._device
        )



    def evaluate_trajectories(self):
        wp.launch(
            kernel=_evaluate_trajectories,
            dim=self.number_of_sampled_trajectories * self.number_of_iterations_per_sample,
            inputs=[
                self.number_of_iterations_per_sample,
                self.position_memory,
                self.linear_vel_memory,
                self.angular_vel_memory,
                self.orientation_memory,
                self.alpha_memory,
                self.beta_memory,
                self.costs,
                self.simulation_time,    
                self.dt,                   
                self.trajectory_type_id  
            ],
            device=self._device
        )
        # print(self.costs.numpy()[:5])


    def compute_weights(self):
        
        # wp.launch(
        #     kernel=_compute_weights,
        #     dim=self.number_of_sampled_trajectories,
        #     inputs=[
        #         self.costs,
        #         self.min_cost,
        #         self.weights,
        #         self.temperature,
        #         self.bias_cost_term,
        #         self.eta, 
        #     ],
        #     device=self._device
        # )
        # print("Weights sum:", self.weights.numpy().sum())

        # Step 1: Find minimum cost
        wp.launch(
            kernel=_compute_min_cost,
            dim=self.number_of_sampled_trajectories,
            inputs=[self.costs, self.min_cost],
            device=self._device
        )
        
        # Set bias to all zeros
        # self.bias_cost_term = wp.zeros((self.number_of_sampled_trajectories), device=self._device, dtype=wp.float32)
        
        # Step 2: Compute eta (normalization factor)
        wp.launch(
            kernel=_compute_eta,
            dim=self.number_of_sampled_trajectories,
            inputs=[
                self.costs,
                self.min_cost,
                self.bias_cost_term,
                self.temperature,
                self.eta
            ],
            device=self._device
        )
        
        # Step 3: Compute final weights
        wp.launch(
            kernel=_compute_final_weights,
            dim=self.number_of_sampled_trajectories,
            inputs=[
                self.costs,
                self.min_cost,
                self.bias_cost_term,
                self.temperature,
                self.eta,
                self.weights
            ],
            device=self._device
        )
        # print("Weights sum:", self.weights.numpy().sum())


    def compute_weighted_average(self):

        for key in self.optimal_sequence:
            self.optimal_sequence[key].zero_()

        for (key1, sampled_actions), (key2, optimal_action) in zip(self.sampled_sequences.items(),
                                                                   self.optimal_sequence.items()):   
            wp.launch(
                kernel=_compute_weighted_average,
                dim=self.number_of_sampled_trajectories * self.number_of_iterations_per_sample,
                inputs=[
                    self.number_of_iterations_per_sample,
                    self.weights,
                    sampled_actions,
                    optimal_action,
                ],
                device=self._device
            )



    def shift_sequence(self):

        wp.launch(
            kernel=_shift_sequence,
            dim=self.number_of_iterations_per_sample - 1,
            inputs=[
                # self.number_of_iterations_per_sample,
                self.optimal_sequence["elevator"],
                self.optimal_sequence["aileron"],
                self.optimal_sequence["rudder"],
                self.optimal_sequence["throttle"]
            ],
            device=self._device
        )



    def MPPI_step(self):
        self.reset_MPPI_material_buffers()
        self.shift_sequence()
        self.sample_inputs_sequences()
        self.convert_inputs_sequences_into_trajectories()
        self.evaluate_trajectories()
        self.compute_weights()
        self.compute_weighted_average()
        
        # Advance simulation time
        self.simulation_time += self.dt
        
        optimal_action = {}
        optimal_action["elevator"] = self.optimal_sequence["elevator"].numpy()[0]
        optimal_action["aileron"] = self.optimal_sequence["aileron"].numpy()[0]
        optimal_action["rudder"] = self.optimal_sequence["rudder"].numpy()[0]
        optimal_action["throttle_left"] = self.optimal_sequence["throttle"].numpy()[0]
        optimal_action["throttle_right"] = self.optimal_sequence["throttle"].numpy()[0]

        return optimal_action
