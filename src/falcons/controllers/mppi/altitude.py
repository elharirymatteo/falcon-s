"""MPPI on the altitude-acquisition scenario: the classical MPPI row of the altitude table.

falcons.benchmark.altitude runs this in a child process and bakes the reference constants
(TRAJ_CONST_ALT, TRAJ_ACQ_H0, TRAJ_REF_VA) into that child's environment before importing this
module: falcons.controllers.mppi.reference freezes them into warp constants at import time.
"""
import numpy as np

from falcons.aircraft.params import load_params
from falcons.controllers.lqr.reference import compute_reference as compute_reference_np
from falcons.controllers.mppi.control import MPPIAltitudeControl
from falcons.controllers.mppi.reference import compute_reference
from falcons.sim.warp.aircraft import Aircraft


class MPPIAltitude:
    """Adapter for running MPPI control scenarios."""

    def __init__(self, aircraft_name: str = "Airship_V7", init_altitude: float = None,
                 mppi_seed: int = 0, samples: int = 1000, horizon: int = 100):
        self.aircraft_name = aircraft_name
        # Height (m) to spawn at. None = spawn ON the reference at t=0, which is what makes the
        # comparison against the other controllers fair: the previous hardcoded 1 m spawn put the
        # aircraft 9 m below a ramp_ascent reference that starts at 10 m, so the run was dominated
        # by (and usually stalled during) the catch-up climb rather than measuring tracking.
        self.init_altitude = init_altitude
        self.mppi_seed = mppi_seed
        self.samples = samples
        self.horizon = horizon

    def run(self, scenario: dict) -> dict:
        """
        Run MPPI scenario and return only data needed for metrics.
        
        Returns:
            dict with keys: 'time', 'positions', 'references', 'actions'
        """
        trajectory_type = scenario['trajectory_type']
        use_sensors = scenario['use_sensor_noise']
        
        # Load configuration from JSON file
        try:
            config_objects = load_params(self.aircraft_name)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return {
                'time': np.array([]),
                'positions': np.array([]),
                'references': np.array([]),
                'actions': np.array([])
            }
        
        # Extract configuration objects
        AP = config_objects['aero_params']
        VP = config_objects['vehicle_params']
        CL = config_objects['control_limits']
        EP = config_objects['environment_params']
        
        # Convert to WARP structs
        AP = AP.as_warp_struct()
        VP.WP = VP.WP.as_warp_struct()
        VP.J = VP.J.as_warp_struct()
        VP.PP = VP.PP.as_warp_struct()
        
        config = {'AP': AP, 'VP': VP, 'CL': CL, 'EP': EP}
        
        # Initialize the real simulated aircraft
        aircraft = Aircraft(1, "cuda", config, save_history=False)
        
        # MPPI specs. The preview horizon (iterations x dt) is the planner's main tuning knob on
        # the acquisition task: a 1 s preview cannot see far enough ahead to arrest a commanded
        # descent, so it dives to catch the reference ramp and overshoots. Sweep it with the
        # `horizon` constructor argument.
        mppi_specs = {
            "number_of_sampled_trajectories": int(self.samples),
            "number_of_iterations_per_sample": int(self.horizon),
            "noise_variance_elevator": 0.10,
            "noise_variance_aileron": 0.08,
            "noise_variance_rudder": 0.08,
            "noise_variance_throttle_both": 0.10,
            "temperature": 3.0
        }
        
        # Define trajectory targets
        targets = {
            'trajectory_type': trajectory_type,
        }
        
        # Disable sensors and wind for the MPPI sampled trajectories
        if hasattr(VP, 'sensor_system'):
            VP.sensor_system = None
        if hasattr(EP, 'constant_wind'):
            EP.constant_wind = (0.0, 0.0, 0.0)
        if hasattr(EP, 'turbulence') and EP.turbulence:
            EP.turbulence = {'enable': False}
        
        config_MPPI = {'AP': AP, 'VP': VP, 'CL': CL, 'EP': EP}
        
        # Initialize the MPPI controller
        MPPI_controller = MPPIAltitudeControl(mppi_specs, targets, config_MPPI, save_history=False)
        # Pin the sampling noise: the controller's RNG is otherwise seeded from entropy by the
        # warp aircraft base whenever nobody calls seed(). This removes one source of run-to-run
        # variation but NOT all of it -- the cost reduction in falcons/controllers/mppi/kernels.py
        # sums rollout costs with wp.atomic_add, whose ordering across threads is not
        # deterministic, so identical seeds still yield slightly different plans. MPPI rows must
        # therefore be reported over repetitions, not as a single rollout; `mppi_seed` picks the
        # realization (None restores the historical entropy seeding).
        if self.mppi_seed is not None:
            MPPI_controller.seed(int(self.mppi_seed))
        
        # Initial state and action — spawn on the reference (see __init__) unless overridden
        z0 = -self.init_altitude if self.init_altitude is not None \
            else float(compute_reference_np(0.0, trajectory_type)[2])
        # Spawn at the airframe's own trim airspeed. The historical hardcoded 30 m/s is
        # roughly the V7's trim but TWICE the Volantex Ranger's, so on that airframe the
        # run measured a deceleration transient from double trim, not the controller.
        init_state = {
            "position": np.array([0, 0, z0], dtype=float),
            "linear_vel": np.array(config_objects["default_initial_state"].linear_vel,
                                   dtype=float),
            "angular_vel": np.array([0, 0, 0], dtype=float),
            "orientation": np.array([0, 0, 0, 1], dtype=float) 
        }
        init_action = {
            "elevator": 0.0,
            "aileron": 0.0,
            "rudder": 0.0,
            "throttle_left": 0.60,   
            "throttle_right": 0.60,
            "elevator_dot": 0.0,
            "aileron_dot": 0.0,
            "rudder_dot": 0.0
        }
        
        # Reset the aircraft and the MPPI controller
        aircraft.reset(init_state, init_action)
        MPPI_controller.reset(init_state, init_action)
        
        # Pre-allocate arrays
        max_steps = 10000
        times = np.zeros(max_steps)
        positions = np.zeros((max_steps, 3))
        references = np.zeros((max_steps, 3))
        actions = np.zeros((max_steps, 5))
        
        print(f"\nRunning MPPI: {scenario['name']}")
        
        # Run simulation
        for step in range(max_steps):
            # MPPI step to get the next action (advances internal simulation time)
            action = MPPI_controller.MPPI_step()
            
            # Compute the reference trajectory using MPPI's internal time
            # Returns vec3f(y_lateral, z_altitude, forward_velocity)
            ref_traj = compute_reference(MPPI_controller.simulation_time, MPPI_controller.trajectory_type_id)
            
            # Apply the action and get sensor measurements
            state, sensor_measurements = aircraft.step(action, use_sensors)
            
            # Use measured or true state
            if use_sensors:
                current_state = {
                    "position": sensor_measurements['gps_position'].numpy()[0],
                    "linear_vel": sensor_measurements['gps_velocity'].numpy()[0],
                    "angular_vel": sensor_measurements['gyroscope'].numpy()[0],
                    "orientation": sensor_measurements['attitude_sensor'].numpy()[0]
                }
            else:
                current_state = {
                    "position": state['position'].numpy()[0],
                    "linear_vel": state['linear_vel'].numpy()[0],
                    "angular_vel": state['angular_vel'].numpy()[0],
                    "orientation": state['orientation'].numpy()[0]
                }
            
            # Map actions to standard format: [throttle_left, throttle_right, elevator, aileron, rudder]
            mapped_actions = [
                action.get('throttle_left', 0),
                action.get('throttle_right', 0),
                action.get('elevator', 0),
                action.get('aileron', 0),
                action.get('rudder', 0)
            ]
            
            # Save data for metrics
            times[step] = MPPI_controller.simulation_time
            positions[step] = current_state['position']
            references[step] = [current_state['position'][0], ref_traj.x, ref_traj.y]
            actions[step] = mapped_actions
            
            # Update MPPI with new state and action from the actual aircraft
            new_init_state = current_state
            new_init_action = {
                "elevator": aircraft._actuator_states['elevator'].numpy()[0],
                "aileron": aircraft._actuator_states['aileron'].numpy()[0],
                "rudder": aircraft._actuator_states['rudder'].numpy()[0],
                "throttle_left": aircraft._actuator_states['throttle_left'].numpy()[0],
                "throttle_right": aircraft._actuator_states['throttle_right'].numpy()[0],
                "elevator_dot": aircraft._actuator_states['elevator_dot'].numpy()[0],
                "aileron_dot": aircraft._actuator_states['aileron_dot'].numpy()[0],
                "rudder_dot": aircraft._actuator_states['rudder_dot'].numpy()[0]
            }
            
            # Check termination conditions using GPU-accelerated check
            crashed, stalled = aircraft.check_termination(VP.alpha_max)
            
            if crashed:
                print(f"Simulation ended at step {step}: Crashed into water")
                actual_steps = step + 1
                break
            if stalled:
                print(f"Simulation ended at step {step}: Stall angle exceeded")
                actual_steps = step + 1
                break
            
            # Reset MPPI controller with updated state for next iteration
            MPPI_controller.reset(new_init_state, new_init_action)
        else:
            actual_steps = max_steps
        
        # Return same format as LQR
        return {
            'time': times[:actual_steps],
            'positions': positions[:actual_steps],
            'references': references[:actual_steps],
            'actions': actions[:actual_steps]
        }
