"""LQR on the altitude-acquisition scenario: the classical LQR row of the altitude table.

falcons.benchmark.altitude runs this in a child process and bakes the reference constants
(TRAJ_CONST_ALT, TRAJ_ACQ_H0, read by falcons.controllers.lqr.reference) plus LQR_GAINS_PATH
(read by LQR_Control) into that child's environment before importing this module.
"""
import numpy as np

from falcons.controllers.lqr.control import LQR_Control
from falcons.controllers.lqr.reference import compute_reference as compute_reference_np


class LQRAltitude:
    """Adapter for running LQR control scenarios."""

    def __init__(self, aircraft_name: str = "Airship_V7", init_altitude: float = None):
        self.aircraft_name = aircraft_name
        # Height (m) to spawn at. None = spawn ON the reference at t=0. The airframe JSON's
        # default_initial_state puts every run at 1 m regardless of the commanded trajectory,
        # which left the aircraft 9 m below a ramp_ascent reference starting at 10 m (the run
        # then terminated in 0.4 s, and its RMSE was just the spawn gap).
        self.init_altitude = init_altitude

    def run(self, scenario: dict) -> dict:
        """
        Run LQR scenario and return only data needed for metrics.
        
        Returns:
            dict with keys: 'time', 'positions', 'references', 'actions'
        """
        env = self._create_environment(scenario)
        z0 = -self.init_altitude if self.init_altitude is not None \
            else float(compute_reference_np(0.0, scenario['trajectory_type'])[2])
        ip = dict(env.airship.IP)
        ip['position'] = (0.0, 0.0, z0)
        env.airship.IP = ip                      # consumed by LQR_Control.initial_state() on reset
        env.reset()
        
        # Pre-allocate arrays for efficiency
        max_steps = 10000
        times = np.zeros(max_steps)
        positions = np.zeros((max_steps, 3))
        references = np.zeros((max_steps, 3))
        actions = np.zeros((max_steps, 5))
        
        print(f"\nRunning LQR: {scenario['name']}")
        
        for step in range(max_steps):
            _, _, done, _, _, action = env.step()
            
            # Store only what metrics need
            times[step] = env.t
            positions[step] = env.airship.state['position']
            ref = env.target_ref()
            references[step] = [positions[step, 0], ref[1], ref[2]]
            actions[step] = self._format_action(action)
            
            if done:
                print(f"  Completed at step {step}")
                # Trim to actual length
                times = times[:step+1]
                positions = positions[:step+1]
                references = references[:step+1]
                actions = actions[:step+1]
                break
        
        env.close()
        
        return {
            'time': times,
            'positions': positions,
            'references': references,
            'actions': actions
        }
    
    def _create_environment(self, scenario: dict) -> LQR_Control:
        """Create LQR environment."""
        return LQR_Control(
            aircraft_name=self.aircraft_name,
            targets={'trajectory_type': scenario['trajectory_type']},
            k_i=np.array([0, -1, -1.5]),
            save_history=False,
            use_sensor_noise=scenario['use_sensor_noise'],
            use_estimator=scenario['use_estimator']
        )
    
    def _format_action(self, action: np.ndarray) -> np.ndarray:
        """Format action to [motor_left, motor_right, elevator, aileron, rudder]."""
        if len(action) >= 5:
            return action[:5]
        else:
            # Single motor case
            return np.array([action[0], 0.0, action[1], action[2], action[3]])
