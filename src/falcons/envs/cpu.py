import numpy as np
from falcons.sim.cpu.aircraft import Aircraft

import os
import uuid
from falcons.envs.trajectory import Trajectory
from pathlib import Path
from falcons.paths import REPO_DIR


class CpuEnv:
    """
    Core Environment for the Airship Model. This class handles the essential elements of the airship control and
    dynamics, leaving specific task parameters and spaces to be defined in derived classes.
    """

    def __init__(
        self,
        aircraft_name: str = None,
        save_history: bool = False,
        save_trajectories: bool = False,
        trajectories_path: str | Path = REPO_DIR / "logs" / "trajectories",
    ):
        """
        Initialize core environment

        Args:
            aircraft_name: Name of aircraft (e.g., 'Airship_V7')
        """
        ## Unique Environment ID Assignment
        self.env_id = str(uuid.uuid4())[:8]

        self.airship = Aircraft(
            aircraft_name=aircraft_name,
            save_history=save_history,
        )
        self.horizon = 10000  # Maximum number of steps in an episode 100 seconds
        self.steps = 0
        self.reward = 0
        self.save_history = save_history
        self.info = {
            "avg_actions": None,
            "avg_state_metrics": None,
            "termination_reasons": None,
            "ema_reward_components": None,
        }
        self.rng = np.random.default_rng()
        self.episode_counter = 0

        # --- Trajectory Tracking Initialization ---
        self.save_trajectories = save_trajectories
        if self.save_trajectories:
            self.trajectories_path = trajectories_path
            self.trajectory = Trajectory(model_name="airship", dt=0.01)

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Increment episode counter for the new episode that is about to start.
        self.episode_counter += 1

        # Reset trajectory object to start recording the new episode
        if self.save_trajectories:
            self.trajectory.clear()

        self.steps = 0
        self.reward = 0
        initial_state = self.initial_state()
        state = self.airship.reset(initial_state)
        self.info = {key: None for key in self.info.keys()}
        action = -np.ones(self.action_space.shape[0])

        return self._extract_state(state, action)

    def step(self, aero_action, motor_action):
        self.steps += 1
        # CHANGED: Use new airship interface directly
        next_state = self.airship.step(aero_action, motor_action)

        # CHANGED: Access history data differently
        aero_params = {}
        forces = {}
        if self.save_history:
            aero_params = self.airship.aero_params
            forces = self.airship.forces
            self.info["aero_params"] = aero_params
            self.info["forces"] = forces
        done = self.check_done()

        # CHANGED: Create full_action for compatibility
        full_action = np.concatenate([aero_action, motor_action])
        extracted_state = self._extract_state(next_state, full_action)

        ## Trajectory Recording and Saving Logic
        if self.save_trajectories:
            # Add data from this step to the trajectory
            self.trajectory.save_state(
                state=self.airship.state,
                action=full_action,
                aero_params=aero_params,
                forces=forces,
            )
            # If the episode is over, save the trajectory to a file
            if done:
                self._save_trajectory()

        return extracted_state, self.reward, done, False, self.info

    def close(self):
        """No-op, as it was on gymnasium's Env before this class stopped inheriting from it.
        LQR_Control.close() chains up to it at the end of every scenario."""

    def get_euler_angles_from_quaternion(self, q):
        """
        Convert quaternion to euler angles
        """
        q0, q1, q2, q3 = q
        phi = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1**2 + q2**2))
        theta = np.arcsin(2 * (q0 * q2 - q3 * q1))
        psi = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2**2 + q3**2))
        return phi, theta, psi

    def initial_state(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _extract_state(self, full_state, action=None):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def check_done(self):
        # Check if plane has crashed into water/ground
        if -self.airship.state["position"][2] + self.airship.VP['wing']['cg_offset_vector'][2] < 0:
            self.info["termination_reasons"] = "Crashed"
        # Check if plane has achieved the attack angle limit, stall also prevents the plane from flying backwards
        elif (
            hasattr(self.airship, 'alpha') and 
            np.abs(self.airship.alpha * 180 / np.pi) > 1.5 * self.airship.AP['stall_angle_deg']
        ):
            self.info["termination_reasons"] = "Stalled"
        # Check if plane has reached low airspeed and considered to be in a stall
        elif hasattr(self.airship, 'Va') and self.airship.Va < 10.0:
            self.info["termination_reasons"] = "Stalled"
        # Check if plane has reached the end of the horizon
        elif self.steps >= self.horizon:
            self.info["termination_reasons"] = "Horizon"

        return True if self.info.get("termination_reasons") is not None else False

    def _save_trajectory(self):
        if not self.save_trajectories or self.trajectory.is_empty:
            return

        filename = f"env_{self.env_id}_ep{self.episode_counter:04d}.pkl"
        filepath = os.path.join(self.trajectories_path, filename)

        self.trajectory.dump(filepath)
