import numpy as np
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import pickle


@dataclass
class Trajectory:
    """
    Buffer for storing the trajectory of the model.
    """ 
    model_name: str
    dt: float
    standard_quaternion: bool = True  # Default maintains backwards compatibility
    states: Dict[str, List[Any]] = field(default_factory=lambda: {'position': [], 'linear_vel': [], 'angular_vel': [], 'orientation': []})
    actions: Dict[str, List[float]] = field(default_factory=dict)
    aero_params: Dict[str, List[Any]] = field(default_factory=lambda: {'alpha': [], 'beta': [], 'Va': [], 'height ratio': [], 'Coef_f': [], 'Coef_m': [], 'oge_ige': []})
    forces: Dict[str, List[Any]] = field(default_factory=lambda: {'Fb': [], 'Mb': [], 'F_aero': [], 'M_aero': [], 'F_thrust': [], 'M_thrust': [], 'Fb_g': []})
    action_keys: Dict[str, list] = field(default_factory=dict)
    ref_trajectory: List[np.ndarray] = field(default_factory=list)  

    def __post_init__(self):
        # Ensure action_keys is a dict and all groups are present
        if not self.action_keys or not isinstance(self.action_keys, dict):
            raise ValueError(f"action_keys must be a dict for model: {self.model_name}")
        for group, keys in self.action_keys.items():
            if not isinstance(keys, list):
                raise ValueError(f"action_keys['{group}'] must be a list.")
            for key in keys:
                if key not in self.actions:
                    self.actions[key] = []
        # If no action keys at all, raise error
        if not any(self.action_keys.values()):
            raise ValueError(f"No action keys defined for model: {self.model_name}")

    def remove_last(self):
        """
        Remove the last state from the trajectory
        """
        for key in self.states.keys():
            if self.states[key]:
                self.states[key].pop()
        for key in self.actions.keys():
            if self.actions[key]:
                self.actions[key].pop()
        for key in self.aero_params.keys():
            if self.aero_params[key]:
                self.aero_params[key].pop()
        for key in self.forces.keys():
            if self.forces[key]:
                self.forces[key].pop()
        if self.ref_trajectory:
            self.ref_trajectory.pop()

    def save_state(self, state: Dict[str, Any], action: List[float], aero_params: Dict[str, Any], forces: Dict[str, Any], ref_trajectory: Optional[np.ndarray] = None):
        """
        Save the state, actions, aerodynamic parameters, forces, and optionally the reference trajectory (x, y, z).
        Converts Warp arrays to numpy, then applies quaternion conversion if needed.
        """
        def get_numpy_array(value):
            if hasattr(value, 'numpy'):  # It's a Warp array
                return value.numpy()[0]
            else:
                return value

        # Save states
        for key in self.states.keys():
            if key in state:
                # Step 1: Convert to numpy if it's a Warp array
                numpy_value = get_numpy_array(state[key])
                
                # Step 2: Apply quaternion conversion if needed
                if key == 'orientation' and len(numpy_value) == 4 and not self.standard_quaternion:
                    # Convert from Warp [x,y,z,w] to standard [w,x,y,z]
                    converted_quat = np.array([numpy_value[3], numpy_value[0], numpy_value[1], numpy_value[2]])
                    self.states[key].append(converted_quat)
                else:
                    # Keep as-is (either not quaternion, or standard_quaternion=True)
                    self.states[key].append(numpy_value)

        # Save actions (order: all groups in action_keys, in order)
        flat_keys = [key for group in self.action_keys.values() for key in group]
        for i, key in enumerate(flat_keys):
            if i < len(action):
                self.actions[key].append(action[i])

        # Save aerodynamic parameters
        if aero_params:
            for key in self.aero_params.keys():
                if key in aero_params:
                    self.aero_params[key].append(aero_params[key])

        # Save forces
        if forces:
            for key in self.forces.keys():
                if key in forces:
                    self.forces[key].append(forces[key])
        
        # Save reference trajectory if provided
        if ref_trajectory is not None:
            ref_numpy = np.array(ref_trajectory) if not isinstance(ref_trajectory, np.ndarray) else ref_trajectory
            self.ref_trajectory.append(ref_numpy)

    def save_to_file(self, save_dir=None, filename=None):
        """
        Save the trajectory to a file
        :param save_dir: Directory to save the file
        :param filename: File name
        """
        import pandas as pd
        import datetime
        try:
            # Define function to flatten arrays in lists
            def flatten_list(list_of_arrays):
                return [item.flatten() if isinstance(item, np.ndarray) and item.ndim > 1 else item for item in list_of_arrays]
            
            # Apply flattening to all values in the dictionaries
            flattened_states = {k: flatten_list(v) for k, v in self.states.items()}
            flattened_actions = {k: flatten_list(v) for k, v in self.actions.items()}
            flattened_aero_params = {k: flatten_list(v) for k, v in self.aero_params.items()}
            flattened_forces = {k: flatten_list(v) for k, v in self.forces.items()}
            
            # Combine all dataframes if they are not empty
            dataframes = [pd.DataFrame.from_dict(d) for d in [flattened_states, flattened_actions, flattened_aero_params, flattened_forces] if d]
            if not dataframes:
                print("No data to save.")
                return
            
            df = pd.concat(dataframes, axis=1)
            
            # Check if save_dir exists, otherwise create it
            if save_dir is None:
                save_dir = os.path.join(os.getcwd(), 'trajectories')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Create a filename if not provided
            if filename is None:
                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
            else:
                filename += '.csv'
            # Save dataframe to CSV
            full_path = os.path.join(save_dir, filename)
            df.to_csv(full_path, index=False)
            print(f"Trajectory saved to {full_path}")

        except Exception as e:
           print(f"An error occurred while saving the trajectory: {e}")

    def save_to_file_v2(self, save_dir=None, filename=None, flatten=False, labels=None):
        """
        Save the trajectory to a file, with options for saving arrays in comma-separated style
        or flattening all columns into individual values.
        :param save_dir: Directory to save the file
        :param filename: File name
        :param flatten: Whether to flatten columns into individual values
        :param labels: List of column names to apply when flattening
        """
        import pandas as pd
        import os
        import datetime
        import numpy as np

        try:
            # Helper function: Convert array to comma-separated string
            def array_to_comma_separated(arr):
                if isinstance(arr, np.ndarray):
                    return ', '.join(map(str, arr.flatten()))
                return arr

            # Helper function: Flatten and label arrays
            def flatten_and_label(dict_data, prefix):
                flat_dict = {}
                for key, values in dict_data.items():
                    if len(values) > 0:  # Check if the list is not empty
                        if isinstance(values[0], np.ndarray):  # Handle arrays
                            for i, subkey in enumerate(labels[:len(values[0].flatten())] if labels else range(len(values[0].flatten()))):
                                flat_dict[f"{prefix}_{key}_{subkey}"] = [
                                    v.flatten()[i] if isinstance(v, np.ndarray) else v
                                    for v in values
                                ]
                        else:  # Handle non-array values
                            flat_dict[f"{prefix}_{key}"] = values
                return flat_dict

            # Process states, actions, aero_params, and forces
            if flatten:
                flattened_states = flatten_and_label(self.states, "state")
                flattened_actions = flatten_and_label(self.actions, "action")
                flattened_aero_params = flatten_and_label(self.aero_params, "aero")
                flattened_forces = flatten_and_label(self.forces, "force")
            else:
                flattened_states = {k: [array_to_comma_separated(v) for v in values] for k, values in self.states.items() if len(values) > 0}
                flattened_actions = {k: [array_to_comma_separated(v) for v in values] for k, values in self.actions.items() if len(values) > 0}
                flattened_aero_params = {k: [array_to_comma_separated(v) for v in values] for k, values in self.aero_params.items() if len(values) > 0}
                flattened_forces = {k: [array_to_comma_separated(v) for v in values] for k, values in self.forces.items() if len(values) > 0}

            # Combine all dataframes if they are not empty
            dataframes = [pd.DataFrame.from_dict(d) for d in [flattened_states, flattened_actions, flattened_aero_params, flattened_forces] if len(d) > 0]
            if not dataframes:
                print("No data to save.")
                return

            df = pd.concat(dataframes, axis=1)

            # Check if save_dir exists, otherwise create it
            if save_dir is None:
                save_dir = os.path.join(os.getcwd(), 'trajectories')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Create a filename if not provided
            if filename is None:
                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
            else:
                filename += '.csv'

            # Save dataframe to CSV
            full_path = os.path.join(save_dir, filename)
            df.to_csv(full_path, index=False)
            print(f"Trajectory saved to {full_path}")

        except Exception as e:
            print(f"Error saving trajectory: {e}")


    def load_from_df(self, df):
        """
        Load the trajectory from a dataframe, assuming the columns are in the same order as the keys in the dictionaries.
        :param df: Dataframe to load the trajectory from
        """
        # Split the dataframe into separate dictionaries
        num_states = len(self.states)
        num_actions = len(self.actions)
        num_aero_params = len(self.aero_params)
        num_forces = len(self.forces)
        num_columns = num_states + num_actions + num_aero_params + num_forces
        state_cols = df.columns[:num_states]
        action_cols = df.columns[num_states:num_states+num_actions]
        aero_param_cols = df.columns[num_states+num_actions:num_states+num_actions+num_aero_params]
        force_cols = df.columns[num_states+num_actions+num_aero_params:num_columns]
        
        # Load the states
        for i, key in enumerate(self.states.keys()):
            self.states[key] = df[state_cols[i]].values.tolist()
        # Load the actions
        for i, key in enumerate(self.actions.keys()):
            self.actions[key] = df[action_cols[i]].values.tolist()
        # Load the aero_params
        for i, key in enumerate(self.aero_params.keys()):
            self.aero_params[key] = df[aero_param_cols[i]].values.tolist()
        # Load the forces
        for i, key in enumerate(self.forces.keys()):
            self.forces[key] = df[force_cols[i]].values.tolist()
        

    def clear(self):
        """
        Clear the trajectory data
        """
        self.states = {k: [] for k in self.states.keys()}
        self.actions = {k: [] for k in self.actions.keys()}
        self.aero_params = {k: [] for k in self.aero_params.keys()}
        self.forces = {k: [] for k in self.forces.keys()}
        self.ref_trajectory = []  
        
    @property
    def is_empty(self) -> bool:
        return all(len(v) == 0 for v in self.states.values())
        
    def dump(self, filepath):
        """
        Saves the collected trajectory data to a file using pickle.

        Args:
            filepath (str): The full path to the output file.
        """
        if self.is_empty:
            print("Warning: Attempted to save an empty trajectory. Skipping.")
            return

        # Ensure the directory exists
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        try:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
            print(f"Trajectory saved to {filepath}")
        except Exception as e:
            print(f"Error saving trajectory to {filepath}: {e}")

    @staticmethod
    def load(filepath):
        """Loads a trajectory from a pickle file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
