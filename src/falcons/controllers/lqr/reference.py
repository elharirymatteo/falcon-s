import os

import numpy as np

# Commanded height of the constant-altitude trajectory, in metres above the water.
# Default 1.0 m = the WIG cruise point used by the benchmark scenarios. Override with the
# TRAJ_CONST_ALT env var (read at import) to sweep the commanded altitude; the MPPI warp
# reference (controllers/mppi/reference.py) reads the same variable so all controllers
# track an identical reference. Must be set before import — see scripts/sweep_traj_altitude.py.
def _const_alt():
    """Commanded height of the constant-altitude / acquisition references [m].

    Read at CALL time on this NumPy path so one process can sweep several commanded altitudes.
    The warp twin (controllers/mppi/reference.py) BAKES the same variable as a module constant,
    so any process that also drives MPPI must fix TRAJ_CONST_ALT before importing it and leave it
    alone -- otherwise the two references silently disagree. Sweeps that involve MPPI therefore
    run one commanded altitude per subprocess.
    """
    return float(os.environ.get("TRAJ_CONST_ALT", "1.0"))


CONST_ALT = _const_alt()          # retained for callers that read the module attribute directly

# Rate-limited altitude ACQUISITION reference: ramp from the spawn altitude to the commanded
# target at ACQ_RATE m/s, then hold. This is the reference the RL altitude policies are trained
# and evaluated against (WarpAltitudeEnv's `ref_rate` sub-target ramp, train/configs.py), so the
# classical controllers need it too — commanding them a raw altitude STEP is a different task and
# on the V7 drives the LQR past 1.5x stall alpha within half a second. Set via the environment
# before import, for the same warp-constant reason as TRAJ_CONST_ALT.
# Unlike CONST_ALT these are read at CALL time, not import time: this is the NumPy path, so
# nothing is baked, and reading them live lets a single process sweep many spawn altitudes.
# The warp twin in controllers/mppi/reference.py must still bake them (warp constants), which is
# why the MPPI side runs one spawn per subprocess.
def _acq_h0():
    return float(os.environ.get("TRAJ_ACQ_H0", "1.0"))      # spawn altitude [m]


def _acq_rate():
    return float(os.environ.get("TRAJ_ACQ_RATE", "2.0"))    # sub-target ramp rate [m/s]

# ==================== INDIVIDUAL TRAJECTORY FUNCTIONS ====================

def constant_altitude_ref(t: float) -> np.ndarray:
    """Constant altitude trajectory - returns [y_pos, z_pos, velocity]"""
    return np.array([0.0, 0.0, -_const_alt()])  # [lateral, altitude, forward_velocity]

def acquire_ramp_ref(t: float) -> np.ndarray:
    """Rate-limited acquisition: TRAJ_ACQ_H0 -> CONST_ALT at TRAJ_ACQ_RATE m/s, then hold."""
    h0, rate = _acq_h0(), _acq_rate()
    step = _const_alt() - h0
    travelled = min(abs(step), rate * t)
    return np.array([0.0, 0.0, -(h0 + np.sign(step) * travelled)])

def altitude_sine_wave_ref(t: float) -> np.ndarray:
    """Sine wave altitude trajectory - returns [y_pos, z_pos, velocity]"""
    y_pos = 0.0  # Straight flight
    z_pos = -6.0 + 5.0 * np.sin(0.15 * t + np.pi / 2)
    x_pos = 0.0
    return np.array([x_pos, y_pos, z_pos])

def lateral_sine_wave_ref(t: float) -> np.ndarray:
    """Sine wave lateral trajectory - returns [y_pos, z_pos, velocity]"""
    y_pos = 5.0 * np.sin(0.03 * t)  # Lateral oscillation
    z_pos = -1.0  # Constant altitude
    x_pos = 0.0
    return np.array([x_pos, y_pos, z_pos])

def combined_sine_wave_ref(t: float) -> np.ndarray:
    """Combined sine wave (both lateral and altitude) - returns [y_pos, z_pos, velocity]"""
    y_pos = 3.0 * np.sin(0.03 * t)  # Lateral oscillation
    z_pos = -3.0 + 2.0 * np.sin(0.18 * t + np.pi / 2)  # Altitude oscillation
    x_pos = 0.0
    return np.array([x_pos, y_pos, z_pos])

def spiral_longitudinal_ref(t: float) -> np.ndarray:
    """Longitudinal spiral trajectory - returns [y_pos, z_pos, velocity]"""
    y_pos = 3.0 * np.sin(0.2 * t)  # Lateral oscillation
    z_pos = -3.0 + 2.0 * np.sin(0.2 * t + np.pi / 2)  # Altitude oscillation
    x_pos = 0.0
    return np.array([x_pos, y_pos, z_pos])

def ramp_ascent_ref(t: float) -> np.ndarray:
    """Ramp ascent trajectory - returns [y_pos, z_pos, velocity]"""
    y_pos = 0.0  # Straight flight
    ascent_rate = 0.55
    z_pos = -10.0 - ascent_rate * t
    x_pos = 0.0
    return np.array([x_pos, y_pos, z_pos])

def ramp_lateral_ref(t: float) -> np.ndarray:
    """Ramp lateral trajectory - returns [y_pos, z_pos, velocity]"""
    lateral_rate = 0.15
    y_pos = lateral_rate * t
    z_pos = -10.0
    x_pos = 0.0
    return np.array([x_pos, y_pos, z_pos])

def sine_wave_ascent_ref(t: float) -> np.ndarray:
    """Sine wave ascent trajectory - returns [y_pos, z_pos, velocity]"""
    radius = 3.0
    angular_freq = 0.03
    y_pos = radius * np.sin(angular_freq * t)
    z_pos = -10.0 - 0.52 * t  # Gradual ascent
    x_pos = 0.0
    return np.array([x_pos, y_pos, z_pos])

def custom_alt_ramp_ref(t: float) -> np.ndarray:
    """Custom altitude ramp trajectory - returns [y_pos, z_pos, velocity]"""
    y_pos = 0.0  # Straight flight
    z_pos = -1.0  # default
    
    if t < 10.0: 
        z_pos = -t - 1.0 
    elif t < 30.0: 
        z_pos = -11.0
    elif t < 40.0: 
        z_pos = -11.0 + (t - 30.0) 
    elif t < 50.0: 
        z_pos = -1.0  
    elif t < 59.0: 
        z_pos = -1.0 + 0.05 * (t - 50.0) 
    else: 
        z_pos = -0.55 
    
    return np.array([0.0, y_pos, z_pos])

def custom_mix_ramp_ref(t: float) -> np.ndarray:
    """Custom altitude ramp trajectory - returns [y_pos, z_pos, velocity]"""
    y_pos = 0.0  # Straight flight
    z_pos = -1.0  # default
    
    if t < 10.0: 
        y_pos = -t - 1.0 
        z_pos = -t - 1.0 
    elif t < 30.0: 
        y_pos = -11.0
        z_pos = -11.0
    elif t < 40.0: 
        y_pos = -11.0 + (t - 30.0)
        z_pos = -11.0 + (t - 30.0) 
    elif t < 50.0: 
        y_pos = -1.0
        z_pos = -1.0  
    elif t < 59.0: 
        y_pos = -1.0 + 0.05 * (t - 50.0)
        z_pos = -1.0 + 0.05 * (t - 50.0) 
    else: 
        y_pos = -0.55
        z_pos = -0.55 
    
    return np.array([0.0, -y_pos, z_pos])

def altitude_step_ref(t: float) -> np.ndarray:
    """Step altitude maneuver - returns [y_pos, z_pos, velocity]"""
    z_pos = -10.0
    if t > 10.0 and t < 65.0:
        z_pos = -12.0  # Step up
    elif t >= 65.0:
        z_pos = -10.0  # Return to original altitude

    y_pos = 0.0  # Constant y_position
    x_pos = 0.0
    return np.array([x_pos, y_pos, z_pos])

def lateral_step_ref(t: float) -> np.ndarray:
    """Step lateral maneuver - returns [y_pos, z_pos, velocity]"""
    y_pos = 0.0
    if t > 10.0 and t < 65.0:
        y_pos = 2.0  # Step to the side
    elif t >= 65.0:
        y_pos = 0.0  # Return to center
    
    z_pos = -10.0  # Constant altitude
    x_pos = 0.0
    return np.array([x_pos, y_pos, z_pos])

def waypoint_following_ref(t: float) -> np.ndarray:
    """Waypoint following trajectory - returns [y_pos, z_pos, velocity]"""
    # TODO: Implement waypoint interpolation
    return np.array([0.0, 0.0, -1.0])

def constant_low_altitude_ref(t: float) -> np.ndarray:
    """Constant low altitude trajectory - returns [y_pos, z_pos, velocity]"""
    return np.array([0.0, 0.0, -0.5])

# ==================== DISPATCHER FUNCTION ====================

def compute_reference(t: float, trajectory_type: str) -> np.ndarray:
    """Dispatch to appropriate trajectory function - returns [y_pos, z_pos, velocity]"""
    trajectory_map = {
        'constant_altitude': constant_altitude_ref,
        'altitude_sine_wave': altitude_sine_wave_ref,
        'lateral_sine_wave': lateral_sine_wave_ref,
        'combined_sine_wave': combined_sine_wave_ref,
        'spiral_longitudinal': spiral_longitudinal_ref,
        'ramp_ascent': ramp_ascent_ref,
        'ramp_lateral': ramp_lateral_ref,
        'sine_wave_ascent': sine_wave_ascent_ref,
        'custom_alt_ramp': custom_alt_ramp_ref,
        'custom_mix_ramp': custom_mix_ramp_ref,
        'altitude_step': altitude_step_ref,
        'lateral_step': lateral_step_ref,
        'waypoint_following': waypoint_following_ref,
        'constant_low_altitude': constant_low_altitude_ref,
        'acquire_ramp': acquire_ramp_ref
    }
    
    if trajectory_type not in trajectory_map:
        available = list(trajectory_map.keys())
        raise ValueError(f"Unknown trajectory type: {trajectory_type}. Available: {available}")
    
    return trajectory_map[trajectory_type](t)

# ==================== AVAILABLE TRAJECTORY TYPES ====================

AVAILABLE_TRAJECTORIES = [
    'constant_altitude',
    'altitude_sine_wave', 
    'lateral_sine_wave',
    'combined_sine_wave',
    'ramp_ascent',
    'ramp_lateral',
    'sine_wave_ascent',
    'custom_alt_ramp',
    'custom_mix_ramp',
    'spiral_longitudinal',
    'altitude_step',
    'lateral_step',
    'waypoint_following',
    'constant_low_altitude',
    'acquire_ramp'
]