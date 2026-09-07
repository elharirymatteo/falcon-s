import os

import warp as wp

# Commanded height of the constant-altitude trajectory (see controllers/lqr/reference.py, which
# reads the same TRAJ_CONST_ALT env var so every controller tracks an identical reference).
# warp bakes constants at module build time and does NOT rebuild when the value is rebound, so
# this must be set in the environment before import — sweep one altitude per process.
CONST_ALT = wp.constant(float(os.environ.get("TRAJ_CONST_ALT", "1.0")))

# Rate-limited altitude ACQUISITION reference — the warp twin of controllers/lqr/reference.py's
# acquire_ramp_ref, reading the same env vars so LQR, MPPI and the RL policies all track one
# identical rate-limited command (the RL policies' training reference; a raw step is a different
# task). Baked at module build time like CONST_ALT: set before import, one target per process.
ACQ_H0 = wp.constant(float(os.environ.get("TRAJ_ACQ_H0", "1.0")))
ACQ_RATE = wp.constant(float(os.environ.get("TRAJ_ACQ_RATE", "2.0")))
# Reference airspeed [m/s]. Every trajectory function historically emitted 30.0, which is only
# right for the V7; the Ranger trims at 15. Baked like the rest of this module.
REF_VA = wp.constant(float(os.environ.get("TRAJ_REF_VA", "30.0")))

# ==================== INDIVIDUAL TRAJECTORY FUNCTIONS ====================

@wp.func
def constant_altitude_ref(t: wp.float32) -> wp.vec3f:
    """Constant altitude trajectory - returns [y_pos, z_pos, velocity]"""
    return wp.vec3f(0.0, -CONST_ALT, REF_VA)  # [lateral, altitude, forward_velocity]

@wp.func
def acquire_ramp_ref(t: wp.float32) -> wp.vec3f:
    """Rate-limited acquisition: ACQ_H0 -> CONST_ALT at ACQ_RATE m/s, then hold."""
    step = CONST_ALT - ACQ_H0
    travelled = wp.min(wp.abs(step), ACQ_RATE * t)
    return wp.vec3f(0.0, -(ACQ_H0 + wp.sign(step) * travelled), REF_VA)

@wp.func
def altitude_sine_wave_ref(t: wp.float32) -> wp.vec3f:
    """Sine wave altitude trajectory - returns [y_pos, z_pos, velocity]"""
    y_pos = 0.0  # Straight flight
    altitude = -6.0 + 5.0 * wp.sin(0.15 * t + wp.PI / 2.0)
    velocity = 30.0
    return wp.vec3f(y_pos, altitude, velocity)

@wp.func
def lateral_sine_wave_ref(t: wp.float32) -> wp.vec3f:
    """Sine wave lateral trajectory - returns [y_pos, z_pos, velocity]"""
    y_pos = 5.0 * wp.sin(0.03 * t)  # Lateral oscillation
    altitude = -1.0  # Constant altitude
    velocity = 30.0
    return wp.vec3f(y_pos, altitude, velocity)

@wp.func
def combined_sine_wave_ref(t: wp.float32) -> wp.vec3f:
    """Combined sine wave (both lateral and altitude) - returns [y_pos, z_pos, velocity]"""
    y_pos = 3.0 * wp.sin(0.03 * t)  # Lateral oscillation
    altitude = -3.0 + 2.0 * wp.sin(0.18 * t + wp.PI / 2.0)  # Altitude oscillation
    velocity = 30.0
    return wp.vec3f(y_pos, altitude, velocity)

@wp.func
def spiral_longitudinal_ref(t: wp.float32) -> wp.vec3f:
    """Longitudinal spiral trajectory - returns [y_pos, z_pos, velocity]"""
    y_pos = 3.0 * wp.sin(0.2 * t)  # Lateral oscillation
    z_pos = -3.0 + 2.0 * wp.sin(0.2 * t + wp.PI / 2.0)  # Altitude oscillation
    velocity = 30.0
    return wp.vec3f(y_pos, z_pos, velocity)

@wp.func
def ramp_ascent_ref(t: wp.float32) -> wp.vec3f:
    """Ramp ascent trajectory - returns [y_pos, z_pos, velocity]"""
    y_pos = 0.0  # Straight flight
    ascent_rate = 0.55
    altitude = -10.0 - ascent_rate * t
    velocity = 30.0
    return wp.vec3f(y_pos, altitude, velocity)

@wp.func
def ramp_lateral_ref(t: wp.float32) -> wp.vec3f:
    """Ramp lateral trajectory - returns [y_pos, z_pos, velocity]"""
    lateral_rate = 0.15
    y_pos = lateral_rate * t
    altitude = -10.0
    velocity = 30.0
    return wp.vec3f(y_pos, altitude, velocity)

@wp.func
def sine_wave_ascent_ref(t: wp.float32) -> wp.vec3f:
    """Sine wave ascent trajectory - returns [y_pos, z_pos, velocity]"""
    radius = 3.0
    angular_freq = 0.03
    y_pos = radius * wp.sin(angular_freq * t)
    altitude = -10.0 - 0.52 * t  # Gradual ascent
    velocity = 30.0
    return wp.vec3f(y_pos, altitude, velocity)

@wp.func
def custom_alt_ramp_ref(t: wp.float32) -> wp.vec3f:
    """Custom altitude ramp trajectory - returns [y_pos, z_pos, velocity]"""
    y_pos = 0.0  # Straight flight
    altitude = -1.0  # default
    
    if t < 10.0: 
        altitude = -t - 1.0 
    elif t < 30.0: 
        altitude = -11.0
    elif t < 40.0: 
        altitude = -11.0 + (t - 30.0) 
    elif t < 50.0: 
        altitude = -1.0  
    elif t < 59.0: 
        altitude = -1.0 + 0.05 * (t - 50.0) 
    else: 
        altitude = -0.55 
    
    return wp.vec3f(y_pos, altitude, 30.0)

@wp.func
def custom_mix_ramp_ref(t: wp.float32) -> wp.vec3f:
    """Custom altitude ramp trajectory - returns [y_pos, z_pos, velocity]"""
    y_pos = 0.0  # Straight flight
    altitude = -1.0  # default
    
    if t < 10.0: 
        y_pos = -t - 1.0 
        altitude = -t - 1.0 
    elif t < 30.0: 
        y_pos = -11.0
        altitude = -11.0
    elif t < 40.0: 
        y_pos = -11.0 + (t - 30.0)
        altitude = -11.0 + (t - 30.0) 
    elif t < 50.0: 
        y_pos = -1.0
        altitude = -1.0  
    elif t < 59.0: 
        y_pos = -1.0 + 0.05 * (t - 50.0)
        altitude = -1.0 + 0.05 * (t - 50.0) 
    else: 
        y_pos = -0.55
        altitude = -0.55 
    
    return wp.vec3f(-y_pos, altitude, 30.0)

@wp.func
def altitude_step_ref(t: wp.float32) -> wp.vec3f:
    """Step altitude maneuver - returns [y_pos, z_pos, velocity]"""
    altitude = -10.0
    if t > 10.0 and t < 65.0:
        altitude = -12.0  # Step up
    elif t >= 65.0:
        altitude = -10.0  # Return to original altitude

    y_pos = 0.0  # Constant y_position
    velocity = 30.0
    return wp.vec3f(y_pos, altitude, velocity)

@wp.func
def lateral_step_ref(t: wp.float32) -> wp.vec3f:
    """Step lateral maneuver - returns [y_pos, z_pos, velocity]"""
    y_pos = 0.0
    if t > 10.0 and t < 65.0:
        y_pos = 2.0  # Step to the side
    elif t >= 65.0:
        y_pos = 0.0  # Return to center
    
    altitude = -10.0  # Constant altitude
    velocity = 30.0
    return wp.vec3f(y_pos, altitude, velocity)

@wp.func
def waypoint_following_ref(t: wp.float32) -> wp.vec3f:
    """Waypoint following trajectory - returns [y_pos, z_pos, velocity]"""
    # TODO: Implement waypoint interpolation
    return wp.vec3f(0.0, -10.0, 30.0)

@wp.func
def constant_low_altitude_ref(t: wp.float32) -> wp.vec3f:
    """Constant low altitude trajectory - returns [y_pos, z_pos, velocity]"""
    return wp.vec3f(0.0, -0.5, 30.0)

# ==================== DISPATCHER FUNCTION ====================

@wp.func
def compute_reference(t: wp.float32, trajectory_type: wp.int32) -> wp.vec3f:
    """Dispatch to appropriate trajectory function - returns [y_pos, z_pos, velocity]"""
    if trajectory_type == 0:
        return constant_altitude_ref(t)
    elif trajectory_type == 1:
        return altitude_sine_wave_ref(t)
    elif trajectory_type == 2:
        return lateral_sine_wave_ref(t)
    elif trajectory_type == 3:
        return combined_sine_wave_ref(t)
    elif trajectory_type == 4:
        return ramp_ascent_ref(t)
    elif trajectory_type == 5:
        return ramp_lateral_ref(t)
    elif trajectory_type == 6:
        return sine_wave_ascent_ref(t)
    elif trajectory_type == 7:
        return custom_alt_ramp_ref(t)
    elif trajectory_type == 8:
        return custom_mix_ramp_ref(t)
    elif trajectory_type == 9:
        return altitude_step_ref(t)
    elif trajectory_type == 10:
        return lateral_step_ref(t)
    elif trajectory_type == 11:
        return waypoint_following_ref(t)
    elif trajectory_type == 12:
        return constant_low_altitude_ref(t)
    elif trajectory_type == 13:
        return spiral_longitudinal_ref(t)
    elif trajectory_type == 14:
        return acquire_ramp_ref(t)
    else:
        return constant_altitude_ref(t)  # default fallback

# ==================== TRAJECTORY TYPE MAPPING ====================

CONSTANT_ALTITUDE = 0
ALTITUDE_SINE_WAVE = 1
LATERAL_SINE_WAVE = 2
COMBINED_SINE_WAVE = 3
RAMP_ASCENT = 4
RAMP_LATERAL = 5
SINE_WAVE_ASCENT = 6
CUSTOM_ALT_RAMP = 7
CUSTOM_MIX_RAMP = 8
ALTITUDE_STEP = 9
LATERAL_STEP = 10
WAYPOINT_FOLLOWING = 11
CONSTANT_LOW_ALTITUDE = 12
SPIRAL_LONGITUDINAL = 13
ACQUIRE_RAMP = 14

def get_trajectory_type_id(trajectory_type: str) -> int:
    """Convert trajectory type string to GPU integer ID"""
    trajectory_map = {
        'constant_altitude': CONSTANT_ALTITUDE,
        'altitude_sine_wave': ALTITUDE_SINE_WAVE,
        'lateral_sine_wave': LATERAL_SINE_WAVE,
        'combined_sine_wave': COMBINED_SINE_WAVE,
        'ramp_ascent': RAMP_ASCENT,
        'ramp_lateral': RAMP_LATERAL,
        'sine_wave_ascent': SINE_WAVE_ASCENT,
        'custom_alt_ramp': CUSTOM_ALT_RAMP,
        'custom_mix_ramp': CUSTOM_MIX_RAMP,
        'altitude_step': ALTITUDE_STEP,
        'lateral_step': LATERAL_STEP,
        'waypoint_following': WAYPOINT_FOLLOWING,
        'constant_low_altitude': CONSTANT_LOW_ALTITUDE,
        'spiral_longitudinal': SPIRAL_LONGITUDINAL,
        'acquire_ramp': ACQUIRE_RAMP
    }
    
    if trajectory_type not in trajectory_map:
        available = list(trajectory_map.keys())
        raise ValueError(f"Unknown trajectory type: {trajectory_type}. Available: {available}")
    
    return trajectory_map[trajectory_type]

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
    'altitude_step',
    'lateral_step',
    'waypoint_following',
    'constant_low_altitude',
    'acquire_ramp'
]