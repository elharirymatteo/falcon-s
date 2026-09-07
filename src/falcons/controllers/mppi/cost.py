import os

import warp as wp

# Longitudinal-task cost tuning, read at import (warp bakes constants at module build time).
#   MPPI_ALPHA_LIMIT_DEG : incidence limit used by alpha_tracking_cost (historical: 5 deg)
#   MPPI_SOFT_ALPHA      : 1 = soft ramp over the last 40% of the limit instead of a hard wall,
#                          which a sampler cannot see past until a rollout is already through it
#   MPPI_W_VZ            : vertical-speed damping weight. The altitude term is a POSITION penalty
#                          and says nothing about sink rate, so a slowly divergent phugoid-like
#                          mode stays invisible to a one-second preview until the position error
#                          has already grown. 0 reproduces the historical cost.
ALPHA_LIMIT_DEG = wp.constant(float(os.environ.get("MPPI_ALPHA_LIMIT_DEG", "5.0")))
W_VZ = wp.constant(float(os.environ.get("MPPI_W_VZ", "0.0")))
SOFT_ALPHA = wp.constant(int(os.environ.get("MPPI_SOFT_ALPHA", "0")))

@wp.func
def altitude_tracking_cost(position: wp.vec3f,
                          lin_vel: wp.vec3f,
                          ang_vel: wp.vec3f,
                          ori: wp.quatf,
                          target_altitude: wp.float32) -> wp.float32:
    """
    Cost for maintaining target altitude
    """
    altitude_error = position[2] - target_altitude
    
    # Quadratic cost for altitude tracking
    # altitude_cost = 1.0 * altitude_error * altitude_error
    # Linear cost for altitude tracking
    altitude_cost = wp.abs(altitude_error)
    
    return altitude_cost

@wp.func
def lateral_tracking_cost(position: wp.vec3f,
                         lin_vel: wp.vec3f,
                         ang_vel: wp.vec3f,
                         ori: wp.quatf,
                         target_lateral: wp.float32) -> wp.float32: 
    """
    Cost for maintaining target lateral position
    """
    y_pos_error = position[1] - target_lateral

    lateral_cost = wp.abs(y_pos_error)

    return lateral_cost

@wp.func
def ground_collision_cost(position: wp.vec3f,
                         lin_vel: wp.vec3f,
                         ang_vel: wp.vec3f,
                         ori: wp.quatf) -> wp.float32:
    """
    High cost for getting close to or below ground (z <= 0 in NED)
    """
    cost = 0.0
    safety_margin = -0.4  # 40cm above ground minimum

    if position[2] > safety_margin:  # Getting too close to ground
        # Very large cost for ground collision
        cost = 10000000.0

    return cost

@wp.func
def ang_vel_tracking_cost(position: wp.vec3f,
                  lin_vel: wp.vec3f,
                  ang_vel: wp.vec3f,
                  ori: wp.quatf) -> wp.float32:
    """
    Cost for maintaining stability (penalize large angular velocities)
    """
    # Penalize angular velocity magnitude (proper way)
    # ang_vel_magnitude_sq = 1.0*ang_vel[0]*ang_vel[0] + 1.0*ang_vel[1]*ang_vel[1] + 1.0*ang_vel[2]*ang_vel[2]
    ang_vel_magnitude_sq = 1.0*wp.abs(ang_vel[0]) + 1.0*wp.abs(ang_vel[1]) + 1.0*wp.abs(ang_vel[2])
    return ang_vel_magnitude_sq

@wp.func
def vel_tracking_cost(position: wp.vec3f,
                     lin_vel: wp.vec3f,
                     ang_vel: wp.vec3f,
                     ori: wp.quatf,
                     target_velocity: wp.float32) -> wp.float32:
    """
    Cost for maintaining target velocity
    """
    # target_velocity = wp.vec3f(30.0, 0.0, 0.0)
    # velocity_error = lin_vel - target_velocity

    # # vel_cost = 1.0*velocity_error[0]*velocity_error[0] + \
    # #            1.0*velocity_error[1]*velocity_error[1] + \
    # #            1.0*velocity_error[2]*velocity_error[2]

    # vel_cost = 4.0*wp.abs(velocity_error[0]) + \
    #            1.0*wp.abs(velocity_error[1]) + \
    #            1.0*wp.abs(velocity_error[2])
    
    # `target_velocity` is the reference's airspeed component (kernels.py: ref_values[2]).
    # The historical code discarded it and hardcoded 30 m/s -- roughly the V7 trim and roughly
    # TWICE the Volantex Ranger's, so on that airframe MPPI was commanded an unreachable speed
    # at cost weight 4.5 and traded away altitude tracking to chase it.
    TAS = wp.length(lin_vel)

    velocity_error = TAS - target_velocity

    vel_cost = 1.0 * wp.abs(velocity_error)
    
    # Quadratic cost for velocity tracking
    return vel_cost

@wp.func
def sink_rate_cost(lin_vel: wp.vec3f, ori: wp.quatf) -> wp.float32:
    """Penalize vertical speed in the world frame -- damps the slow altitude divergence that a
    pure position cost cannot see within a one-second preview."""
    vw = wp.quat_rotate(ori, lin_vel)
    return wp.abs(vw[2])


@wp.func
def excessive_altitude_cost(position: wp.vec3f,
                           lin_vel: wp.vec3f,
                           ang_vel: wp.vec3f,
                           ori: wp.quatf) -> wp.float32:
    """
    Cost for flying too high (too negative z in NED)
    """
    cost = 0.0
    # max_altitude = -20.0  # Don't go higher than 10m above ground
    
    # if position[2] < max_altitude:  # Too high
    #     altitude_violation = max_altitude - position[2]
    #     cost = altitude_violation * altitude_violation
    
    return cost

@wp.func
def alpha_tracking_cost(position: wp.vec3f,
                        lin_vel: wp.vec3f,
                        ang_vel: wp.vec3f,
                        ori: wp.quatf,
                        alpha: wp.float32) -> wp.float32:
    """
    Cost for maintaining alpha angle (angle of attack)
    """
    alpha_deg = wp.abs(alpha) * 180.0 / wp.pi
    cost = float(0.0)
    if SOFT_ALPHA != 0:
        # Soft ramp over the last 40% of the limit, then a wall -- the same shape as
        # incidence_guard_cost. A purely terminal wall (the branch below) is invisible to the
        # sampler until a rollout is already past it, and once EVERY sample is past it they all
        # carry the same 1e7 and the softmax has no gradient left to steer with.
        if alpha_deg > 0.6 * ALPHA_LIMIT_DEG:
            cost = 40.0 * (alpha_deg - 0.6 * ALPHA_LIMIT_DEG)
        if alpha_deg > ALPHA_LIMIT_DEG:
            cost = 10000.0
    elif alpha_deg > ALPHA_LIMIT_DEG:
        cost = 10000000.0
    return cost    

@wp.func
def bank_tracking_cost(ori: wp.quatf, target_bank: wp.float32) -> wp.float32:
    """Cost for tracking a commanded bank angle phi* [rad] (attitude-executor task)."""
    qx = ori[0]; qy = ori[1]; qz = ori[2]; qw = ori[3]
    roll = wp.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    return wp.abs(roll - target_bank)


@wp.func
def climb_rate_tracking_cost(lin_vel: wp.vec3f, ori: wp.quatf,
                             target_hdot: wp.float32) -> wp.float32:
    """Cost for tracking a commanded climb rate hdot* [m/s]. climb_rate = -vz_world."""
    vw = wp.quat_rotate(ori, lin_vel)
    return wp.abs(-vw[2] - target_hdot)


@wp.func
def airspeed_tracking_cost(lin_vel: wp.vec3f, target_va: wp.float32) -> wp.float32:
    """Cost for tracking a commanded airspeed Va* [m/s] (no-wind: |v_body|)."""
    return wp.abs(wp.length(lin_vel) - target_va)


@wp.func
def incidence_guard_cost(alpha: wp.float32, alpha_limit: wp.float32) -> wp.float32:
    """Guard on angle of attack. Unlike alpha_tracking_cost's fixed 5deg threshold, the limit is
    the airframe's own stall margin -- a coordinated turn needs extra alpha (load factor
    1/cos phi), so a 5deg cap would forbid the maneuver itself. The cost ramps up over the last
    40% of the margin: a purely terminal wall is invisible to the sampler until a rollout is
    already past it (and past it the poly aero model extrapolates, which sends the whole rollout
    to inf), so the soft ramp is what actually keeps the samples inside the fitted envelope."""
    a = wp.abs(alpha)
    cost = float(0.0)
    if a > 0.6 * alpha_limit:
        cost = 40.0 * (a - 0.6 * alpha_limit)
    if a > alpha_limit:
        cost = 10000.0     # dominates every tracking term (rejected at temperature 3) without
    return cost            # inflating the sum to where the softmax loses all resolution


@wp.func
def turn_rate_damping_cost(ang_vel: wp.vec3f, psidot: wp.float32,
                           target_bank: wp.float32) -> wp.float32:
    """Penalize body rates that DEVIATE from the commanded coordinated turn
    (p, q, r) = (0, psidot sin phi*, psidot cos phi*). A plain |omega| penalty would fight the
    turn itself; this one only damps the bang-bang rate excursions that walk a 1 s preview
    rollout into a departure."""
    return (wp.abs(ang_vel[0])
            + wp.abs(ang_vel[1] - psidot * wp.sin(target_bank))
            + wp.abs(ang_vel[2] - psidot * wp.cos(target_bank)))


@wp.func
def sideslip_cost(beta: wp.float32) -> wp.float32:
    """Linear sideslip penalty [deg] -- drives coordinated (ball-centred) turns. Linear, not the
    exponential of beta_tracking_cost, which saturates the total cost at a few degrees."""
    return wp.abs(beta * 180.0 / wp.pi)


@wp.func
def beta_tracking_cost(position: wp.vec3f,
                       lin_vel: wp.vec3f,
                       ang_vel: wp.vec3f,
                       ori: wp.quatf,
                       beta: wp.float32) -> wp.float32:
    """
    Cost for maintaining beta angle (sideslip angle)
    """
    # penalize beta exponentially if too high
    beta_deg = beta * 180.0 / wp.pi  # Convert to degrees
    beta_threshold = 0.0  # Threshold for beta angle``
    cost = 0.0
    if wp.abs(beta_deg) > beta_threshold: # beta_threshold = 2.0
        beta_violation = wp.abs(beta_deg) - beta_threshold
        cost = wp.exp(2.0 * beta_violation) - 1.0

    return cost
