import warp as wp


# Per-step altitude-keeping reward, matching AltitudeKeepingEnv.get_reward
# (positive altitude mode, linear airspeed mode, alpha/va-safety disabled).
# Terminal crash/stall/horizon adjustments are applied by the wrapper, not here.
@wp.kernel
def compute_reward(
    position: wp.array(dtype=wp.vec3f),
    linear_vel: wp.array(dtype=wp.vec3f),
    angular_vel: wp.array(dtype=wp.vec3f),
    Va: wp.array(dtype=wp.float32),
    alpha: wp.array(dtype=wp.float32),
    action: wp.array(dtype=wp.float32, ndim=2),
    prev_action: wp.array(dtype=wp.float32, ndim=2),
    target_altitude: wp.array(dtype=wp.float32),
    target_airspeed: wp.array(dtype=wp.float32),   # per-env (A3 va_h task); constant = trim elsewhere
    w_alt: wp.float32,
    w_va: wp.float32,
    w_pr: wp.float32,
    w_sm: wp.float32,
    altitude_sigma: wp.float32,
    altitude_zone: wp.float32,
    va_band: wp.float32,
    recovery_threshold: wp.float32,
    overshoot_weight: wp.float32,
    overshoot_threshold: wp.float32,
    survival_bonus: wp.float32,
    alpha_safety_weight: wp.float32,
    alpha_safety_start: wp.float32,
    alpha_soft_deg: wp.float32,
    va_safety_weight: wp.float32,
    va_safety_start: wp.float32,
    va_stall_threshold: wp.float32,
    g: wp.float32,
    energy_weight: wp.float32,
    energy_sigma: wp.float32,
    damp_weight: wp.float32,
    damp_zone: wp.float32,
    pr_cap: wp.float32,
    effort_weight: wp.float32,
    reward: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    h_error = -position[tid][2] - target_altitude[tid]
    ae = Va[tid] - target_airspeed[tid]
    pre = angular_vel[tid][1]
    ah = wp.abs(h_error)

    gz = h_error / altitude_sigma
    altitude_reward = wp.exp(-0.5 * gz * gz)
    airspeed_reward = -wp.min(wp.abs(ae) / wp.max(target_airspeed[tid], 1.0), 1.0)
    pitch_rate_penalty = -wp.min(pre * pre, pr_cap)

    d0 = action[tid, 0] - prev_action[tid, 0]
    d1 = action[tid, 1] - prev_action[tid, 1]
    raw_smooth = -(wp.abs(d0) + wp.abs(d1)) * 0.5
    recovery = wp.max(0.0, 1.0 - ah / recovery_threshold)
    smoothness_penalty = raw_smooth * recovery
    # UNGATED action-rate (control-effort) penalty: damps the transient throttle/elevator oscillation
    # (the recovery-gated smoothness above is OFF far from target, where the oscillation lives).
    effort_penalty = -(d0 * d0 + d1 * d1)

    altitude_taper = wp.max(0.0, 1.0 - ah / altitude_zone)
    speed_taper = wp.max(0.0, 1.0 - wp.abs(ae) / va_band)
    stability_bonus = w_alt * 2.0 * altitude_taper * speed_taper

    overshoot_penalty = float(0.0)
    if h_error > overshoot_threshold:
        excess = h_error - overshoot_threshold
        overshoot_penalty = -wp.min((excess / overshoot_threshold) * (excess / overshoot_threshold), 1.0)

    # incidence shaping (Shukla/Khanzada out-of-band margin): 0 inside the safe envelope,
    # -1 at alpha_soft_deg. Deliberately saturating BELOW the hard alpha_max gate -- this is what
    # actually keeps a policy inside the envelope the derivative set was fitted in.
    alpha_safety_penalty = float(0.0)
    if alpha_safety_weight > 0.0:
        alpha_deg = wp.abs(alpha[tid]) * 180.0 / wp.PI
        onset = alpha_safety_start * alpha_soft_deg
        span = wp.max(alpha_soft_deg - onset, 1e-3)
        ov = wp.max(0.0, (alpha_deg - onset) / span)
        alpha_safety_penalty = -wp.min(ov * ov, 1.0)

    va_safety_penalty = float(0.0)
    if va_safety_weight > 0.0:
        span = wp.max(va_safety_start - va_stall_threshold, 1e-3)
        ov = wp.max(0.0, (va_safety_start - Va[tid]) / span)
        va_safety_penalty = -wp.min(ov * ov, 1.0)

    # TECS total-energy reward: penalize total specific energy error (altitude units),
    # eE = h_error + (Va^2 - Va_target^2)/(2g). Couples h and Va so excess kinetic energy
    # is bled off before it overshoots altitude.
    energy_reward = float(0.0)
    if energy_weight > 0.0:
        ke_dev = (Va[tid] * Va[tid] - target_airspeed[tid] * target_airspeed[tid]) / (2.0 * g)
        eE = h_error + ke_dev
        ge = eE / energy_sigma
        energy_reward = wp.exp(-0.5 * ge * ge)

    # approach damping: penalize vertical speed when near target -> decelerate into the
    # target instead of charging it (cuts overshoot + aggressive-approach crashes).
    damp_penalty = float(0.0)
    if damp_weight > 0.0:
        vz = linear_vel[tid][2]
        proximity = wp.clamp(1.0 - ah / damp_zone, 0.0, 1.0)
        damp_penalty = -wp.min(vz * vz, 25.0) * proximity

    reward[tid] = (
        w_alt * altitude_reward
        + w_va * airspeed_reward
        + w_pr * pitch_rate_penalty
        + w_sm * smoothness_penalty
        + stability_bonus
        + survival_bonus
        + overshoot_weight * overshoot_penalty
        + alpha_safety_weight * alpha_safety_penalty
        + va_safety_weight * va_safety_penalty
        + energy_weight * energy_reward
        + damp_weight * damp_penalty
        + effort_weight * effort_penalty
    )
