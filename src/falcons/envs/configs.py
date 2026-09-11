"""Per-aircraft and per-task env configuration for PPO training.

Values are tuning history — comments document why each setting is what it is.
"""

# Per-task training defaults (std schedule end, default budget).
TASK_DEFAULTS = {
    "altitude":     dict(total_steps=150_000_000, std_end=0.5),
    "attitude":     dict(total_steps=150_000_000, std_end=0.5),   # attitude+speed executor
}

SPAWN_START = 5.0            # altitude spawn-curriculum start [m]

# ------------------------------------------------------------------ altitude

# Per-aircraft ramp trajectory config: climb rates to test + triangle amplitude
RAMP_CONFIGS = {
    "Airship_V7":      dict(rates=[1.0, 2.0, 3.0, 5.0], amp=20.0),
    "Airship_A0S":     dict(rates=[0.5, 1.0, 1.5, 2.0], amp=10.0),
    "Volantex_Ranger": dict(rates=[1.0, 2.0, 3.0, 4.0], amp=15.0),
    "Navion":          dict(rates=[1.0, 2.0, 3.0, 5.0], amp=20.0),
}

# Per-aircraft best operating points (dynamic altitude-keeping, rate-limited reference).
#   spawn    : max initial offset from target [m] (curriculum 5 -> spawn)
#   sigma    : gaussian altitude-reward half-width [m] (tighter = more precise)
#   w_pr     : pitch-rate penalty weight (low=precise/Airship; high=smooth-safe/fast GA planes)
#   ref_rate : sub-target ramp rate [m/s] (>0 = rate-limited ref; 0 = direct setpoint)
PLANE_CONFIGS = {
    "Airship_V7":  dict(spawn=10.0, sigma=3.0, w_pr=0.2, ref_rate=2.0),
    "Airship_A0S": dict(spawn=10.0, sigma=1.0, w_pr=0.2, ref_rate=1.0, damp_w=0.8, w_alt=6.0, zone=1.0, i_clamp=12.0),   # field-test prototype: narrow zone (1m) nulls steady undershoot bias; sigma=1.0/w_alt=6 keep it ROBUST to limited PX6/EKF2 state. ANTI-OVERSHOOT (June 2026): the slow altitude swing on step targets (overshoot then -5.7m undershoot, altitude_viz 0%) was an INTEGRAL LIMIT-CYCLE (ierr swung +/-20 against the default clamp=25), NOT the phugoid. i_clamp 25->12 kills it: undershoot -5.7->-0.0m, bias +2.4->+0.0m, settled RMSE 2.6->1.1m
    "Volantex_Ranger": dict(spawn=10.0, sigma=1.0, w_pr=0.2, ref_rate=2.0, w_alt=6.0, zone=1.0, damp_w=0.8,
                            # safe gust-rejection DR: low wind (cap curriculum at 0.5) + MODEST airspeed-keeping.
                            # NB: dr_va_start near trim (15) penalizes constantly -> collapse; keep va_safety_start
                            # at the default ~0.75*trim (no dr_va_start) and just lift w_va/va_safety a little.
                            dr_turb_cap=0.5, dr_w_va=3.5, dr_va_safety=4.5, dr_energy=4.0, dr_mix=True),   # 1.355 kg single-motor RC trainer, trim 15 m/s; precision recipe (tight sigma/zone + damping) -- loose defaults left ~0.44m settled jitter. dr_mix: alternate calm/gusty batches to keep calm precision AND gust rejection
    # Optuna winner env params (study ppo_altitude_Navion trial 35, Jul 2026): slow
    # ref_rate is the decisive fix -- the 3s piston-motor lag can't chase 2 m/s target
    # slews (old config RMSE 10.8m). Canonical retrain on this config (default algo
    # params): rob 1.000 / RMSE 6.70m. The full winner incl. its algo params
    # (lr0=3.9e-4, std1=0.56 -- not encoded here) reaches 5.29m
    # (warp_ppo_altitude_Navion_optuna_best.pt).
    "Navion":      dict(spawn=10.0, sigma=4.94, w_pr=3.0, ref_rate=0.24, zone=1.84,
                        damp_w=0.52, w_alt=5.2),
}


def altitude_env_cfg(aircraft, spawn, horizon=2000):
    c = PLANE_CONFIGS[aircraft]
    cfg = {"horizon": horizon, "spawning_distance": spawn, "altitude_sigma": c["sigma"],
           "target_mode": "dynamic", "target_lo": 30.0, "target_hi": 90.0,
           "aircraft": aircraft, "ref_rate": c["ref_rate"],
           "reward_weights": [c.get("w_alt", 5.0), 2.0, c["w_pr"], 0.1], "pr_cap": 9.0,
           "target_altitude_zone": c.get("zone", 3.0),
           "alpha_safety_weight": 3.0, "va_safety_weight": 3.0,
           "energy_weight": c.get("energy_w", 3.0), "damp_weight": c.get("damp_w", 0.5),
           "damp_zone": c.get("damp_zone", 10.0), "overshoot_weight": c.get("ov_w", 2.0),
           "effort_weight": c.get("effort_w", 0.0), "i_clamp": c.get("i_clamp", 25.0)}
    return cfg


# ------------------------------------------------------------------ attitude+speed executor

# Second goal is climb-rate hdot* (m/s), not pitch-attitude: a sustained pitch-attitude hold
# bleeds Va into stall at cruise power (bring-up run 1: rob 0.48, stalls from pitch; phi0/theta15
# eval = rob 0.61 vs phi45/theta0 = 0.96) AND isn't a sustainable goal (a bank with theta*=0
# descends). Climb-rate makes every command a sustainable steady state. w_energy up + va_safety
# raised so the energy floor bites well before stall, not at 0.75*Va. Bank (phi_max=45) is proven
# sustainable. va_safety is absolute m/s (~0.86*trim per airframe).
ATTITUDE_CONFIGS = {
    "Airship_V7":      dict(phi_max_deg=45.0, hdot_max=1.5, w_energy=4.0, va_safety=24.0),
    "Volantex_Ranger": dict(phi_max_deg=35.0, hdot_max=1.5, w_energy=4.0, va_safety=13.0),
    # A0S: twitchy light airframe -> gentle bank envelope (matches BANK_HEADING_CONFIGS 25deg);
    # trim 13 m/s. NB lateral moment coeffs are unvalidated (see a0s model caveats).
    "Airship_A0S":     dict(phi_max_deg=25.0, hdot_max=2.0, w_energy=4.0, va_safety=11.2),
    # GA piston airframes (post thrust fix, ad25cd0): real climb margin at cruise, so a wider
    # hdot envelope than the power-limited V7/Volantex. va_safety ~0.86*trim like the others.
    "Navion":          dict(phi_max_deg=45.0, hdot_max=3.0, w_energy=4.0, va_safety=64.5),
}


def attitude_env_cfg(aircraft, horizon=2000):
    cfg = {"horizon": horizon, "aircraft": aircraft}
    cfg.update(ATTITUDE_CONFIGS.get(aircraft, {}))
    return cfg


# Reference airspeed per airframe [m/s]: the classical reference modules emit this to the
# controllers, and the evaluation protocol commands it.
TRIM_VA = {"Airship_V7": 28.0, "Volantex_Ranger": 15.0,
           "Airship_A0S": 18.0, "Navion": 45.0}
