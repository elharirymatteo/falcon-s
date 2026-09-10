"""Ground effect as the thrust required to fly, measured two ways and indexed by height over span.

`run_trim` is open loop: at each band it SOLVES the level-flight trim (angle of attack, elevator,
throttle) so that the total body force and pitching moment vanish, integrating nothing and learning
nothing, so every airframe reports every band. `run_energy` is closed loop: a trained policy holds
the band and the mean squared throttle it spends is compared between the two arms.

Run: run_trim(planes) -> <results_dir>/ge_trim.csv, run_energy(planes) -> <results_dir>/ge_energy.csv
"""
import csv
import os
from pathlib import Path

import numpy as np
from scipy.optimize import root

from falcons.aircraft.config import PLANES
from falcons.paths import CKPT_DIR, RESULTS_DIR

GE_PLANES = list(PLANES)
# Altitude is indexed by h/b (height over wing span), not metres, because that is the variable the
# lifting-line correction actually depends on: a 5 m span airframe at 5 m and a 1.6 m span airframe
# at 1.6 m are at the same point in ground effect but at very different heights. Reporting in metres
# is what makes the effect look airframe-specific when it is not.
RATIOS = [0.25, 0.5, 1.0, 2.0, 3.0, 6.0]      # h/b bands: deep GE -> out of ground effect
REF_RATIO = 6.0                                # the out-of-ground-effect band used to pick airspeed
# Trim unknowns are solved in scaled units so the root finder sees one length scale:
# x = (alpha / A_SCALE [rad], elevator [deg], throttle [-]). Elevator limits are in DEGREES and
# the stall angle is in RADIANS, per the airframe JSONs.
A_SCALE = 0.05
STARTS = [(0.5, 0.0, 0.3), (0.0, 0.0, 0.5), (1.5, -2.0, 0.7), (-0.5, 2.0, 0.2), (2.5, -5.0, 0.9)]
RES_TOL = 1e-5                                 # residuals are forces/weight, moments/(weight*mac)
MIN_ALT = 0.35                                # do not command a target below this [m]
TRACK_TOL = 1.0                               # a cell counts as a valid HOLD only if the settled
TRACK_FRAC = 0.20                             # RMSE is under TRACK_TOL m and TRACK_FRAC of the target
N = 64                                        # parallel episodes per cell
STEPS = 4000                                  # 40 s
SETTLE = 2000                                 # measure over the second half


# ─── the reference curve, read off the aeroplane's OWN measured height sweep.
#     There is no analytic correlation here any more: ground effect is the OpenVSP data.
def _drag_ratio(aircraft, h):
    """CD(h) / CD(free air) at the trim point the sweep was measured at, in percent change.

    This replaces the lifting-line mu_d*mu_l^2 the simulator used to apply. It is not an
    independent prediction -- it is the same measured table the plant reads -- so it checks the
    plumbing and the height datum, not the physics.
    """
    from falcons.aircraft.config import AircraftConfig
    from falcons.aircraft.params import DerivativeAeroParameters
    from falcons.sim.aero_contract import AeroInputs
    from falcons.sim.cpu.physics.aerodynamics import DerivativeAerodynamics

    cfg = AircraftConfig(aircraft).load()
    wing = cfg["vehicle_params"]["wing"]
    params = DerivativeAeroParameters.from_config(cfg)
    mk = lambda ge: DerivativeAerodynamics(params, wing["span"], wing["mac"], ge_enable=ge)
    u = AeroInputs(alpha=params.alpha_run, beta=0.0, v=params.v_ref,
                   elevator=params.de_run, aileron=0.0, rudder=0.0,
                   p=0.0, q=0.0, r=0.0, h=float(h))
    return 100.0 * (mk(True).coefficients(u).CD / mk(False).coefficients(u).CD - 1.0)


def build(aircraft, ge):
    """One Aircraft (batch of 1) with the ground-effect correction set to `ge`, plus
    the scalars the trim solve and its validity check need."""
    from falcons.sim.warp.aircraft import Aircraft
    from falcons.aircraft.params import load_params
    co = load_params(aircraft)
    AP = co["aero_params"].as_warp_struct()
    VP = co["vehicle_params"]
    m, alpha_max, mac = float(VP.m), float(VP.alpha_max), float(VP.WP.mac)
    span = float(VP.WP.span)
    VP.WP = VP.WP.as_warp_struct(); VP.J = VP.J.as_warp_struct(); VP.PP = VP.PP.as_warp_struct()
    VP.sensor_system = None
    cl = co["control_limits"]
    model = Aircraft(1, "cuda", {"AP": AP, "VP": VP, "CL": cl,
                                 "EP": co["environment_params"]},
                     save_history=False, in_ground_effect=ge)
    model.solver_type = 1
    return dict(model=model, m=m, alpha_max=alpha_max, mac=mac, span=span,
                elev=[float(v) for v in cl.elevator_limits],
                thr=[float(v) for v in cl.throttle_limits],
                va0=float(co["default_initial_state"].linear_vel[0]))


def evaluate(P, h, va, x):
    """Place the aircraft at height `h` in level flight at `va` with trim candidate
    x = (alpha, elevator, throttle), and return (residuals, thrust, lift, drag).

    Level flight pins the flight-path angle to zero, so pitch = alpha and the body-frame velocity
    is (va cos a, 0, va sin a). Forces are read from the plant after one step: the buffers hold the
    forces computed FROM the state at entry, which is the state set here."""
    a, de, dt = float(x[0]) * A_SCALE, float(x[1]), float(x[2])
    mdl = P["model"]
    one = np.ones(1, dtype=np.float32)
    q = np.array([[0.0, np.sin(a / 2), 0.0, np.cos(a / 2)]], dtype=np.float32)   # pitch = alpha
    mdl.reset({"position": np.array([[0.0, 0.0, -h]], dtype=np.float32),
               "linear_vel": np.array([[va * np.cos(a), 0.0, va * np.sin(a)]], dtype=np.float32),
               "angular_vel": np.zeros((1, 3), dtype=np.float32), "orientation": q},
              # scalars, not 1-element arrays: Aircraft._set_buffer_data broadcasts either, but the
              # array path reaches float(ndarray), which numpy 2 deprecates -- one sweep of this
              # solve emits 17.6k DeprecationWarnings. Same value in the float32 buffer either way.
              {"elevator": de, "aileron": 0.0, "rudder": 0.0,
               "throttle_left": dt, "throttle_right": dt,
               "elevator_dot": 0.0, "aileron_dot": 0.0, "rudder_dot": 0.0})
    mdl.step({"elevator": de * one, "aileron": 0.0 * one, "rudder": 0.0 * one,
              "throttle_left": dt * one, "throttle_right": dt * one})
    Fb, Mb = mdl._Fb.numpy()[0], mdl._Mb.numpy()[0]
    w = P["m"] * 9.80665
    r = [Fb[0] / w, Fb[2] / w, Mb[1] / (w * P["mac"])]      # non-dimensional: weight, weight*chord
    return r, float(np.linalg.norm(mdl._Fb_thrust.numpy()[0])), \
        float(mdl._L_tot.numpy()[0]), float(mdl._D_tot.numpy()[0])


def trim(P, h, va):
    """Solve the three trim equations; return the cell dict, or None if no solution lies inside the
    control limits and below stall (an airframe that cannot hold this speed at this height)."""
    for x0 in STARTS:
        sol = root(lambda x: evaluate(P, h, va, x)[0], np.array(x0), method="hybr",
                   options=dict(xtol=1e-10))
        a, de, dt = float(sol.x[0]) * A_SCALE, float(sol.x[1]), float(sol.x[2])
        r, T, L, D = evaluate(P, h, va, sol.x)
        if max(abs(v) for v in r) > RES_TOL:
            continue                                       # not actually a root
        if not (P["thr"][0] <= dt <= P["thr"][1] and P["elev"][0] <= de <= P["elev"][1]):
            continue                                       # outside the control authority
        if abs(a) >= P["alpha_max"]:                       # radians; the hard incidence gate
            continue                            # outside the envelope the set was fitted in
        return dict(alpha_deg=np.degrees(a), elevator=de, throttle=dt, thrust_N=T, lift_N=L,
                    drag_N=D)
    return None


def airspeed(on, off):
    """The airframe's declared cruise if it trims out of ground effect, else the fastest speed
    below it that does, found by a scan. Both arms share it so the paired ratio is taken at one
    operating point. The V7 is the airframe that needs the scan: its declared 28 m/s exceeds its
    full-throttle level speed."""
    h = REF_RATIO * off["span"]
    for va in [off["va0"]] + list(np.arange(0.95, 0.44, -0.05) * off["va0"]):
        if trim(off, h, float(va)) and trim(on, h, float(va)):
            return float(va)
    return None


def theory_pct(aircraft, h):
    """Percent drag change at height `h` [m] from the airframe's measured OpenVSP sweep -- the
    factor the plant's own coefficients carry.

    Evaluated at the commanded height, full stop. The old correlation had to be evaluated at
    `h + cg_z_offset` because it took the WING's altitude; the measured sweep is indexed by CG
    height (FC_Zcg_ = 0 in the VSPAERO runs), which is what the plant now uses.
    """
    return _drag_ratio(aircraft, h)


def run_trim(planes, results_dir=RESULTS_DIR):
    import warp as wp
    wp.init()
    os.makedirs(results_dir, exist_ok=True)
    rows = []
    for ac in planes:
        on, off = build(ac, True), build(ac, False)
        va = airspeed(on, off)
        if va is None:
            print(f"{ac:17s} no level trim at any airspeed out of ground effect -- skipped")
            continue
        note = "declared" if abs(va - off["va0"]) < 1e-6 else f"reduced from {off['va0']:.1f}"
        print(f"{ac:17s} Va {va:6.2f} m/s ({note})")
        for r in RATIOS:
            h = r * off["span"]
            t_on, t_off = trim(on, h, va), trim(off, h, va)
            th = theory_pct(ac, h)
            # Drag with ground effect applied but the trim FROZEN at its out-of-ground-effect
            # value. The gap between this and the re-trimmed drag is the part of the saving that
            # comes from the aircraft re-trimming (lower alpha) rather than from the correction
            # itself, which is what the fixed-lift-coefficient prediction leaves out.
            d_froz = np.nan
            if t_on and t_off:
                x = np.array([np.radians(t_off["alpha_deg"]) / A_SCALE,
                              t_off["elevator"], t_off["throttle"]])
                d_froz = evaluate(on, h, va, x)[3]
            d = (100.0 * (t_on["thrust_N"] - t_off["thrust_N"]) / t_off["thrust_N"]) \
                if (t_on and t_off) else np.nan
            rows.append(dict(aircraft=ac, span=round(off["span"], 2), h_over_b=r,
                             altitude=round(h, 2), airspeed=round(va, 2),
                             thrust_ge_on=t_on["thrust_N"] if t_on else np.nan,
                             thrust_ge_off=t_off["thrust_N"] if t_off else np.nan,
                             delta_thrust_pct=d, theory_pct=th,
                             drag_ge_on=t_on["drag_N"] if t_on else np.nan,
                             drag_ge_off=t_off["drag_N"] if t_off else np.nan,
                             drag_frozen_alpha=d_froz,
                             alpha_ge_on=t_on["alpha_deg"] if t_on else np.nan,
                             alpha_ge_off=t_off["alpha_deg"] if t_off else np.nan,
                             throttle_ge_on=t_on["throttle"] if t_on else np.nan,
                             throttle_ge_off=t_off["throttle"] if t_off else np.nan,
                             elevator_ge_on=t_on["elevator"] if t_on else np.nan,
                             elevator_ge_off=t_off["elevator"] if t_off else np.nan,
                             trim_valid=int(bool(t_on and t_off))))
            print(f"  h/b {r:4.2f} (h {h:6.2f} m)  T {rows[-1]['thrust_ge_on']:8.2f}/"
                  f"{rows[-1]['thrust_ge_off']:8.2f} N  dT {d:+6.1f}%  theory {th:+6.1f}%",
                  flush=True)
    cols = ["aircraft", "span", "h_over_b", "altitude", "airspeed", "thrust_ge_on",
            "thrust_ge_off", "delta_thrust_pct", "theory_pct", "drag_ge_on", "drag_ge_off",
            "drag_frozen_alpha", "alpha_ge_on", "alpha_ge_off",
            "throttle_ge_on", "throttle_ge_off", "elevator_ge_on", "elevator_ge_off", "trim_valid"]
    out = Path(results_dir) / "ge_trim.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out}")
    return out


# ───── closed loop ─────
def geometry(aircraft):
    """(span, MAC) — the height scales the measurement and the sweep are indexed by."""
    from falcons.aircraft.params import load_params
    w = load_params(aircraft)["vehicle_params"].WP
    return float(w.span), float(w.mac)


def thrust_saving(aircraft, r, span):
    """Percent drag change at h/b = r, read off the airframe's measured sweep.

    Level trim has thrust balancing drag, so the drag ratio is the thrust the band should save.
    The height datum is now plain CG altitude -- the wing z offset the old correlation needed
    went out with it -- so the curve is evaluated at exactly the commanded height.
    """
    return _drag_ratio(aircraft, r * span)


def cell(aircraft, target, ge, ckpt_dir=CKPT_DIR):
    """Hold `target` m with ground effect `ge`; return (survival, settled RMSE, energy, thrust).

    `energy` is the benchmark's control-effort metric (mean squared throttle). `thrust` is the mean
    thrust the airframe actually produces, read off the plant's own `_Fb_thrust`: that is the
    quantity the lifting-line correction predicts, and the one that can be compared against the
    open-loop trim sweep of `run_trim`. The two are not interchangeable, because thrust is
    quadratic-with-linear-term in throttle (`compute_per_engine_thrust_force`)."""
    import torch
    import warp as wp
    from falcons.envs.configs import PLANE_CONFIGS
    from falcons.envs.altitude import AltitudeEnv
    from falcons.controllers.policies import load_policy

    cfg = PLANE_CONFIGS[aircraft]
    env = AltitudeEnv(N, "cuda", {"target_mode": "dynamic", "spawning_distance": 0.0,
                                  "aircraft": aircraft, "ref_rate": cfg["ref_rate"],
                                  "horizon": STEPS + 1, "in_ground_effect": ge,
                                  "i_clamp": cfg.get("i_clamp", 25.0)})
    env.reset()
    tgt = np.full(N, target, dtype=np.float32)
    pos = np.zeros((N, 3), dtype=np.float32); pos[:, 2] = -target      # spawn ON the reference:
    vel = np.zeros((N, 3), dtype=np.float32); vel[:, 0] = env.base_vel  # this is a hold task
    wp.copy(env._true_target, wp.array(tgt, device="cuda"))
    wp.copy(env._target, wp.array(tgt, device="cuda"))
    wp.copy(env.model._state["position"], wp.array(pos, dtype=wp.vec3f, device="cuda"))
    wp.copy(env.model._state["linear_vel"], wp.array(vel, dtype=wp.vec3f, device="cuda"))
    env._ierr.zero_(); env._refrate.zero_(); env._obs_launch()

    policy = load_policy("ppo", "altitude", aircraft, 0, ckpt_dir=ckpt_dir).act
    obs = wp.to_torch(env._obs)
    live = np.ones(N, dtype=bool)
    sq = np.zeros(N); en = np.zeros(N); th = np.zeros(N); cnt = 0
    with torch.no_grad():
        for t in range(STEPS):
            obs, _, d, _ = env.step(policy(obs).clamp(-1.0, 1.0))
            h = -env.model._state["position"].numpy()[:, 2]
            thr = env.model._actuator_states["throttle_left"].numpy()
            T = np.linalg.norm(env.model._Fb_thrust.numpy(), axis=1)
            live &= ~d.cpu().numpy().astype(bool)
            if t >= SETTLE:
                e = h - target
                sq += np.where(live, e * e, 0.0)
                en += np.where(live, thr * thr, 0.0)
                th += np.where(live, T, 0.0)
                cnt += 1
            if not live.any():
                break
    if cnt == 0 or not live.any():
        return 0.0, np.nan, np.nan, np.nan
    return (float(live.mean()), float(np.sqrt(np.mean(sq[live] / cnt))),
            float(np.mean(en[live] / cnt)), float(np.mean(th[live] / cnt)))


def run_energy(planes, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR):
    """What a trained policy spends (mean squared throttle) while holding each band.

    The comparison is PAIRED: the same policy holds the same commanded altitude on the same
    airframe, and the only thing that changes between the two arms is `in_ground_effect`. Any
    difference in the throttle it needs is therefore attributable to the ground-effect correction
    alone -- the policy's training distribution, the airframe's trim point and the controller's
    tuning all cancel. That is what makes it possible to report an energy result for policies that
    were trained at cruise.

    Three limits, which are why `run_trim` exists alongside it: the number reported here is a
    throttle-squared ratio while the lifting-line prediction is a THRUST ratio (the two differ
    because thrust is quadratic-with-linear-term in throttle, see `compute_per_engine_thrust_force`);
    the paired design cancels a constant controller bias but not a controller that flies the two
    arms differently; and a band is reported only where a policy can hold it, which is why the A0S
    loses its three deepest bands and the general-aviation planes lose all six. Airframes whose
    flight envelope does not reach a band (the fast general-aviation planes cannot cruise at a few
    metres) are reported as a survival failure, not omitted."""
    import warp as wp
    wp.init()
    os.makedirs(results_dir, exist_ok=True)
    rows = []
    for ac in planes:
        b, _mac = geometry(ac)
        for r in RATIOS:
            h = r * b
            if h < MIN_ALT:
                continue
            s_on, rmse_on, e_on, t_on = cell(ac, h, True, ckpt_dir)
            s_off, rmse_off, e_off, t_off = cell(ac, h, False, ckpt_dir)
            d = (100.0 * (e_on - e_off) / e_off) if e_off and np.isfinite(e_off) else np.nan
            dT = (100.0 * (t_on - t_off) / t_off) if t_off and np.isfinite(t_off) else np.nan
            th = thrust_saving(ac, r, b)
            # the paired energy comparison only means something if BOTH arms are actually holding
            # the commanded altitude; a policy that is gliding away has no trim throttle to report
            ok = int(min(s_on, s_off) > 0.5
                     and max(rmse_on, rmse_off) <= min(TRACK_TOL, TRACK_FRAC * h))
            rows.append(dict(aircraft=ac, span=round(b, 2), h_over_b=r, altitude=round(h, 2),
                             surv_ge_on=s_on, surv_ge_off=s_off,
                             rmse_ge_on=rmse_on, rmse_ge_off=rmse_off,
                             energy_ge_on=e_on, energy_ge_off=e_off, delta_energy_pct=d,
                             thrust_ge_on=t_on, thrust_ge_off=t_off, delta_thrust_pct=dT,
                             theory_pct=th, hold_valid=ok))
            print(f"{ac:17s} h/b {r:4.2f} (h {h:6.2f} m)  surv {s_on:.2f}/{s_off:.2f}  "
                  f"T {t_on:8.2f}/{t_off:8.2f} N  dT {dT:+6.1f}%  dE {d:+6.1f}%  "
                  f"theory {th:+6.1f}%  "
                  f"rmse {rmse_on:.3f}/{rmse_off:.3f}  {'hold' if ok else 'NOT-HOLDING'}",
                  flush=True)
    cols = ["aircraft", "span", "h_over_b", "altitude", "surv_ge_on", "surv_ge_off",
            "rmse_ge_on", "rmse_ge_off", "energy_ge_on", "energy_ge_off", "delta_energy_pct",
            "thrust_ge_on", "thrust_ge_off", "delta_thrust_pct", "theory_pct", "hold_valid"]
    out = Path(results_dir) / "ge_energy.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for x in rows:
            w.writerow(x)
    print(f"wrote {out}")
    return out


# figure/plot/replot (and the COLORS only they used) are not here: Task 8 ports them to figures.py.
