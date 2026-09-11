"""Every figure the benchmark reports, one function per figure, all writing into
<results_dir>/figures and returning the file they wrote.

    altitude_viz(algo, plane)   the altitude dashboard, one per (algo, airframe)
    attitude_exec(planes)       the command interface: commanded bank fan + all three channels
    maneuver_facet(plane)       small multiples of bank tracking, maneuvers x methods
    maneuver_render(planes)     the qualitative strip: eight 3-D tracks coloured by bank
    ge_energy()                 ground effect: thrust change and tracking accuracy against h/b
    robustness_ppo()            the PPO executor under four disturbance families

Run: render() -> <results_dir>/figures/*
"""
import csv
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from falcons.paths import CKPT_DIR, RESULTS_DIR

DT = 0.01
BANK_LIM = 50.0        # deg; the bank colour scale, shared by attitude_exec and maneuver_render
# One palette across every figure: the tracking figures name methods, the ground-effect figure names
# airframes. LQR purple, MPPI red, PPO (CAPS) blue, SAC green, TD3 orange.
METHOD_COLORS = {"LQR": "#8250c4", "MPPI": "#d1495b", "PPO": "#2a78d6", "SAC": "#008300",
                 "TD3": "#e8601c"}
PLANE_COLORS = {"Airship_V7": "#2a78d6", "Airship_A0S": "#d1495b", "Volantex_Ranger": "#008300",
                "Navion": "#8250c4"}
# The two airframes the attitude figures report, in panel order, with their printed names.
PRETTY = {"Airship_V7": "Airship V7", "Volantex_Ranger": "Volantex Ranger"}
SHAPE_PRETTY = {"circle": "circle", "figure8": "figure-8", "helix": "helix", "sturn": "s-turn"}


def _figdir(results_dir):
    d = Path(results_dir) / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ───────────────────────────── altitude dashboard ─────────────────────────────
# constants for this dashboard alone; `N` is its Monte Carlo batch, not a batch anywhere else.
BAND = 1.0        # settled = |alt - target| < BAND [m]
BASE = 60.0       # spawn altitude [m]
K = 1000          # 10 s of stable convergence before advancing to the next target
M = 5             # altitude waypoints per sequence
N = 256           # Monte Carlo batch
NTRACK = 10       # episodes recorded in full (trajectories)
H = 13000         # 130 s total


def _altitude_run(algo, plane, ckpt_dir):
    """Sequential altitude waypoints tracked by one learned policy, over N Monte Carlo episodes,
    NTRACK of them recorded in full."""
    import torch
    import warp as wp
    from falcons.envs.altitude import AltitudeEnv
    from falcons.envs.configs import PLANE_CONFIGS
    from falcons.controllers.policies import load_policy

    with torch.no_grad():
        rr = PLANE_CONFIGS[plane]["ref_rate"]
        env = AltitudeEnv(N, "cuda", {"target_mode": "dynamic", "spawning_distance": 0.0,
                                      "aircraft": plane, "ref_rate": rr, "horizon": H + 1,
                                      "i_clamp": PLANE_CONFIGS[plane].get("i_clamp", 25.0)})
        act = load_policy(algo, "altitude", plane, 0, ckpt_dir=ckpt_dir).act

        # per-env altitude sequences: realistic climb/descent steps
        seq = np.zeros((N, M), np.float32); cur = np.full(N, BASE)
        np.random.seed(42)
        for j in range(M):
            up = np.random.rand(N) < 0.5
            step = np.where(up, np.random.uniform(4, 8, N), -np.random.uniform(4, 18, N))
            cur = np.clip(cur + step, 25, 95); seq[:, j] = cur

        env.reset()
        pos0 = np.zeros((N, 3), np.float32); pos0[:, 2] = -BASE
        vel0 = np.zeros((N, 3), np.float32); vel0[:, 0] = env.base_vel
        wp.copy(env._true_target, wp.array(seq[:, 0].copy(), device="cuda"))
        wp.copy(env._target,      wp.array(np.full(N, BASE, np.float32), device="cuda"))
        wp.copy(env.model._state["position"],   wp.array(pos0, dtype=wp.vec3f, device="cuda"))
        wp.copy(env.model._state["linear_vel"], wp.array(vel0, dtype=wp.vec3f, device="cuda"))
        env._ierr.zero_(); env._refrate.zero_(); env._obs_launch()
        obs = wp.to_torch(env._obs)

        idx = np.zeros(N, int); sc = np.zeros(N, int); done = np.zeros(N, bool)
        frozen = np.zeros(NTRACK, bool)
        # full trajectory: (north, east, alt, alt_cmd, elevator, throttle)
        traj = [[] for _ in range(NTRACK)]
        inj = [[(0, float(seq[i, 0]))] for i in range(NTRACK)]   # injection events
        endstep = np.full(NTRACK, H)
        # MC arrays for all N episodes
        err_all = np.full((H, N), np.nan)

        for t in range(H):
            cur_tgt = seq[np.arange(N), np.minimum(idx, M - 1)]
            wp.copy(env._true_target, wp.array(cur_tgt, device="cuda"))

            p   = env.model._state["position"].numpy()
            ast = env.model._actuator_states
            el  = ast["elevator"].numpy()
            th  = ast["throttle_left"].numpy()
            alt = -p[:, 2]

            for i in range(NTRACK):
                if not frozen[i]:
                    traj[i].append((p[i, 0], p[i, 1], alt[i], cur_tgt[i], el[i], th[i]))

            err_all[t] = np.abs(alt - cur_tgt)

            hit = (np.abs(alt - cur_tgt) < BAND) & ~done
            sc[hit] += 1; sc[~hit & ~done] = 0
            adv = (sc >= K) & ~done; idx[adv] += 1; sc[adv] = 0
            for i in range(NTRACK):
                if adv[i] and not frozen[i] and idx[i] < M:
                    inj[i].append((t, float(seq[i, idx[i]])))
            done |= idx >= M

            obs, _, d, _ = env.step(act(obs).clamp(-1.0, 1.0))
            dd = d.cpu().numpy()
            for i in range(NTRACK):
                if dd[i] and not frozen[i]:
                    frozen[i] = True; endstep[i] = min(endstep[i], t)

        # settled RMSE: last 50 s for episodes that finished
        settled_rmse = []
        for i in range(NTRACK):
            tr = np.array(traj[i])
            if len(tr) > 5000:
                e = np.abs(tr[-5000:, 2] - tr[-5000:, 3])
                settled_rmse.append(np.sqrt(np.mean(e ** 2)))

        return dict(traj=traj, inj=inj, endstep=endstep, seq=seq,
                    err_all=err_all,
                    completion=(idx >= M).mean(),
                    mean_legs=idx.clip(max=M).mean(),
                    settled_rmse=np.nanmean(settled_rmse) if settled_rmse else np.nan)


def altitude_viz(algo, plane, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR):
    """Sequential altitude waypoints tracked by the dynamic altitude policy, over a 256-episode
    Monte Carlo. The only figure that flies anything -- it has no cache and takes about a minute.
    2x3 panels:
      (A) 3D flight paths, colour = altitude
      (B) altitude commanded vs achieved for NTRACK coloured episodes
      (C) Monte Carlo error bands (N=256, median + 10-90%)
      (D) side-view (North vs alt), colour = altitude
      (E) elevator + throttle (longest recorded episode)
      (F) statistics text box
    """
    import warp as wp
    from falcons.envs.configs import PLANE_CONFIGS
    wp.init()
    d = _altitude_run(algo, plane, ckpt_dir)
    traj = d["traj"]; inj = d["inj"]
    tm = np.arange(H) * DT
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, NTRACK))

    # altitude range across all tracked episodes → colour scale for 3D/side panels
    all_alts = np.concatenate([np.array(t)[:, 2] for t in traj if len(t) > 50])
    alt_lo = float(np.percentile(all_alts, 2))
    alt_hi = float(np.percentile(all_alts, 98))

    fig = plt.figure(figsize=(18, 11))

    # ---- A: 3D flight paths (colour = altitude) ----
    ax3 = fig.add_subplot(2, 3, 1, projection="3d")
    for i in range(min(6, NTRACK)):
        tr = np.array(traj[i])
        if len(tr) < 50:
            continue
        pts = tr[:, :3].reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = Line3DCollection(segs, cmap="viridis", lw=1.4, norm=plt.Normalize(alt_lo, alt_hi))
        lc.set_array(tr[:-1, 2]); ax3.add_collection3d(lc)
        ax3.scatter([tr[0, 0]], [tr[0, 1]], [tr[0, 2]], c="tab:green", s=30, marker="^")
    ax3.set_xlabel("North [m]"); ax3.set_ylabel("East [m]"); ax3.set_zlabel("alt [m]")
    ax3.set_title("3D flight paths  (colour = altitude [m])")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(alt_lo, alt_hi))
    sm.set_array(np.linspace(alt_lo, alt_hi, 100))

    # ---- B: altitude tracking, commanded vs achieved ----
    axb = fig.add_subplot(2, 3, 2)
    for i, c in enumerate(colors):
        tr = np.array(traj[i])
        if len(tr) < 50:
            continue
        t_ax = np.arange(len(tr)) * DT
        axb.plot(t_ax, tr[:, 2], color=c, lw=1.5, alpha=0.9)
        axb.plot(t_ax, tr[:, 3], color=c, lw=0.8, ls="--", alpha=0.5)
        ev = np.array(inj[i])
        axb.scatter(ev[:, 0] * DT, ev[:, 1], color=c, s=55, zorder=5,
                    marker="o", edgecolors="white", linewidths=1.0)
    axb.set_xlabel("time [s]"); axb.set_ylabel("altitude [m]")
    axb.set_title("altitude tracking  (— achieved,  -- commanded,  ● injected)")
    axb.grid(alpha=0.25); axb.set_ylim(15, 105)
    for sp in ("top", "right"): axb.spines[sp].set_visible(False)

    # ---- C: Monte Carlo error bands (N=256 episodes) ----
    axc = fig.add_subplot(2, 3, 3)
    err = d["err_all"]
    with np.errstate(all="ignore"):
        med = np.nanmedian(err, axis=1)
        lo  = np.nanpercentile(err, 10, axis=1)
        hi  = np.nanpercentile(err, 90, axis=1)
    valid = ~np.isnan(med)
    axc.fill_between(tm[valid], lo[valid], hi[valid], color="tab:blue", alpha=0.20)
    axc.plot(tm[valid], med[valid], color="tab:blue", lw=1.8, label=f"median  (N={N})")
    axc.axhline(BAND, color="k", ls="--", lw=1.0, label=f"±{BAND:.0f} m capture band")
    axc.set_xlabel("time [s]"); axc.set_ylabel("|alt − cmd| [m]")
    axc.set_title(f"tracking error — {N} Monte Carlo episodes (10–90% band)")
    axc.grid(alpha=0.25); axc.legend(fontsize=9); axc.set_ylim(bottom=0)
    for sp in ("top", "right"): axc.spines[sp].set_visible(False)

    # ---- D: side view (North vs alt), colour = altitude ----
    axd = fig.add_subplot(2, 3, 4)
    for i in range(min(6, NTRACK)):
        tr = np.array(traj[i])
        if len(tr) < 50:
            continue
        pts = np.column_stack([tr[:, 0], tr[:, 2]]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", lw=1.4, norm=plt.Normalize(alt_lo, alt_hi))
        lc.set_array(tr[:-1, 2]); axd.add_collection(lc)
    axd.autoscale()
    axd.set_xlabel("North [m]"); axd.set_ylabel("altitude [m]")
    axd.set_title("side view  (colour = altitude [m])"); axd.grid(alpha=0.25)
    for sp in ("top", "right"): axd.spines[sp].set_visible(False)

    # ---- E: elevator + throttle (longest recorded episode) ----
    axe = fig.add_subplot(2, 3, 5)
    ri = max(range(NTRACK), key=lambda i: len(traj[i]))
    tr = np.array(traj[ri]); t_r = np.arange(len(tr)) * DT
    axe.plot(t_r, tr[:, 4], color="tab:blue", lw=1.2, label="elevator")
    axe2 = axe.twinx()
    axe2.plot(t_r, tr[:, 5], color="tab:orange", lw=1.2, label="throttle")
    axe2.set_ylim(0, 1.05); axe2.set_ylabel("throttle [0–1]", color="tab:orange")
    axe.set_xlabel("time [s]"); axe.set_ylabel("elevator [deg]", color="tab:blue")
    axe.set_title("controls — representative episode"); axe.grid(alpha=0.25)
    lines1, lbl1 = axe.get_legend_handles_labels()
    lines2, lbl2 = axe2.get_legend_handles_labels()
    axe.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8, loc="upper right")
    for sp in ("top", "right"): axe.spines[sp].set_visible(False)

    # ---- F: statistics ----
    axf = fig.add_subplot(2, 3, 6)
    axf.axis("off")
    stats = (
        f"Random altitude-waypoint generation\n"
        f"and Monte Carlo Simulations\n\n"
        f"N = {N} episodes × {M} altitude targets\n\n"
        f"Full completion:   {d['completion']*100:.0f}%\n"
        f"Mean captured:     {d['mean_legs']:.2f} / {M}\n"
        f"Settled RMSE:      {d['settled_rmse']:.2f} m\n\n"
        f"Constraints:\n"
        f"  Altitude rate  < {PLANE_CONFIGS[plane]['ref_rate']:.1f} m/s\n"
        f"  No stall / crash"
    )
    axf.text(0.08, 0.55, stats, fontsize=11.5, transform=axf.transAxes,
             verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", fc="0.97", ec="0.7", lw=1))

    # attach to the side view alone: spanning two panels in different rows leaves the bar
    # floating in the middle of the figure
    fig.colorbar(sm, ax=axd, fraction=0.03, pad=0.02).set_label("altitude [m]")
    # No suptitle: wherever this figure is printed, the caption carries the description.
    # The run summary that used to live in the title is in the statistics panel (axf) instead.
    fig.tight_layout()
    out = _figdir(results_dir) / f"altitude_{algo}_viz_{plane}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (completion {d['completion']*100:.0f}%, "
          f"RMSE {d['settled_rmse']:.2f} m)")
    return out


# ───────────────────────────── attitude executor ─────────────────────────────
AE_N = 64          # demo batch: the fan of commanded bank angles
AE_H = 4500        # 45 s (a ~30 deg-bank turn is about one full circle in 30 s)
CH = [(3, 4, "$\\phi$", "$^\\circ$"), (5, 6, "$\\dot h$", "m/s"),
      (7, 8, "$V_a$", "m/s")]               # (achieved col, commanded col, symbol, unit)
CH_C = ["#2a78d6", "#008300", "#e8601c"]


def _attitude_demo(plane, ckpt_dir):
    """The scripted demo episodes behind the command-interface figure: each episode holds a coordinated
    turn at its own bank angle (hdot* = 0, so altitude is held) for the first two thirds, then
    levels and climbs. PPO seed 0 drives it.

    Columns per step, as recorded:
    (N, E, alt, roll_deg, phi_cmd_deg, hdot, hdot_cmd, Va, Va_cmd, elevator, aileron, rudder,
     throttle).
    """
    import torch
    import warp as wp
    from falcons.envs.attitude import AttitudeEnv
    from falcons.envs.configs import attitude_env_cfg
    from falcons.controllers.policies import load_policy

    def rolldeg(q):
        qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return np.degrees(np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2)))

    def climb_rate(q, vbody):
        # world vertical speed vz = (R(q) v_body)[2]; climb rate = -vz  (NED z-down)
        t = 2.0 * np.cross(q[:, :3], vbody)
        vw = vbody + q[:, 3:4] * t + np.cross(q[:, :3], t)
        return -vw[:, 2]

    with torch.no_grad():
        env = AttitudeEnv(AE_N, "cuda", attitude_env_cfg(plane, horizon=AE_H + 1))
        act = load_policy("ppo", "attitude", plane, 0, ckpt_dir=ckpt_dir).act

        banks = np.linspace(-env.phi_max_full, env.phi_max_full, AE_N).astype(np.float32)
        climb = min(1.0, env.hdot_max_full)                   # gentle climb within envelope
        zero = np.zeros(AE_N, np.float32)
        phi3 = np.stack([banks, banks, zero], axis=1)
        hd3 = np.stack([zero, zero, np.full(AE_N, climb, np.float32)], axis=1)
        va3 = np.full((AE_N, 3), env.base_vel, np.float32)

        env.reset()
        wp.copy(env._phi3, wp.array(phi3, device="cuda"))
        wp.copy(env._hdot3, wp.array(hd3, device="cuda"))
        wp.copy(env._va3, wp.array(va3, device="cuda"))
        env._select_launch(); env._obs_launch()
        obs = wp.to_torch(env._obs)

        # record NTRACK episodes SPREAD across the bank fan (so both left and right turns show)
        track_idx = np.linspace(0, AE_N - 1, NTRACK).astype(int)
        frozen = np.zeros(NTRACK, bool)
        traj = [[] for _ in range(NTRACK)]
        for t in range(AE_H):
            s = env.model._state
            p = s["position"].numpy(); q = s["orientation"].numpy(); vb = s["linear_vel"].numpy()
            ast = env.model._actuator_states
            roll = rolldeg(q); climb_r = climb_rate(q, vb)
            Va = env.model._Va.numpy()
            phi_cmd = np.degrees(wp.to_torch(env._phi_tgt).cpu().numpy())
            hdot_cmd = wp.to_torch(env._hdot_tgt).cpu().numpy()
            va_cmd = wp.to_torch(env._va_tgt).cpu().numpy()
            el = ast["elevator"].numpy(); ai = ast["aileron"].numpy()
            ru = ast["rudder"].numpy(); th = ast["throttle_left"].numpy()
            for k, i in enumerate(track_idx):
                if not frozen[k]:
                    traj[k].append((p[i, 0], p[i, 1], -p[i, 2], roll[i], phi_cmd[i],
                                    climb_r[i], hdot_cmd[i], Va[i], va_cmd[i],
                                    el[i], ai[i], ru[i], th[i]))
            obs, _, d, _ = env.step(act(obs).clamp(-1.0, 1.0))
            dd = d.cpu().numpy()
            for k, i in enumerate(track_idx):
                if dd[i]:
                    frozen[k] = True
    # np.array over the recorded tuples, with no dtype of its own: every field is a float32 scalar
    # off a warp array, so the row array is float32
    return [np.array(t) for t in traj]


def _attitude_rollout(plane, ckpt_dir, results_dir):
    """Recorded demo episodes for `plane`, cached on disk so a shipped cache replays without
    flying. The headline numbers are computed from the demo itself (`rob` is the fraction of the
    recorded episodes that ran the whole horizon); they are printed, never drawn."""
    path = Path(results_dir) / "traces" / f"attitude_exec_{plane}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    traj = _attitude_demo(plane, ckpt_dir)
    kept = [t for t in traj if len(t) > 50]
    rmse = lambda ia, ic: float(np.sqrt(np.mean(np.concatenate(
        [(t[:, ia] - t[:, ic]) ** 2 for t in kept]))))
    out = dict(traj=kept, dt=DT, rob=float(np.mean([len(t) >= AE_H for t in traj])),
               phi=rmse(3, 4), hdot=rmse(5, 6), va=rmse(7, 8))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def _fan(ax, traj):
    """Ground tracks over the commanded bank fan, coloured by achieved bank."""
    for tr in traj:
        pts = tr[:, :2].reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="coolwarm", lw=1.0,
                            norm=plt.Normalize(-BANK_LIM, BANK_LIM))
        lc.set_array(tr[:-1, 3])
        ax.add_collection(lc)
    ax.scatter([traj[0][0, 0]], [traj[0][0, 1]], c="k", s=14, marker="^", zorder=5)
    # Frame on the turns. The shallowest commanded banks fly almost straight and leave the area the
    # turning episodes occupy; autoscaling to them shrinks every circle to a dot. The radius of the
    # widest CLOSED turn sets the window, and the shallow tracks simply run out of frame.
    closed = [t for t in traj if np.hypot(*(t[-1, :2] - t[0, :2])) < 0.5 * np.ptp(t[:, :2])]
    span = max((np.abs(t[:, :2] - traj[0][0, :2]).max() for t in (closed or traj)), default=1.0)
    ax.set_xlim(traj[0][0, 0] - 1.15 * span, traj[0][0, 0] + 1.15 * span)
    ax.set_ylim(traj[0][0, 1] - 1.15 * span, traj[0][0, 1] + 1.15 * span)
    ax.set_aspect("equal")
    ax.set_xlabel("North [m]", fontsize=7); ax.set_ylabel("East [m]", fontsize=7)
    ax.tick_params(labelsize=6.5); ax.grid(alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _channels(ax, traj, dt):
    """The three commanded channels of one episode, each in its own lane.

    Lanes are half-height (traces span at most +-0.5 of the 2.0 lane pitch), which keeps a clear
    strip above every lane for its label whatever shape the channel takes, and each is normalised
    by its own range because the three carry different units. The label carries the physical
    numbers the normalisation removes.
    """
    # The steepest commanded turn that runs the whole episode. The fan spans the airframe's full
    # bank envelope and its outermost members depart within seconds, so picking on commanded bank
    # alone lands on a two-second trace.
    full = max(len(t) for t in traj)
    mid = max([t for t in traj if len(t) > 0.95 * full], key=lambda t: abs(t[len(t) // 4, 4]))
    t = np.arange(len(mid)) * dt
    for lane, ((ia, ic, lab, unit), col) in enumerate(zip(CH, CH_C)):
        a, c = mid[:, ia], mid[:, ic]
        span = max(np.ptp(np.concatenate([a, c])), 1e-6)
        base, ref = 2.0 * (2 - lane), np.median(c)
        ax.plot(t, base + 0.5 * (a - ref) / span, color=col, lw=1.1)
        ax.plot(t, base + 0.5 * (c - ref) / span, color=col, lw=0.9, ls="--", alpha=0.75)
        lo, hi = float(c[len(c) // 4]), float(c[-1])
        ax.text(0.0, base + 0.62, f"{lab}: {lo:.0f}$\\rightarrow${hi:.0f} {unit}",
                color=col, fontsize=6, ha="left", va="bottom")
    ax.set_xlabel("time [s]", fontsize=7)
    ax.set_yticks([]); ax.set_ylim(-0.9, 5.5)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def _colour_key(fig, ax):
    """Bank-angle key inside the fan panel, where it cannot be read as a label belonging to the
    panel beside it."""
    cax = ax.inset_axes([0.05, 0.11, 0.34, 0.035])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="coolwarm",
                                            norm=plt.Normalize(-BANK_LIM, BANK_LIM)),
                      cax=cax, orientation="horizontal", ticks=[-BANK_LIM, 0, BANK_LIM])
    cb.set_label("bank [deg]", fontsize=6, labelpad=2)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=5.5, pad=1)
    cb.outline.set_linewidth(0.4)


def attitude_exec(planes, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR):
    """The command interface itself: (A) top-down ground tracks over a fan of commanded bank angles
    spanning the airframe's envelope, coloured by achieved bank -- the executor as a continuum, not
    four demos; (B) all three commanded channels (phi*, hdot*, Va*) against what was achieved, on
    one time axis, each normalised by its own commanded step so they share a vertical scale.

    One airframe -> a single-column two-panel figure. Two -> the airframes side by side across the
    page width, each contributing its fan and its channel lanes."""
    ds = [(ac, _attitude_rollout(ac, ckpt_dir, results_dir)) for ac in planes]
    n = len(ds)
    # constrained_layout, not tight_layout: the fan panels carry an equal aspect ratio, so their
    # drawn box is smaller than the cell they sit in, and tight_layout places the neighbouring
    # x-labels against the nominal cell instead and pushes them off the canvas.
    fig, ax = plt.subplots(1, 2 * n, figsize=(7.16 if n > 1 else 3.5, 2.35 if n > 1 else 1.9),
                           layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.06, hspace=0.0)
    for k, (ac, d) in enumerate(ds):
        _fan(ax[2 * k], d["traj"])
        _channels(ax[2 * k + 1], d["traj"], d["dt"])
        _colour_key(fig, ax[2 * k])
        if n > 1:                        # name the pair, since two airframes now share the figure
            ax[2 * k].set_title(ac.replace("_", " "), fontsize=8, loc="left", pad=3)
            if k:
                ax[2 * k].set_ylabel("")
        print(f"  {ac}: rob {d['rob']:.3f}, phi {d['phi']:.2f} deg, "
              f"hdot {d['hdot']:.2f} m/s, Va {d['va']:.2f} m/s")
    name = f"fig_attitude_exec_{planes[0]}" if n == 1 else "fig_attitude_exec"
    out = _figdir(results_dir) / f"{name}.pdf"
    # constrained_layout solves at draw time, against text extents that depend on the dpi, and this
    # figure -- equal-aspect panels carrying an inset colourbar -- takes a second pass at the
    # output dpi to settle. Ask for that pass directly, rather than get it as a side effect of
    # writing a PNG nothing reads before the PDF.
    fig.set_dpi(300); fig.canvas.draw()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")
    return out


# ───────────────────────────── maneuver small multiples ─────────────────────────────
KDASH = "0.15"        # commanded dashed line
BAND_GREY = "0.6"     # tolerance-band fill


def _maneuver_runs(plane, ckpt_dir, results_dir):
    """Every (maneuver, method) cell of one airframe, read back from the rollout cache
    falcons.benchmark.maneuvers writes. Nothing re-flies while that cache is warm."""
    from falcons.benchmark.maneuvers import SHAPES, cell, controllers
    from falcons.envs.configs import attitude_env_cfg
    tol_deg = np.degrees(attitude_env_cfg(plane).get("sig_phi", 0.15))   # tracking tolerance band
    methods = list(controllers(plane, ckpt_dir))
    cache = Path(results_dir) / "traces"
    runs = {sh: {m: cell(plane, sh, m, ckpt_dir=ckpt_dir, cache_dir=cache, refresh=False,
                         tol_deg=tol_deg) for m in methods} for sh in SHAPES}
    return runs, methods, tol_deg


def maneuver_facet(plane, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR):
    """Small-multiples grid of BANK tracking: rows = maneuvers, columns = methods, one trace per
    cell over the commanded reference + tolerance band. This is the section's only time-series
    figure.

    Why not the overlay: SAC and TD3 are bang-bang on these streams (ReLU actors, no CAPS in the
    loss), so on a shared axis their traces cover the three tight trackers entirely. Faceting keeps
    every method at full opacity and turns the jitter into a legible per-method finding.

    Why bank only: the climb channel is commanded non-zero in the helix alone (streams.py) --- the
    other three maneuvers command hdot* = 0 throughout, so a climb facet spends fifteen of twenty
    panels showing a hold at zero. Its one finding, that SAC and TD3 inject 1-3 m/s of vertical
    oscillation while tracking bank, is the hdot_rmse column of the table.
    """
    from falcons.benchmark.maneuvers import SHAPES, time_in_band
    runs_by_shape, methods, tol_deg = _maneuver_runs(plane, ckpt_dir, results_dir)
    key, cmd = "roll", "phi_cmd"
    fig, ax = plt.subplots(len(SHAPES), len(methods), figsize=(2.35 * len(methods), 8.4),
                           sharex="row", sharey="row", squeeze=False)
    for i, sh in enumerate(SHAPES):
        runs = runs_by_shape[sh]
        ref = max((runs[m]["trace"] for m in methods
                   if runs[m]["trace"] is not None and len(runs[m]["trace"]["t"])),
                  key=lambda d: len(d["t"]), default=None)
        for j, m in enumerate(methods):
            a = ax[i, j]
            if ref is not None:
                a.fill_between(ref["t"], ref[cmd] - tol_deg, ref[cmd] + tol_deg,
                               color=BAND_GREY, alpha=0.18, lw=0)
                a.plot(ref["t"], ref[cmd], "--", color=KDASH, lw=1.0)
            r = runs[m]
            tr = r["trace"]
            if tr is not None and len(tr["t"]):
                a.plot(tr["t"], tr[key], color=METHOD_COLORS[m], lw=1.0)
            if r["survival"] == 0:
                a.text(0.5, 0.9, f"crash @ {r['t_fail']:.0f} s", transform=a.transAxes,
                       fontsize=7.5, ha="center", color="#a33")
            elif r["phi"] is not None:
                nr = r["phi"]["nrmse"]
                a.text(0.03, 0.03,
                       (f"nRMSE {nr*100:.0f}%\n" if nr is not None else "")
                       + f"band {time_in_band(tr, tol_deg)*100:.0f}%",
                       transform=a.transAxes, fontsize=7, family="monospace", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.85", alpha=0.85))
            if i == 0:
                a.set_title(m, color=METHOD_COLORS[m], fontsize=11, pad=4)
            if j == 0:
                a.set_ylabel(sh, fontsize=10)
            a.grid(alpha=0.2); a.tick_params(labelsize=7.5)
            for sp in ("top", "right"): a.spines[sp].set_visible(False)
    for j in range(len(methods)):
        ax[-1, j].set_xlabel("time [s]", fontsize=9)
    fig.tight_layout()
    out = _figdir(results_dir) / f"fig_maneuver_facet_{plane}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    return out


# ───────────────────────────── qualitative 3-D strip ─────────────────────────────
PAD = 8                 # px of margin kept around a frame's ink bounding box


def aircraft_glyph(pos, yaw, roll, s):
    """Small aircraft glyph at pos: fuselage (along heading), banked wings, tailplane.
    s = glyph size [m]. Returns dict of line endpoint pairs + the nose point."""
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])       # heading
    lat0 = np.array([-np.sin(yaw), np.cos(yaw), 0.0])     # level right-wing (starboard)
    up = np.array([0.0, 0.0, 1.0])                        # plot up (= -NED_down = +altitude)
    # positive roll dips the RIGHT wing toward the ground (-altitude): -sin(roll)*up, not +.
    wing = np.cos(roll) * lat0 - np.sin(roll) * up        # right-wing rotated by bank
    fin = -np.sin(roll) * lat0 + np.cos(roll) * up        # fuselage "up" (fin leans to raised wing)
    nose = pos + 1.1 * s * fwd
    tail = pos - 0.9 * s * fwd
    return {
        "fuselage": (tail, nose),
        "wings": (pos + 0.9 * s * wing, pos - 0.9 * s * wing),   # through cg
        "tail": (tail, tail + 0.45 * s * fin),                  # vertical fin at the back
        "nose": nose,
    }


def _ppo_track(plane, maneuver, ckpt_dir, results_dir):
    """The PPO seed-0 rollout of one cell, with the flight path the simulator actually flew, or
    None when there is no track to draw.

    `cell` owns the rollout cache and its file naming, so it is asked first -- and then the seed-0
    entry is read back out of the cache rather than taken as `cell` returned it: a cell flown over
    several training seeds comes back aggregated, and the trace an aggregate keeps is the
    median-RMSE seed. This figure draws one flight per panel, and which flight that is should not
    depend on how the other seeds happened to score.

    A run cached before `fly` recorded `pos`/`yaw` carries every channel the table scores and no
    path; rather than rewrite that cache -- the table reproduces itself from it byte for byte, and
    the shipped trace set is exactly such a cache -- that cell is re-flown at PPO seed 0 and used
    in memory. Nothing is written back.
    """
    from falcons.benchmark.maneuvers import ALGO, cell, controllers, fly
    from falcons.envs.configs import attitude_env_cfg
    tol_deg = np.degrees(attitude_env_cfg(plane).get("sig_phi", 0.15))
    cache = Path(results_dir) / "traces"
    entries = controllers(plane, ckpt_dir)["PPO"]
    suffix, seed0 = next(((s, f) for s, f in entries if s == ""), entries[0])   # "" is seed 0
    cell(plane, maneuver, "PPO", ckpt_dir=ckpt_dir, cache_dir=cache, refresh=False,
         tol_deg=tol_deg)                                       # fills the cache; nothing re-flies
    with open(cache / f"{plane}_{maneuver}_{ALGO['PPO']}{suffix}.pkl", "rb") as f:
        got = pickle.load(f)
    trace = (got[0] if isinstance(got, list) else got).get("trace")
    if trace is None or "pos" not in trace:
        print(f"  {plane}/{maneuver}: cached run predates the pos channel -- re-flying PPO seed 0")
        trace = fly(plane, maneuver, seed0)["trace"]
    # a run that departed on its first step leaves an empty trace: there is nothing to plot, and
    # _track_axes would divide the axis limits by a zero span
    return trace if trace is not None and trace["pos"].shape[0] else None


def _track_axes(ax, tr):
    """One finished maneuver: the flown track coloured by bank, the aircraft at the pose it ended
    in. `pos` is NED, so altitude is -down. Axis scaling and the glyph match the attitude
    animation's frame, at the camera angle its slow orbit ends on."""
    north, east = tr["pos"][:, 0], tr["pos"][:, 1]
    alt, roll_deg = -tr["pos"][:, 2], tr["roll"]
    # glyph size: a modest fraction of the path, clamped so it always reads as a small plane
    gs = float(np.clip(0.035 * max(np.ptp(north), np.ptp(east), 1.0), 7.0, 16.0))

    pts = np.column_stack([north, east, alt]).reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    trail = Line3DCollection(segs, cmap="coolwarm", lw=2.2,
                             norm=plt.Normalize(-BANK_LIM, BANK_LIM))
    trail.set_array(roll_deg[:-1])
    ax.add_collection3d(trail)

    g = aircraft_glyph(np.array([north[-1], east[-1], alt[-1]]), tr["yaw"][-1],
                       np.radians(roll_deg[-1]), gs)
    for key, col, lw in (("fuselage", "0.15", 3), ("wings", "tab:blue", 3), ("tail", "0.15", 2)):
        a, b = g[key]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=col, lw=lw,
                solid_capstyle="round")
    ax.plot([g["nose"][0]], [g["nose"][1]], [g["nose"][2]], color="tab:red", marker="o",
            ms=5, ls="")

    pad = 0.1 * max(np.ptp(north), np.ptp(east), 1.0)
    rx, ry = np.ptp(north) + 2 * pad, np.ptp(east) + 2 * pad
    # equal-ish axis UNITS so the altitude axis isn't stretched (which made the glyph look
    # vertical); give z enough span to fit the glyph + show the climb honestly.
    z_span = max(np.ptp(alt) + 4 * gs, 7 * gs)
    zc = 0.5 * (alt.min() + alt.max())
    ax.set_xlim(north.min() - pad, north.max() + pad)
    ax.set_ylim(east.min() - pad, east.max() + pad)
    ax.set_zlim(zc - 0.5 * z_span, zc + 0.5 * z_span)
    ax.set_box_aspect((rx, ry, z_span))
    ax.set_xlabel("North [m]"); ax.set_ylabel("East [m]"); ax.set_zlabel("altitude [m]")
    ax.view_init(elev=28, azim=30)      # where the animation's slow orbit ends


def _frame(tr):
    """One maneuver rendered on a canvas of its own and cropped to the bounding box of its ink.

    A 3-D axes reserves a wide margin around its box, and four of them tiled straight into one
    small figure come out as postage stamps in a sea of white, so each frame is cropped to its ink
    before it is tiled.
    """
    fig = plt.figure(figsize=(11, 9), dpi=110)     # the attitude animation's frame size
    _track_axes(fig.add_subplot(111, projection="3d"), tr)
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    ink = (a < 245).any(axis=2)
    ys, xs = np.where(ink)
    if not len(ys):
        return a
    return a[max(0, ys.min() - PAD):ys.max() + PAD, max(0, xs.min() - PAD):xs.max() + PAD]


def _panel(plane, maneuver, ckpt_dir, results_dir):
    """One cropped frame of the strip, or None when that cell has no track: the panel is then left
    empty, keeping the grid and its labels intact rather than failing the whole figure."""
    tr = _ppo_track(plane, maneuver, ckpt_dir, results_dir)
    return None if tr is None else _frame(tr)


def maneuver_render(planes, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR):
    """Qualitative 3-D strip for the maneuver section: the finished ground track of every named
    maneuver, coloured by bank angle, with the aircraft drawn at the pose it ended in. One row per
    airframe, maneuver names along the top and one shared bank-angle key underneath.

    Each frame is drawn directly from the recorded rollout and cropped to its ink. The alternative
    -- rendering an animation per cell and pulling its last frame out of the mp4 -- makes the figure
    a by-product of a video nothing else needs, and those frames carry a HUD and a per-frame
    colourbar that have to be painted back out again."""
    from falcons.benchmark.maneuvers import SHAPES
    rows = [(ac, [(sh, _panel(ac, sh, ckpt_dir, results_dir)) for sh in SHAPES])
            for ac in planes]
    ncol = max(len(f) for _, f in rows)
    cbar_h = 0.12
    fig, ax = plt.subplots(len(rows), ncol, figsize=(3.1 * ncol, 2.5 * len(rows)), squeeze=False)
    for r, (ac, frames) in enumerate(rows):
        for c in range(ncol):
            a = ax[r, c]
            a.axis("off")
            if c < len(frames) and frames[c][1] is not None:
                a.imshow(frames[c][1])
    fig.tight_layout(rect=[0.02, cbar_h, 1, 0.965])

    # Maneuver names on ONE baseline: each frame is cropped to its own ink box, so the axes end up
    # at different heights and per-axes titles would sit at four different heights.
    for c, (sh, _) in enumerate(rows[0][1]):
        pos = ax[0, c].get_position()
        fig.text(pos.x0 + pos.width / 2, 0.972, SHAPE_PRETTY.get(sh, sh),
                 ha="center", va="bottom", fontsize=10)
    for r, (ac, _) in enumerate(rows):
        pos = ax[r, 0].get_position()
        fig.text(0.012, pos.y0 + pos.height / 2, ac.replace("_", " "),
                 ha="left", va="center", rotation=90, fontsize=9)

    # one key for the whole strip: the track colour is the bank angle, blue left, red right
    cax = fig.add_axes([0.36, cbar_h * 0.55, 0.28, cbar_h * 0.16])
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(-BANK_LIM, BANK_LIM))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[-BANK_LIM, 0, BANK_LIM])
    cb.set_label("bank angle [deg]:  left turn  ←  0  →  right turn", fontsize=8,
                 labelpad=4)
    cb.ax.xaxis.set_label_position("top")        # label above the bar, ticks below it
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.6)
    out = _figdir(results_dir) / "fig_maneuver_render.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")
    return out


# ───────────────────────────── ground effect ─────────────────────────────
RATIOS = [0.25, 0.5, 1.0, 2.0, 3.0, 6.0]


def ge_energy(results_dir=RESULTS_DIR):
    """Single-column figure, the two panels stacked. Top: change in thrust from ground effect, for
    the closed-loop policy (solid), for the open-loop trim of `ground_effect.run_trim` (dash-dot,
    every airframe at every band because it needs no policy) and for the lifting-line prediction
    (dotted). All three are thrust ratios, so they sit on one axis; the throttle-squared
    control-effort metric stays in the CSV but is not plotted against the prediction, being a
    different quantity. Bottom: tracking accuracy of the two arms, log-scaled because the three
    airframes' errors span two decades. No titles: the caption carries the description.

    Drawn at its final printed width so it is included at \\columnwidth without rescaling; scaling
    a figure rescales every font in it.
    """
    d = Path(results_dir)
    with open(d / "ge_energy.csv") as f:
        rows = [{k: (v if k == "aircraft" else float(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]

    trim = {}
    tp = d / "ge_trim.csv"
    if tp.exists():
        with open(tp) as f:
            for r in csv.DictReader(f):
                trim.setdefault(r["aircraft"], []).append(
                    (float(r["h_over_b"]), float(r["delta_thrust_pct"]), float(r["theory_pct"])))

    fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.1), sharex=True, layout="constrained")
    fig.get_layout_engine().set(h_pad=0.02, hspace=0.03)
    planes = []
    for ac in dict.fromkeys(r["aircraft"] for r in rows):
        c = PLANE_COLORS.get(ac)
        rs = [r for r in rows if r["aircraft"] == ac and r["hold_valid"]]
        if not rs:
            continue          # no policy holds any band: its trim curve would have no counterpart
        planes.append((ac, c, rs[0]["span"]))
        x = [r["h_over_b"] for r in rs]
        ax[0].plot(x, [r["delta_thrust_pct"] for r in rs], "o-", color=c, lw=1.4, ms=3)
        ax[1].plot(x, [r["rmse_ge_on"] for r in rs], "o-", color=c, lw=1.4, ms=3)
        ax[1].plot(x, [r["rmse_ge_off"] for r in rs], "o--", color=c, lw=1.0, ms=2.5, alpha=0.6)
        t = sorted(trim.get(ac, []))
        if t:
            ax[0].plot([v[0] for v in t], [v[1] for v in t], "-.", color=c, lw=1.2)
            ax[0].plot([v[0] for v in t], [v[2] for v in t], ":", color=c, lw=1.0, alpha=0.8)
    ax[0].axhline(0, color="0.5", lw=0.7)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    # Label the bands themselves. The default log locator puts a single decade tick at h/b = 1
    # across this 0.25-6 range, which leaves the reader no way to place the deep-ground-effect end.
    ax[1].set_xticks(RATIOS); ax[1].set_xticklabels([f"{r:g}" for r in RATIOS])
    ax[1].set_xticks([], minor=True)
    ax[1].set_xlabel("height over span  $h/b$", fontsize=8)
    ax[0].set_ylabel("thrust change [%]", fontsize=8)
    ax[1].set_ylabel("settled altitude RMSE [m]", fontsize=8)
    ax[1].set_ylim(top=ax[1].get_ylim()[1] * 3.2)      # headroom for the airframe key
    # Two keys, one per panel: the upper names the three measurements, the lower the airframes,
    # so neither panel carries a legend for something it does not draw.
    ax[0].legend(handles=[Line2D([], [], color="0.25", lw=1.4, marker="o", ms=3, label="policy"),
                          Line2D([], [], color="0.25", lw=1.2, ls="-.", label="open-loop trim"),
                          Line2D([], [], color="0.25", lw=1.0, ls=":", label="lifting-line")],
                 fontsize=6.5, frameon=False, loc="lower right", handlelength=1.8,
                 borderpad=0.1, labelspacing=0.25)
    ax[1].legend(handles=[Line2D([], [], color=c, lw=1.4,
                                 label=f"{a.replace('_', ' ')} ($b$={b:.2f} m)")
                          for a, c, b in planes]
                 + [Line2D([], [], color="0.25", lw=1.0, ls="--", alpha=0.6, label="GE off")],
                 fontsize=6.5, frameon=False, loc="upper left", handlelength=1.8,
                 borderpad=0.1, labelspacing=0.25)
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
        a.tick_params(labelsize=7)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    out = _figdir(results_dir) / "fig_ge_energy.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return out


# ───────────────────────────── robustness ─────────────────────────────
X = ["nom", "mild", "medium", "severe"]
# Order is the finding: noise costs nothing, the two latencies cost the same, wind is the wall.
FAM = {"obs_noise": ("sensor noise", "#2a78d6"), "sensor_delay": ("sensor delay", "#008300"),
       "action_delay": ("action delay", "#e8601c"), "wind": ("turbulence", "#d1495b")}


def robustness_ppo(results_dir=RESULTS_DIR):
    """Single-column robustness figure: PPO only, both airframes, one line per disturbance family
    against severity. `benchmark.robustness` writes the full sweep; this figure needs the compact
    version, with the families as lines so that the ordering between them is the thing the reader
    sees. Reads the stored CSV, so it never re-flies anything."""
    with open(Path(results_dir) / "robustness.csv") as f:
        rs = [r for r in csv.DictReader(f) if r["method"] == "PPO"]
    idx = {(r["aircraft"], r["disturbance"], r["severity"]): r for r in rs}

    fig, ax = plt.subplots(1, 2, figsize=(3.45, 1.95), sharey=True)
    for a, ac in zip(ax, PRETTY):
        nom = idx[(ac, "nominal", "-")]
        for fam, (lab, c) in FAM.items():
            xs, ys = [0], [float(nom["phi_rmse"])]
            for i, sev in enumerate(["mild", "medium", "severe"], start=1):
                r = idx.get((ac, fam, sev))
                if r is None or not r["phi_rmse"]:
                    continue                      # departed: no tracking metric to plot
                xs.append(i); ys.append(float(r["phi_rmse"]))
            a.plot(xs, ys, "o-", color=c, lw=1.3, ms=3, label=lab)
        # Survival is the second axis of the result and leaves 1.00 in only two cells. Rather than
        # spend a panel on it, flag those cells along the top of the plot; a line that simply stops
        # (Ranger turbulence) is a cell where every maneuver departed and there is no RMSE to draw.
        for fam, (_, c) in FAM.items():
            for i, sev in enumerate(["mild", "medium", "severe"], start=1):
                r = idx.get((ac, fam, sev))
                if r and float(r["survival"]) < 1.0:
                    a.text(i, 0.97, f"s={float(r['survival']):.2f}",
                           transform=a.get_xaxis_transform(),
                           ha="center", va="top", fontsize=5.5, color=c)
        a.set_title(PRETTY[ac], fontsize=7.5, pad=3)
        a.set_xticks(range(4)); a.set_xticklabels(X, fontsize=6.5)
        a.tick_params(axis="y", labelsize=6.5)
        a.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].set_ylabel("bank RMSE [deg]", fontsize=7)
    ax[0].legend(fontsize=5.8, frameon=False, loc="lower right", handlelength=1.4,
                 borderpad=0.1, labelspacing=0.25)
    fig.tight_layout(pad=0.4)
    out = _figdir(results_dir) / "fig_robustness_ppo.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return out


FIGURES = {
    **{f"altitude_viz_{a}_{p}": (lambda a=a, p=p: (lambda ckpt_dir, results_dir: altitude_viz(a, p, ckpt_dir, results_dir)))()
       for a in ("ppo", "sac", "td3") for p in ("Airship_V7", "Volantex_Ranger")},
    "attitude_exec": lambda ckpt_dir, results_dir: attitude_exec(("Airship_V7", "Volantex_Ranger"), ckpt_dir, results_dir),
    "maneuver_facet_Airship_V7": lambda ckpt_dir, results_dir: maneuver_facet("Airship_V7", ckpt_dir, results_dir),
    "maneuver_render": lambda ckpt_dir, results_dir: maneuver_render(("Airship_V7", "Volantex_Ranger"), ckpt_dir, results_dir),
    "ge_energy": lambda ckpt_dir, results_dir: ge_energy(results_dir),
    "robustness_ppo": lambda ckpt_dir, results_dir: robustness_ppo(results_dir),
}


def render(names=None, ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR):
    (Path(results_dir) / "figures").mkdir(parents=True, exist_ok=True)
    return [FIGURES[n](ckpt_dir, results_dir) for n in (names or FIGURES)]
