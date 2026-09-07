"""Named maneuver streams (attitude-space) for the executor comparison: circle, figure8, helix
and sturn as pure functions, so any controller can be driven by the identical
(phi*, hdot*, Va*) reference.

Sized to each aircraft's own envelope (phi_max_full, hdot_max_full, base_vel).
"""
import numpy as np

DT = 0.01


def maneuver_stream(name, phi_max_full, hdot_max_full, base_vel):
    A = 0.8 * phi_max_full
    C = hdot_max_full
    V = base_vel
    Tturn = 2.0 * np.pi * V / (9.81 * np.tan(A))
    if name == "circle":
        T = 1.3 * Tturn
        t = np.arange(int(T / DT)) * DT
        phi = A * np.tanh(t / 2.0)
        hdot = np.zeros_like(t)
    elif name == "figure8":
        T = 2.0 * Tturn
        t = np.arange(int(T / DT)) * DT
        phi = A * np.tanh(3.0 * np.sin(2.0 * np.pi * t / T))
        hdot = np.zeros_like(t)
    elif name == "helix":
        bank = 0.6 * A
        Tg = 2.0 * np.pi * V / (9.81 * np.tan(bank))
        T = 1.6 * Tg
        t = np.arange(int(T / DT)) * DT
        phi = bank * np.tanh(t / 2.0)
        climb_dur = min(18.0, 0.4 * T)
        r_up = np.clip((t - 2.0) / 3.0, 0.0, 1.0)
        r_dn = np.clip((climb_dur + 3.0 - t) / 3.0, 0.0, 1.0)
        hdot = min(1.0, C) * np.minimum(r_up, r_dn)
    elif name == "sturn":
        T = 3.2 * Tturn
        t = np.arange(int(T / DT)) * DT
        phi = 0.7 * A * np.sin(2.0 * np.pi * t / (Tturn * 0.8))
        hdot = np.zeros_like(t)
    else:
        raise ValueError(f"unknown maneuver '{name}' (circle|figure8|helix|sturn)")
    va = np.full_like(t, V)
    return (phi.astype(np.float32), hdot.astype(np.float32), va.astype(np.float32))


MANEUVERS = ["circle", "figure8", "helix", "sturn"]
