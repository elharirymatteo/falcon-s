"""Pure-torch batched altitude-keeping env -- a no-warp port of WarpAltitudeEnv for cross-backend
validation. Same dynamics (poly aero + rate damping + thrust + gravity), same actuator model
(scale->2nd/1st-order RK45), same DP-RK45 rigid-body integrator (forces frozen per step), same 11-dim
obs + reward + termination + reference ramp as the warp stack. Goal: train PPO here and check it
converges to the same RMSE/robustness -> the warp result isn't a warp-kernel artifact.

Quaternion convention matches warp: [x, y, z, w]. Runs 4096 envs on the GPU in plain torch.
Validate one-step parity vs warp before trusting training (see __main__).
"""
import numpy as np
import pandas as pd
import torch

from falcons.aircraft.config import AircraftConfig
from falcons.aircraft.params import load_params, PropulsionParameters
from falcons.sim.torch.wind import TorchDryden

TRIM_THROTTLE = 0.45


# ---------------------------------------------------------------- helpers (torch, batched [N])
def isa_density(h):
    T0, p0, L, g, R = 288.15, 101325.0, 0.0065, 9.80665, 287.05287
    h = h.clamp(0.0, 11000.0)
    T = T0 - L * h
    p = p0 * (T / T0) ** (g / (R * L))
    return p / (R * T)


def safe_pow(base, exp):
    """base[N,1] ** exp[K] -> [N,K], correct for negative base + integer exp (avoids NaN)."""
    absp = base.abs() ** exp                                  # base.abs()>=0 safe; 0**0=1
    sign = torch.where((exp.long() % 2 == 0), torch.ones_like(exp), torch.sign(base))
    return sign * absp


def quat_normalize(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def quat_rotate(q, v):                                        # rotate v by q ([x,y,z,w])
    xyz = q[:, :3]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + q[:, 3:4] * t + torch.cross(xyz, t, dim=-1)


def quat_rotate_inv(q, v):
    qc = q.clone(); qc[:, :3] = -qc[:, :3]
    return quat_rotate(qc, v)


def quat_mul(a, b):                                           # [x,y,z,w]
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


def quat_rpy(roll, pitch, yaw):                               # euler -> [x,y,z,w] (warp wp.quat_rpy)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], dim=-1)


class AltitudeEnv:
    def __init__(self, num_envs, device="cuda", cfg=None):
        cfg = cfg or {}
        self.n = num_envs; self.device = device; dev = device
        self.horizon = cfg.get("horizon", 2000)
        self.dt = 0.01
        self.target = cfg.get("target_altitude", 50.0)
        self.spawn = cfg.get("spawning_distance", 1.0)
        w = cfg.get("reward_weights", [5.0, 2.0, 0.2, 0.1])
        self.w_alt, self.w_va, self.w_pr, self.w_sm = w
        self.sigma = cfg.get("altitude_sigma", 10.0)
        self.zone = cfg.get("target_altitude_zone", 3.0)
        self.va_band = cfg.get("va_band", 4.0); self.recovery = 8.0
        self.ov_w = cfg.get("overshoot_weight", 2.0); self.ov_thr = cfg.get("overshoot_threshold", 5.0)
        self.crash_pen = cfg.get("crash_penalty", 40.0); self.stall_pen = cfg.get("stall_penalty", 30.0)
        self.horizon_bonus = 10.0
        self.alpha_safety_w = cfg.get("alpha_safety_weight", 1.0); self.alpha_safety_start = cfg.get("alpha_safety_start", 0.7)
        self.va_safety_w = cfg.get("va_safety_weight", 1.0)
        self.g = 9.81
        self.energy_w = cfg.get("energy_weight", 3.0); self.energy_sigma = cfg.get("energy_sigma", 5.0)
        self.damp_w = cfg.get("damp_weight", 0.0); self.damp_zone = cfg.get("damp_zone", 10.0)
        self.pr_cap = cfg.get("pr_cap", 1.0)
        self.ref_rate = cfg.get("ref_rate", 0.0)
        self.i_clamp = 25.0
        self.mode = cfg.get("target_mode", "fixed"); self.mode_id = {"fixed": 0, "dynamic": 1}[self.mode]
        self.tgt_lo = cfg.get("target_lo", 30.0); self.tgt_hi = cfg.get("target_hi", 90.0)
        self.in_ground_effect = bool(cfg.get("in_ground_effect", True))
        self.turbulence = cfg.get("turbulence", None)          # None | "light" | "moderate" | ...
        self.constant_wind = cfg.get("constant_wind", [0.0, 0.0, 0.0])   # inertial (NED) m/s

        self.aircraft = cfg.get("aircraft", "Airship_V7")
        raw = AircraftConfig(self.aircraft).load()
        self.stall_deg = float(raw["aero_params"]["stall_angle_deg"])
        self.alpha_limit = float(np.radians(1.5 * self.stall_deg))
        self.base_vel = cfg.get("base_vel", float(raw["default_initial_state"]["linear_vel"][0]))
        self.target_va = cfg.get("target_airspeed", self.base_vel)
        self.va_min = cfg.get("va_min", 0.6 * self.base_vel); self.va_stall = self.va_min
        self.va_safety_start = cfg.get("va_safety_start", 0.75 * self.base_vel)
        self.va_scale = max(0.5 * self.base_vel, 5.0); self.vz_scale = self.va_scale

        # geometry / inertia / aero / propulsion (from the shared config)
        VP = raw["vehicle_params"]; wing = VP["wing"]
        self.S = float(wing["area"]); self.b = float(wing["span"]); self.c = float(wing["mac"])
        self.taper = float(wing["taper_ratio"]); self.AR = float(wing["aspect_ratio"])
        self.cg_z = float(wing["cg_offset_vector"][2])
        self.mass = float(VP["mass"])
        self.J = torch.tensor(VP["inertia_matrix"], dtype=torch.float32, device=dev)
        self.Jinv = torch.inverse(self.J)
        self.Clp = float(raw["aero_params"].get("Clp", 0.0))
        self.Cmq = float(raw["aero_params"].get("Cmq", 0.0))
        self.Cnr = float(raw["aero_params"].get("Cnr", 0.0))
        # actuator params
        surf = VP["actuator_system"]["aero_surfaces"]["elevator"]
        self.omega0 = float(surf["omega_0"]); self.zeta = float(surf["zeta"])
        mot = list(VP["actuator_system"]["motors"].values())[0]
        self.T_s = float(mot["T"])
        co = load_params(self.aircraft)
        self.elev_min, self.elev_max = [float(v) for v in co["control_limits"].elevator_limits]
        self.thr_min, self.thr_max = [float(v) for v in co["control_limits"].throttle_limits]
        # propulsion (single-motor split handled by the fixed loader)
        pp = PropulsionParameters.from_config(raw)
        self.td1 = torch.tensor(pp.td1, dtype=torch.float32, device=dev)
        self.td2 = torch.tensor(pp.td2, dtype=torch.float32, device=dev)
        tvi = torch.tensor(pp.tvi, dtype=torch.float32, device=dev)
        self.tvi = tvi / tvi.norm()
        self.km, self.kq, self.ko, self.Cp, self.Sp = pp.k_m, pp.k_q, pp.k_o, pp.C_p, pp.Sp

        # poly aero terms (CSV: alpha,beta,delta_e,delta_a,delta_r | CD,CY,CL,CMx,CMy,CMz)
        poly = pd.read_csv(raw["aero_params"]["poly_params_file"]).values.astype(np.float32)
        self.exps = torch.tensor(poly[:, :5], device=dev)               # [30,5]
        self.pcoef = torch.tensor(poly[:, 5:11], device=dev)            # [30,6] CD,CY,CL,Cl,Cm,Cn

        self.num_obs = 11; self.num_act = 2
        self._alloc()
        # wind: constant (inertial, rotated to body each step) + optional Dryden gusts
        self.const_wind_ned = torch.tensor(self.constant_wind, dtype=torch.float32, device=dev)
        self.wind_enabled = bool(self.turbulence) or bool(self.const_wind_ned.abs().sum() > 0)
        self.dryden = TorchDryden(self.n, self.dt, self.b, self.turbulence, dev,
                                  seed=cfg.get('seed')) if self.turbulence else None

    def _alloc(self):
        n, dev = self.n, self.device
        z = lambda *s: torch.zeros(*s, device=dev)
        self.pos = z(n, 3); self.vel = z(n, 3); self.omega = z(n, 3)
        self.quat = z(n, 4); self.quat[:, 3] = 1.0
        self.elev = z(n); self.ail = z(n); self.rud = z(n); self.thr = z(n)
        self.elev_d = z(n); self.ail_d = z(n); self.rud_d = z(n)
        self.tgt = z(n); self.true_tgt = z(n); self.ierr = z(n); self.refrate = z(n)
        self.prev = z(n, 2); self.step_count = torch.zeros(n, dtype=torch.long, device=dev)
        self.gust_lin = z(n, 3); self.Va_prev = torch.full((n,), 1.0, device=dev)

    # ---------------- actuator RK45 (Dormand-Prince, constant action over step)
    def _act2_rk45(self, action, x, xd):
        w2 = self.omega0 * self.omega0; c = 2.0 * self.zeta * self.omega0; dt = self.dt
        def f(x_, xd_): return xd_, (-c * xd_ - w2 * x_ + w2 * action)
        a = [1/5, 3/40, 9/40, 44/45, -56/15, 32/9, 19372/6561, -25360/2187, 64448/6561, -212/729,
             9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
        b1, b3, b4, b5, b6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
        dx1, dxd1 = f(x, xd); k1x, k1d = dt*dx1, dt*dxd1
        dx2, dxd2 = f(x+a[0]*k1x, xd+a[0]*k1d); k2x, k2d = dt*dx2, dt*dxd2
        dx3, dxd3 = f(x+a[1]*k1x+a[2]*k2x, xd+a[1]*k1d+a[2]*k2d); k3x, k3d = dt*dx3, dt*dxd3
        dx4, dxd4 = f(x+a[3]*k1x+a[4]*k2x+a[5]*k3x, xd+a[3]*k1d+a[4]*k2d+a[5]*k3d); k4x, k4d = dt*dx4, dt*dxd4
        dx5, dxd5 = f(x+a[6]*k1x+a[7]*k2x+a[8]*k3x+a[9]*k4x, xd+a[6]*k1d+a[7]*k2d+a[8]*k3d+a[9]*k4d); k5x, k5d = dt*dx5, dt*dxd5
        dx6, dxd6 = f(x+a[10]*k1x+a[11]*k2x+a[12]*k3x+a[13]*k4x+a[14]*k5x,
                      xd+a[10]*k1d+a[11]*k2d+a[12]*k3d+a[13]*k4d+a[14]*k5d); k6x, k6d = dt*dx6, dt*dxd6
        xn = x + b1*k1x + b3*k3x + b4*k4x + b5*k5x + b6*k6x
        xdn = xd + b1*k1d + b3*k3d + b4*k4d + b5*k5d + b6*k6d
        return xn, xdn

    def _act1_rk45(self, action, x):
        invT = 1.0 / self.T_s; dt = self.dt
        f = lambda xv: invT * (action - xv)
        a = [1/5, 3/40, 9/40, 44/45, -56/15, 32/9, 19372/6561, -25360/2187, 64448/6561, -212/729,
             9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
        b1, b3, b4, b5, b6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
        k1 = dt*f(x); k2 = dt*f(x+a[0]*k1); k3 = dt*f(x+a[1]*k1+a[2]*k2)
        k4 = dt*f(x+a[3]*k1+a[4]*k2+a[5]*k3); k5 = dt*f(x+a[6]*k1+a[7]*k2+a[8]*k3+a[9]*k4)
        k6 = dt*f(x+a[10]*k1+a[11]*k2+a[12]*k3+a[13]*k4+a[14]*k5)
        return x + b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6

    def _scale(self, a, lo, hi):
        return (a.clamp(-1, 1) + 1.0) * 0.5 * (hi - lo) + lo

    # ---------------- forces/moments (frozen over the integration step)
    def _forces(self):
        asv = self.vel - self.gust_lin                          # body-frame wind + gusts
        Va = asv.norm(dim=-1).clamp_min(1e-6)
        alpha = torch.atan2(asv[:, 2], asv[:, 0])
        beta = torch.asin((asv[:, 1] / (Va + 1e-6)).clamp(-1, 1))
        rho = isa_density(-self.pos[:, 2])
        Q = 0.5 * rho * Va * Va
        # poly coeffs (alpha,beta in deg; deflections in deg)
        bases = torch.stack([alpha * 180.0 / np.pi, beta * 180.0 / np.pi, self.elev, self.ail, self.rud], dim=-1)
        term = torch.ones(self.n, self.exps.shape[0], device=self.device)
        for ci in range(5):
            term = term * safe_pow(bases[:, ci:ci+1], self.exps[:, ci])
        coeffs = term @ self.pcoef                              # [N,6] CD,CY,CL,Cl,Cm,Cn
        CD, CY, CL, Cl, Cm, Cn = coeffs.unbind(dim=-1)
        # ground effect
        D = Q * self.S * CD; Y = Q * self.S * CY; L = Q * self.S * CL
        if self.in_ground_effect:
            h = (-self.pos[:, 2] + self.cg_z).abs() / self.b
            mu_l = 1.0 + (1.0 - 2.25 * (self.taper**0.00273 - 0.997) * (self.AR**0.717 + 13.6)) * \
                (288.0 * h.clamp_min(1e-6)**0.787 * torch.exp(-9.14 * h.clamp_min(1e-6)**0.327)) / (self.AR**0.882)
            t1 = 1.0 - (1.0 - 0.157 * max(self.taper**0.775 - 0.373, 0.0) * max(self.AR**0.417 - 1.27, 0.0)) * \
                torch.exp(-4.74 * h.clamp_min(1e-6)**0.814)
            t2 = h*h * torch.exp(-3.88 * h.clamp_min(1e-6)**0.758)
            mu_d = t1 - t2
            D = D * mu_d * mu_l**2; L = L * mu_l
        Fw = torch.stack([-D, Y, -L], dim=-1)
        q_aero = quat_rpy(torch.zeros_like(alpha), alpha, -beta)   # -beta: see the CPU plant
        Fb_aero = quat_rotate_inv(q_aero, Fw)
        grav = torch.zeros_like(self.vel); grav[:, 2] = self.g
        Fb_g = self.mass * quat_rotate_inv(self.quat, grav)
        # thrust (both engines same throttle)
        Vd = Va + self.thr * (self.km - Va)
        Tp = 0.5 * rho * self.Sp * self.Cp * Vd * (Vd - Va)
        F1 = Tp[:, None] * self.tvi[None, :]
        Fb_thrust = 2.0 * F1
        Mprop = -self.kq * (self.ko * self.thr)**2
        Mb_thrust = torch.cross(self.td1[None, :].expand_as(F1), F1, dim=-1) + \
                    torch.cross(self.td2[None, :].expand_as(F1), F1, dim=-1)            # +Mprop -Mprop cancel
        Mb_aero = torch.stack([Cl * Q * self.S * self.b, Cm * Q * self.S * self.c, Cn * Q * self.S * self.b], dim=-1)
        Vc = Va.clamp_min(1.0)
        Mb_aero[:, 0] += self.Clp * (self.omega[:, 0] * self.b / (2*Vc)) * Q * self.S * self.b
        Mb_aero[:, 1] += self.Cmq * (self.omega[:, 1] * self.c / (2*Vc)) * Q * self.S * self.c
        Mb_aero[:, 2] += self.Cnr * (self.omega[:, 2] * self.b / (2*Vc)) * Q * self.S * self.b
        Fb = Fb_aero + Fb_g + Fb_thrust
        Mb = Mb_aero + Mb_thrust
        self.Va_prev = Va
        return Fb, Mb, Va, alpha

    def _deriv(self, p, v, w, q, Fb, Mb):
        q = quat_normalize(q)
        dv = Fb / self.mass - torch.cross(w, v, dim=-1)
        Jw = w @ self.J.T
        dw = (Mb - torch.cross(w, Jw, dim=-1)) @ self.Jinv.T
        dp = quat_rotate(q, v)
        wq = torch.cat([w, torch.zeros(self.n, 1, device=self.device)], dim=-1)
        dq = 0.5 * quat_mul(q, wq)   # body-rate form: q_dot = 1/2 q (x) (omega, 0)
        return dp, dv, dw, dq

    def _rk45_rigid(self, Fb, Mb):
        dt = self.dt; p0, v0, w0 = self.pos, self.vel, self.omega; q0 = quat_normalize(self.quat)
        a21 = 1/5; a31, a32 = 3/40, 9/40; a41, a42, a43 = 44/45, -56/15, 32/9
        a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
        a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
        b1, b3, b4, b5, b6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
        dp1, dv1, dw1, dq1 = self._deriv(p0, v0, w0, q0, Fb, Mb)
        k1 = (dt*dp1, dt*dv1, dt*dw1, dt*dq1)
        dp2, dv2, dw2, dq2 = self._deriv(p0, v0+a21*k1[1], w0+a21*k1[2], q0+a21*k1[3], Fb, Mb)
        k2 = (dt*dp2, dt*dv2, dt*dw2, dt*dq2)
        dp3, dv3, dw3, dq3 = self._deriv(p0, v0+a31*k1[1]+a32*k2[1], w0+a31*k1[2]+a32*k2[2], q0+a31*k1[3]+a32*k2[3], Fb, Mb)
        k3 = (dt*dp3, dt*dv3, dt*dw3, dt*dq3)
        dp4, dv4, dw4, dq4 = self._deriv(p0, v0+a41*k1[1]+a42*k2[1]+a43*k3[1], w0+a41*k1[2]+a42*k2[2]+a43*k3[2], q0+a41*k1[3]+a42*k2[3]+a43*k3[3], Fb, Mb)
        k4 = (dt*dp4, dt*dv4, dt*dw4, dt*dq4)
        dp5, dv5, dw5, dq5 = self._deriv(p0, v0+a51*k1[1]+a52*k2[1]+a53*k3[1]+a54*k4[1], w0+a51*k1[2]+a52*k2[2]+a53*k3[2]+a54*k4[2], q0+a51*k1[3]+a52*k2[3]+a53*k3[3]+a54*k4[3], Fb, Mb)
        k5 = (dt*dp5, dt*dv5, dt*dw5, dt*dq5)
        dp6, dv6, dw6, dq6 = self._deriv(p0, v0+a61*k1[1]+a62*k2[1]+a63*k3[1]+a64*k4[1]+a65*k5[1], w0+a61*k1[2]+a62*k2[2]+a63*k3[2]+a64*k4[2]+a65*k5[2], q0+a61*k1[3]+a62*k2[3]+a63*k3[3]+a64*k4[3]+a65*k5[3], Fb, Mb)
        k6 = (dt*dp6, dt*dv6, dt*dw6, dt*dq6)
        self.pos = p0 + b1*k1[0]+b3*k3[0]+b4*k4[0]+b5*k5[0]+b6*k6[0]
        self.vel = v0 + b1*k1[1]+b3*k3[1]+b4*k4[1]+b5*k5[1]+b6*k6[1]
        self.omega = w0 + b1*k1[2]+b3*k3[2]+b4*k4[2]+b5*k5[2]+b6*k6[2]
        self.quat = quat_normalize(q0 + b1*k1[3]+b3*k3[3]+b4*k4[3]+b5*k5[3]+b6*k6[3])

    # ---------------- obs / reward / termination
    def _obs(self, Va, alpha):
        q = self.quat
        pitch = torch.asin((2.0 * (q[:, 3]*q[:, 1] - q[:, 2]*q[:, 0])).clamp(-1, 1))
        en = (self.elev - self.elev_min) / (self.elev_max - self.elev_min) * 2 - 1
        tn = (self.thr - self.thr_min) / (self.thr_max - self.thr_min) * 2 - 1
        c = lambda x: x.clamp(-3, 3)
        o = torch.stack([
            c((-self.pos[:, 2] - self.tgt) / 25.0), c(self.omega[:, 1] / 2.0),
            c((Va - self.target_va) / self.va_scale), c(pitch / (np.pi/4)),
            c(self.vel[:, 2] / self.vz_scale), c(alpha / np.radians(20)),
            c(en / 1.0), c(self.elev_d / 100.0), c(tn / 1.0),
            c(self.ierr / 10.0), c(self.refrate / 5.0),
        ], dim=-1)
        return o

    def _reward(self, action, Va, alpha):
        h_err = -self.pos[:, 2] - self.tgt
        ae = Va - self.target_va; pre = self.omega[:, 1]; ah = h_err.abs()
        alt_r = torch.exp(-0.5 * (h_err / self.sigma)**2)
        air_r = -(ae.abs() / max(self.target_va, 1.0)).clamp(max=1.0)
        pr_pen = -(pre*pre).clamp(max=self.pr_cap)
        d = (action - self.prev).abs(); raw_sm = -(d[:, 0] + d[:, 1]) * 0.5
        rec = (1.0 - ah / self.recovery).clamp_min(0.0); sm_pen = raw_sm * rec
        alt_tap = (1.0 - ah / self.zone).clamp_min(0.0); sp_tap = (1.0 - ae.abs()/self.va_band).clamp_min(0.0)
        stab = self.w_alt * 2.0 * alt_tap * sp_tap
        ov_pen = torch.where(h_err > self.ov_thr, -(((h_err - self.ov_thr)/self.ov_thr)**2).clamp(max=1.0), torch.zeros_like(h_err))
        a_pen = torch.zeros_like(h_err)
        if self.alpha_safety_w > 0:
            ad = alpha.abs()*180/np.pi; onset = self.alpha_safety_start*self.stall_deg
            span = max(self.stall_deg - onset, 1e-3); ov = ((ad - onset)/span).clamp_min(0.0)
            a_pen = -(ov*ov).clamp(max=1.0)
        v_pen = torch.zeros_like(h_err)
        if self.va_safety_w > 0:
            span = max(self.va_safety_start - self.va_stall, 1e-3); ov = ((self.va_safety_start - Va)/span).clamp_min(0.0)
            v_pen = -(ov*ov).clamp(max=1.0)
        e_r = torch.zeros_like(h_err)
        if self.energy_w > 0:
            ke = (Va*Va - self.target_va**2)/(2*self.g); eE = h_err + ke
            e_r = torch.exp(-0.5 * (eE/self.energy_sigma)**2)
        d_pen = torch.zeros_like(h_err)
        if self.damp_w > 0:
            prox = (1.0 - ah/self.damp_zone).clamp(0, 1)
            d_pen = -(self.vel[:, 2]**2).clamp(max=25.0) * prox
        return (self.w_alt*alt_r + self.w_va*air_r + self.w_pr*pr_pen + self.w_sm*sm_pen + stab
                + self.ov_w*ov_pen + self.alpha_safety_w*a_pen + self.va_safety_w*v_pen
                + self.energy_w*e_r + self.damp_w*d_pen)

    @torch.no_grad()
    def reset(self, mask=None):
        n, dev = self.n, self.device
        m = torch.ones(n, dtype=torch.bool, device=dev) if mask is None else mask
        k = int(m.sum())
        if k == 0:
            return
        if self.mode_id == 1:
            tgt = self.tgt_lo + (self.tgt_hi - self.tgt_lo) * torch.rand(k, device=dev)
        else:
            tgt = torch.full((k,), self.target, device=dev)
        alt = tgt + self.spawn * (2.0 * torch.rand(k, device=dev) - 1.0)
        self.pos[m] = torch.stack([torch.zeros(k, device=dev), torch.zeros(k, device=dev), -alt], dim=-1)
        self.true_tgt[m] = tgt
        sub = alt if self.ref_rate > 0 else tgt
        self.vel[m] = torch.tensor([self.base_vel, 0.0, 0.0], device=dev)
        self.omega[m] = 0.0
        self.quat[m] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev)
        self.elev[m] = 0; self.ail[m] = 0; self.rud[m] = 0; self.thr[m] = TRIM_THROTTLE
        self.elev_d[m] = 0; self.ail_d[m] = 0; self.rud_d[m] = 0
        self.step_count[m] = 0; self.prev[m] = 0; self.tgt[m] = sub; self.ierr[m] = 0
        self.gust_lin[m] = 0; self.Va_prev[m] = self.base_vel
        if self.dryden is not None:
            self.dryden.reset(m)
        if mask is None:
            Fb, Mb, Va, alpha = self._forces()
            return self._obs(Va, alpha)

    def _update_wind(self):
        """Constant wind (inertial -> body) + Dryden gusts, evaluated as in the warp step order:
        before the physics update, on the previous step's airspeed and the current pose."""
        if not self.wind_enabled:
            return
        gust = quat_rotate_inv(self.quat, self.const_wind_ned.expand(self.n, 3))
        if self.dryden is not None:
            lin, _ang = self.dryden.update(self.Va_prev, -self.pos[:, 2])   # angular gusts unused
            gust = gust + lin
        self.gust_lin = gust

    @torch.no_grad()
    def step(self, action):
        action = action.clamp(-1, 1)
        self._update_wind()
        # actuators (cmd: elevator=a0, throttle=a1, ail=rud=0)
        se = self._scale(action[:, 0], self.elev_min, self.elev_max)
        self.elev, self.elev_d = self._act2_rk45(se, self.elev, self.elev_d)
        self.elev = self.elev.clamp(self.elev_min, self.elev_max)
        zero = torch.zeros(self.n, device=self.device)
        sa = self._scale(zero, *([self.elev_min, self.elev_max]))  # aileron limits ~ use surfaces; ail cmd 0
        self.ail, self.ail_d = self._act2_rk45(self._scale(zero, -15.0, 15.0), self.ail, self.ail_d)
        self.ail = self.ail.clamp(-15.0, 15.0)
        self.rud, self.rud_d = self._act2_rk45(self._scale(zero, -15.0, 15.0), self.rud, self.rud_d)
        self.rud = self.rud.clamp(-15.0, 15.0)
        st = self._scale(action[:, 1], self.thr_min, self.thr_max)
        self.thr = self._act1_rk45(st, self.thr).clamp(self.thr_min, self.thr_max)

        Fb, Mb, Va, alpha = self._forces()
        self._rk45_rigid(Fb, Mb)

        # reference ramp + integral
        if self.ref_rate > 0:
            diff = self.true_tgt - self.tgt
            step = diff.clamp(-self.ref_rate*self.dt, self.ref_rate*self.dt)
            self.tgt = self.tgt + step; self.refrate = step / self.dt
        self.ierr = (self.ierr + (-self.pos[:, 2] - self.tgt) * self.dt).clamp(-self.i_clamp, self.i_clamp)

        Va2 = self.vel.norm(dim=-1).clamp_min(1e-6)
        alpha2 = torch.atan2(self.vel[:, 2], self.vel[:, 0])
        rew = self._reward(action, Va2, alpha2)

        # termination: reason 1 crash, 2 stall (crash priority)
        reason = torch.zeros(self.n, dtype=torch.long, device=self.device)
        crash = (-self.pos[:, 2] + self.cg_z) < 0.0
        stall = (alpha2.abs() > self.alpha_limit) | (Va2 < self.va_min)
        nan = ~torch.isfinite(self.pos).all(dim=-1)
        reason[stall] = 2; reason[crash | nan] = 1
        terminated = reason > 0
        self.step_count += 1
        truncated = self.step_count >= self.horizon
        done = terminated | truncated
        # terminal reward
        rew = rew + torch.where(reason == 1, -self.crash_pen, torch.zeros_like(rew))
        rew = rew + torch.where(reason == 2, -self.stall_pen, torch.zeros_like(rew))
        hb = truncated & (~terminated) & ((-self.pos[:, 2] - self.tgt).abs() < self.zone)
        rew = rew + torch.where(hb, self.horizon_bonus, torch.zeros_like(rew))

        self.prev = action.clone()
        term_obs = self._obs(Va2, alpha2)                      # pre-reset obs (truncation bootstrap)
        info = {"reason": reason, "truncated": truncated, "terminal_obs": term_obs}
        if done.any():
            self.reset(done)                                   # also zeroes prev[done]
            obs = self._obs(*self._post_reset_va_alpha())
        else:
            obs = term_obs
        return obs, rew, done, info

    def _post_reset_va_alpha(self):
        Va = self.vel.norm(dim=-1).clamp_min(1e-6)
        alpha = torch.atan2(self.vel[:, 2], self.vel[:, 0])
        return Va, alpha
