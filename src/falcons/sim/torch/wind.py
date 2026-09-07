"""Batched Dryden turbulence + constant wind for the pure-torch backend.

Port of falcons/sim/warp/dryden.py / wind.py: same MIL-F-8785C shaping filters, same explicit
Euler step, same output convention (gusts are body-frame velocities subtracted from the aircraft
velocity to form the airspeed vector). Filter states live per environment, so a batch of N envs
runs the model independently.

The scalar CPU model (falcons/sim/cpu/physics/dryden.py) is the parity reference; the noise argument
of update() exists so a test can drive both with the identical white-noise sequence.
"""
import torch

KNOT = 1.852 / 3.6
W20_KNOTS = {"very light": 2, "light": 15, "moderate": 30, "severe": 45}
M_TO_FT = 1.0 / 0.3048
FT_TO_M = 0.3048


class TorchDryden:
    """Dryden turbulence (MIL-F-8785C) for N parallel environments."""

    def __init__(self, num_envs, dt, b, intensity="light", device="cpu", seed=None):
        self.n = num_envs
        self.dt = dt
        self.b = b
        self.device = device
        self.W_20_ms = W20_KNOTS[intensity] * KNOT
        self.gen = torch.Generator(device=device)
        if seed is not None:
            self.gen.manual_seed(int(seed))
        z = lambda *s: torch.zeros(*s, device=device)
        self.xu = z(num_envs); self.xv = z(num_envs, 2); self.xw = z(num_envs, 2)
        self.xp = z(num_envs); self.xq = z(num_envs, 3); self.xr = z(num_envs, 3)

    def reset(self, mask=None):
        """Zero the filter states (all envs, or the rows selected by a bool mask)."""
        if mask is None:
            for s in (self.xu, self.xv, self.xw, self.xp, self.xq, self.xr):
                s.zero_()
        else:
            self.xu[mask] = 0; self.xv[mask] = 0; self.xw[mask] = 0
            self.xp[mask] = 0; self.xq[mask] = 0; self.xr[mask] = 0

    # --- LTI Euler steps: output y uses the PRE-update state, matching warp/numpy
    @staticmethod
    def _step1(x, u, dt, a, c):
        y = c * x
        return x + dt * (a * x + u), y

    @staticmethod
    def _step2(x, u, dt, a11, a12, c1, c2):
        x1, x2 = x[:, 0], x[:, 1]
        y = c1 * x1 + c2 * x2
        x1n = x1 + dt * (a11 * x1 + a12 * x2 + u)
        x2n = x2 + dt * x1
        return torch.stack([x1n, x2n], dim=-1), y

    @staticmethod
    def _step3(x, u, dt, a11, a12, a13, c1, c2):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        y = c1 * x1 + c2 * x2
        x1n = x1 + dt * (a11 * x1 + a12 * x2 + a13 * x3 + u)
        x2n = x2 + dt * x1
        x3n = x3 + dt * x2
        return torch.stack([x1n, x2n, x3n], dim=-1), y

    def update(self, Va, h, noise=None):
        """Advance one step.

        Va: [N] airspeed (m/s), h: [N] altitude (m, positive up).
        noise: optional [N,6] standard normals (u,v,w,p,q,r); drawn internally if omitted.
        Returns (linear_gusts [N,3], angular_gusts [N,3]) in the body frame.
        """
        dt, b = self.dt, self.b
        h_ft = h * M_TO_FT
        sigma_w = 0.1 * self.W_20_ms
        sigma_u = sigma_w / (0.177 + 0.000823 * h_ft) ** 0.4
        sigma_v = sigma_u

        high = h_ft >= 1000.0
        L_low = (h_ft / (0.177 + 0.000823 * h_ft) ** 1.2) * FT_TO_M
        L_u = torch.where(high, h_ft * FT_TO_M, L_low)
        L_v = L_u
        L_w = torch.where(high, h_ft * FT_TO_M, h)

        Va = Va.clamp_min(0.1)
        if noise is None:
            noise = torch.randn(self.n, 6, device=self.device, generator=self.gen)
        w = noise / dt ** 0.5
        pi = torch.pi
        sqrt3 = 3.0 ** 0.5

        # longitudinal u_w (1st order)
        K_u = torch.sqrt(2.0 * L_u / (pi * Va)); T_u = L_u / Va
        self.xu, u_w = self._step1(self.xu, w[:, 0], dt, -1.0 / T_u, sigma_u * K_u / T_u)

        # lateral v_w (2nd order)
        K_v = torch.sqrt(L_v / (pi * Va)); T_v = L_v / Va
        self.xv, v_w = self._step2(self.xv, w[:, 1], dt, -2.0 / T_v, -1.0 / T_v**2,
                                   sigma_v * K_v * sqrt3 * T_v / T_v**2, sigma_v * K_v / T_v**2)

        # vertical w_w (2nd order)
        K_w = torch.sqrt(L_w / (pi * Va)); T_w = L_w / Va
        self.xw, w_w = self._step2(self.xw, w[:, 2], dt, -2.0 / T_w, -1.0 / T_w**2,
                                   sigma_w * K_w * sqrt3 * T_w / T_w**2, sigma_w * K_w / T_w**2)

        # roll rate p_w (1st order)
        K_p = torch.sqrt(0.8 * (pi * L_w / (4.0 * b)) ** (1.0 / 3.0) / (L_w * Va))
        T_p = 4.0 * b / (pi * Va)
        self.xp, p_w = self._step1(self.xp, w[:, 3], dt, -1.0 / T_p, sigma_w * K_p / T_p)

        # pitch rate q_w (3rd order, driven by the vertical gust filter)
        T_q = 4.0 * b / (pi * Va)
        a_q = -sigma_w * K_w * sqrt3 * T_w / Va; b_q = -sigma_w * K_w / Va
        c_q = T_q * T_w**2; d_q = 2.0 * T_q * T_w + T_w**2; e_q = 2.0 * T_w + T_q
        self.xq, q_w = self._step3(self.xq, w[:, 4], dt, -d_q / c_q, -e_q / c_q, -1.0 / c_q,
                                   a_q / c_q, b_q / c_q)

        # yaw rate r_w (3rd order, driven by the lateral gust filter)
        T_r = 3.0 * b / (pi * Va)
        a_r = sigma_v * K_v * sqrt3 * T_v / Va; b_r = sigma_v * K_v / Va
        c_r = T_r * T_v**2; d_r = 2.0 * T_r * T_v + T_v**2; e_r = 2.0 * T_v + T_r
        self.xr, r_w = self._step3(self.xr, w[:, 5], dt, -d_r / c_r, -e_r / c_r, -1.0 / c_r,
                                   a_r / c_r, b_r / c_r)

        return torch.stack([u_w, v_w, w_w], dim=-1), torch.stack([p_w, q_w, r_w], dim=-1)
