"""The aerodynamic model: an OpenVSP linear derivative set about one measured flight condition,
with ground effect read from a measured height sweep.

This is the reference implementation. The warp, torch and MPPI backends mirror it term for term,
and `tests/test_aero_parity.py` holds them to it.

WHAT THIS MODEL IS. VSPAERO was run at one operating point -- for the Volantex Ranger, alpha =
3.91 deg and elevator = -4.44 deg at 10.875 m/s -- and reports the coefficients there plus their
derivatives. The model is that linearisation, evaluated at the deltas from the run point:

    C = C_run + dC/dalpha * (alpha - alpha_run) + dC/dde * (de - de_run) + (rate and lateral terms)

so there is NO stall, no Reynolds or Mach dependence, and no validity beyond a few degrees of the
run point. The hard alpha_max termination is what keeps a rollout inside the fitted envelope; it
is not a stall model, and the model will happily keep generating lift past it.

WHAT IS DELIBERATELY MISSING. OpenVSP reports about 60 derivatives; this uses 27. The rest
(CMm_Beta, CL_p, CS_Alpha, CMl_q, CD_Beta, ...) are numerical residue in the VLM solution that
should be identically zero by symmetry, so they are dropped rather than fed into the dynamics.
Do not "complete" the set -- see `falcons.aircraft.params.CHANNELS`.

Only C_D, C_L and C_m carry a trim term (`*_Total`). C_S, C_l and C_n do not, because they are
zero at the symmetric run point to within solver noise (CS_Total = 2.3e-4, CMl_Total = 2.1e-5,
CMn_Total = -1.1e-4).
"""
from typing import Dict

import numpy as np

# From `derivatives`, not `params`: params.py defines the warp structs and imports warp at
# module scope, and the CPU reference plant must not need a GPU stack to run.
from falcons.aircraft.derivatives import IDX, DerivativeAeroParameters
from falcons.sim.aero_contract import AeroCoefs, AeroInputs
from .atmosphere import AtmosphereStateMixin

# Airspeed floor for the rate non-dimensionalisation. p*b/(2V) diverges as V -> 0, and a plant
# sitting still on the ground would otherwise see unbounded damping.
V_FLOOR = 0.1


class DerivativeAerodynamics:
    """The coefficient model for one airframe.

    Holds the loaded derivative set and the span/MAC it non-dimensionalises rates by. Stateless
    per step: `coefficients` is a pure function of `AeroInputs`.
    """

    def __init__(self, params: DerivativeAeroParameters, span: float, mac: float,
                 ge_enable: bool = True):
        self.p = params
        self.span = float(span)
        self.mac = float(mac)
        self.ge_enable = bool(ge_enable)

    # ─── ground effect

    def ground_effect(self, h: float):
        """Coefficient set and induced-drag factor at height `h` [m], positive up.

        Ground effect is stored as an increment anchored at the top of the measured sweep, so
        `free + increment` is exactly the free-air set there and above. Clamping h/c to the swept
        range is therefore correct at BOTH ends: no separate "out of ground effect" branch, and
        no step at the hand-off.

        Below the sweep the increment is HELD, not extrapolated. That floor is real -- the sweep
        starts at h/c = 2.07 (h/b = 0.20) -- so deeper ground effect is not modelled.
        """
        if not self.ge_enable:
            return self.p.free, self.p.k_ind_free
        hc = np.clip(h / self.mac, self.p.hc[0], self.p.hc[-1])
        # One interpolation over the stacked table; np.interp is per-column, so this is the
        # vectorised form of the 27 scalar interp1 calls the model was specified as.
        increment = np.array([np.interp(hc, self.p.hc, col) for col in self.p.ge_increment.T])
        k_ind = float(np.interp(hc, self.p.hc, self.p.k_ind))
        return self.p.free + increment, k_ind

    # ─── the model

    def coefficients(self, u: AeroInputs) -> AeroCoefs:
        v_c = max(u.v, V_FLOOR)
        p_hat = u.p * self.span / (2.0 * v_c)
        q_hat = u.q * self.mac / (2.0 * v_c)
        r_hat = u.r * self.span / (2.0 * v_c)

        # deltas from the operating point the tables were MEASURED at
        da = u.alpha - self.p.alpha_run
        de = u.elevator - self.p.de_run

        c, k_ind = self.ground_effect(u.h)

        C_D = (c[IDX["CD_Total"]] + c[IDX["CD_Alpha"]] * da
               + k_ind * (c[IDX["CL_Alpha"]] * da) ** 2
               + c[IDX["CD_elevator"]] * de + c[IDX["CD_q"]] * q_hat)

        C_S = (c[IDX["CS_Beta"]] * u.beta + c[IDX["CS_aileron"]] * u.aileron
               + c[IDX["CS_rudder"]] * u.rudder
               + c[IDX["CS_p"]] * p_hat + c[IDX["CS_r"]] * r_hat)

        C_L = (c[IDX["CL_Total"]] + c[IDX["CL_Alpha"]] * da
               + c[IDX["CL_elevator"]] * de + c[IDX["CL_q"]] * q_hat)

        C_l = (c[IDX["CMl_Beta"]] * u.beta + c[IDX["CMl_aileron"]] * u.aileron
               + c[IDX["CMl_rudder"]] * u.rudder
               + c[IDX["CMl_p"]] * p_hat + c[IDX["CMl_r"]] * r_hat)

        C_m = (c[IDX["CMm_Total"]] + c[IDX["CMm_Alpha"]] * da
               + c[IDX["CMm_elevator"]] * de + c[IDX["CMm_q"]] * q_hat)

        C_n = (c[IDX["CMn_Beta"]] * u.beta + c[IDX["CMn_aileron"]] * u.aileron
               + c[IDX["CMn_rudder"]] * u.rudder
               + c[IDX["CMn_p"]] * p_hat + c[IDX["CMn_r"]] * r_hat)

        return AeroCoefs(CD=C_D, CS=C_S, CL=C_L, Cl=C_l, Cm=C_m, Cn=C_n)

    def coefficients_with_diagnostics(self, u: AeroInputs) -> Dict[str, float]:
        """`coefficients`, plus what the in-ground-effect run cost relative to free air.

        The old empirical model reported multiplicative factors `mu_l`/`mu_d`; those are gone with
        it. Ground effect here is not a factor on a free-air coefficient, so the diagnostic is the
        pair of values and their ratio, which is what the history and the JSBSim comparison
        actually plot.
        """
        c = self.coefficients(u)
        if not self.ge_enable:
            free = c
        else:
            free = DerivativeAerodynamics(self.p, self.span, self.mac,
                                          ge_enable=False).coefficients(u)
        return {
            "CD": c.CD, "CS": c.CS, "CL": c.CL, "Cl": c.Cl, "Cm": c.Cm, "Cn": c.Cn,
            "CY": c.CS,                       # the repo's older spelling, kept for force assembly
            "CL_free": free.CL, "CD_free": free.CD,
            "CL_ratio_ige": c.CL / free.CL if free.CL else 1.0,
            "CD_ratio_ige": c.CD / free.CD if free.CD else 1.0,
        }


class AerodynamicsCalculatorMixin(AtmosphereStateMixin):
    """Adds the coefficient model to an aircraft class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aerodynamics = None

    def _initialize_aerodynamics_calculator(self):
        if self.aerodynamics is None:
            wing = self.VP["wing"]
            self.aerodynamics = DerivativeAerodynamics(
                DerivativeAeroParameters.from_config(self.config),
                span=wing["span"], mac=wing["mac"],
                ge_enable=self.AP.get("ge_enable", True),
            )

    def compute_aerodynamic_coeffs(self, alpha: float, beta: float, aero_action: np.ndarray,
                                   state: Dict[str, np.ndarray]):
        """Compute the coefficients for the current state and store them on the aircraft.

        `aero_action` is post-actuator deflections in DEGREES, packed per
        `falcons.sim.aero_contract.SURFACES`; `aero_inputs` is what converts and names them.
        """
        from falcons.sim.aero_contract import aero_inputs

        self._initialize_aerodynamics_calculator()
        u = aero_inputs(alpha=alpha, beta=beta, v=self.Va,
                        deflections_deg=aero_action,
                        ang_vel=state["angular_vel"], h=-state["position"][2])
        self.coeffs = self.aerodynamics.coefficients_with_diagnostics(u)
