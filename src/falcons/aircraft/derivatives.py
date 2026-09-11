"""The OpenVSP derivative set, loaded. Deliberately free of warp.

This is the aerodynamic model's data, and the CPU reference plant is built on it, so it must be
usable without a GPU stack installed: `tools/jsbsim_validate` runs the CPU plant against JSBSim in
an environment with neither warp nor torch, and that is the whole point of it as an independent
check. `params.py` re-exports everything here, so every existing import site still works, and
`DerivativeAeroParameters.as_warp_struct` imports warp locally when a GPU mirror is actually
wanted.
"""
import dataclasses
from typing import Any, Dict

import numpy as np
import pandas as pd


# The coefficient channels the aerodynamic model actually reads, in a FIXED order: the ground-effect
# increment table is an (n_heights, N_CHANNELS) array indexed by this list, and the warp port shares
# the layout. Order is load-bearing -- do not sort or regroup it.
#
# Roughly two thirds of what OpenVSP reports is deliberately absent (CMm_Beta, CL_p, CS_Alpha,
# CMl_q, CD_Beta, ...). Those are numerical noise in the VLM solution that should be identically
# zero by symmetry, so the model drops them rather than feeding solver residue into the dynamics.
# This is intentional: do not "complete" the set.
CHANNELS = (
    "CD_Total", "CD_Alpha", "CD_elevator", "CD_q",
    "CL_Total", "CL_Alpha", "CL_elevator", "CL_q",
    "CMm_Total", "CMm_Alpha", "CMm_elevator", "CMm_q",
    "CS_Beta", "CS_aileron", "CS_rudder", "CS_p", "CS_r",
    "CMl_Beta", "CMl_aileron", "CMl_rudder", "CMl_p", "CMl_r",
    "CMn_Beta", "CMn_aileron", "CMn_rudder", "CMn_p", "CMn_r",
)
IDX = {name: i for i, name in enumerate(CHANNELS)}


def _induced_drag_factor(cd_alpha, cl_total, cl_alpha):
    """k_ind in CD = ... + k_ind*(CL_Alpha*da)^2, the quadratic that gives the locally linearised
    drag polar its curvature. Precomputed per height so the division never runs per step."""
    denom = 2.0 * cl_total * cl_alpha
    if np.any(np.abs(denom) < 1e-9):
        raise ValueError(
            "induced-drag factor is singular: CL_Total*CL_Alpha ~ 0 in the derivative data, so "
            "CD_Alpha/(2*CL_Total*CL_Alpha) does not exist. The set was probably measured at a "
            "flight condition carrying no lift."
        )
    return cd_alpha / denom


@dataclasses.dataclass
class DerivativeAeroParameters:
    """OpenVSP linear derivative set for one airframe, plus its measured ground-effect surface.

    Built from <plane>_derivatives.csv (one VSPAERO operating point) and
    <plane>_ge_derivatives.csv (the same set swept over height). Everything here is init-time work:
    the per-step model does an interpolation and a handful of multiplies, no divisions and no unit
    conversions.

    Ground effect is stored as an ADDITIVE INCREMENT anchored at the top of the height sweep:

        increment[i] = ge[i] - ge[top]        so increment[top] == 0 exactly

    which makes the in-ground-effect coefficients `free + increment(h/c)` equal the free-air set
    exactly at and above the table top. A plain clamped interpolation is then correct at both ends
    -- no separate "above the table" branch, and no step discontinuity at the hand-off, which the
    raw table does have (its top row is close to but not equal to the free-air set).

    Below the sweep's lowest height the increment freezes at its deepest measured value. That floor
    is real: the sweep starts at h/c = 2.07 (h/b = 0.20), so ground effect deeper than that is
    NOT modelled, it is held constant. Re-run the extraction lower if that regime matters.
    """
    derivatives_file: str = ""
    ge_derivatives_file: str = ""

    def __post_init__(self):
        d = self._read_named(self.derivatives_file)

        # Operating point the tables were MEASURED at: the model works in deltas from here, so
        # these are the only degrees in the whole path and they are converted once.
        #
        # A missing row is NOT defaulted to zero. `de_run` offsets every elevator command, so
        # guessing it wrong biases the pitch axis by a fixed amount at every step -- silently, and
        # in trim, which is where it would be least visible. An extraction that did not record its
        # own deflections has to be re-run.
        for row in ("FC_AoA_", "deflect_elevator_deg"):
            if row not in d:
                raise ValueError(
                    f"{self.derivatives_file}: no {row!r} row, so the operating point the "
                    f"coefficients were linearised about is unknown. Re-export this airframe -- "
                    f"the model subtracts this value from every command, and assuming zero would "
                    f"bias the axis in trim.")
        self.alpha_run: float = float(np.radians(d["FC_AoA_"]))
        self.de_run: float = float(np.radians(d["deflect_elevator_deg"]))

        # Reference geometry, kept for provenance. The JSON is authoritative and
        # AircraftConfig.load has already refused to proceed if the two disagree.
        self.span_ref: float = d["FC_Bref_"]
        self.area_ref: float = d["FC_Sref_"]
        self.mac_ref: float = d["FC_Cref_"]
        self.v_ref: float = d["FC_Vinf_"]

        missing = [c for c in CHANNELS if c not in d]
        if missing:
            raise ValueError(f"{self.derivatives_file}: missing coefficients {missing}")
        self.free: np.ndarray = np.array([d[c] for c in CHANNELS], dtype=np.float64)
        self.k_ind_free: float = float(_induced_drag_factor(
            self.free[IDX["CD_Alpha"]], self.free[IDX["CL_Total"]], self.free[IDX["CL_Alpha"]]))

        self._load_ground_effect()

    @staticmethod
    def _read_named(path: str) -> dict:
        df = pd.read_csv(path)
        return dict(zip(df["name"], df["value"].astype(float)))

    def _load_ground_effect(self):
        """The height sweep -> (hc, increments, k_ind). The CSV is long-format
        `h_m,name,value`; heights are in METRES and h/c is carried as its own `h_over_c` row."""
        df = pd.read_csv(self.ge_derivatives_file)
        table = df.pivot(index="h_m", columns="name", values="value").sort_index()

        missing = [c for c in CHANNELS if c not in table.columns]
        if missing:
            raise ValueError(f"{self.ge_derivatives_file}: missing coefficients {missing}")
        if "h_over_c" not in table.columns:
            raise ValueError(f"{self.ge_derivatives_file}: no h_over_c column to index height by")

        self.hc: np.ndarray = table["h_over_c"].to_numpy(dtype=np.float64)
        if np.any(np.diff(self.hc) <= 0):
            raise ValueError(f"{self.ge_derivatives_file}: h_over_c is not strictly increasing")

        measured = table[list(CHANNELS)].to_numpy(dtype=np.float64)   # (n, N_CHANNELS)
        self.ge_increment: np.ndarray = measured - measured[-1]        # anchored: last row is 0

        # k_ind from the ANCHORED coefficients, not the raw table, so that at the top row it equals
        # k_ind_free exactly and the interpolation is continuous with the free-air branch.
        anchored = self.free + self.ge_increment
        self.k_ind: np.ndarray = _induced_drag_factor(
            anchored[:, IDX["CD_Alpha"]], anchored[:, IDX["CL_Total"]], anchored[:, IDX["CL_Alpha"]])

    @property
    def hc_floor(self) -> float:
        """Lowest h/c the ground-effect sweep covers; below this the increment is held."""
        return float(self.hc[0])

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        aero_params = config.get("aero_params", {})
        return cls(
            derivatives_file=aero_params["derivatives_file"],
            ge_derivatives_file=aero_params["ge_derivatives_file"],
        )

    def as_warp_struct(self, device: str = "cuda"):
        """The GPU mirror. Warp is imported HERE rather than at module scope so that loading a
        derivative set costs nothing but numpy and pandas -- see the module docstring."""
        import warp as wp

        from falcons.aircraft.params import AerodynamicsParametersStruct
        s = AerodynamicsParametersStruct()
        s.free = wp.array(self.free, dtype=wp.float32, device=device)
        s.hc = wp.array(self.hc, dtype=wp.float32, device=device)
        s.ge_increment = wp.array(self.ge_increment, dtype=wp.float32, ndim=2, device=device)
        s.k_ind = wp.array(self.k_ind, dtype=wp.float32, device=device)
        s.k_ind_free = wp.float32(self.k_ind_free)
        s.alpha_run = wp.float32(self.alpha_run)
        s.de_run = wp.float32(self.de_run)
        s.n_heights = wp.int32(len(self.hc))
        return s
