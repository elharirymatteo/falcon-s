# vehicle_config.py
import dataclasses
import numpy as np
import warp as wp
import pandas as pd
import json
from typing import Dict, Any, Optional
from falcons.aircraft.config import AircraftConfig

# ==================== WARP GPU STRUCTS ====================

@wp.struct
class WingParametersStruct:
    span: wp.float32
    area: wp.float32
    mac: wp.float32
    aspect_ratio: wp.float32
    taper_ratio: wp.float32

@wp.struct
class AerodynamicsParametersStruct:
    """The derivative set on the GPU. `free` and `ge_increment` are laid out by
    `CHANNELS`; `ge_increment` is (n_heights, N_CHANNELS) and `hc`/`k_ind` are (n_heights,).

    Rate damping is NOT a separate field any more. CL_q, CMm_q, CMl_p, CMn_r, CS_p and CS_r are
    measured members of the set and are applied inside the coefficient model, so the old scalar
    Clp/Cmq/Cnr would double-count them.
    """
    free: wp.array(dtype=wp.float32)          # (N_CHANNELS,) free-air coefficients
    hc: wp.array(dtype=wp.float32)            # (n,) h/c grid of the ground-effect sweep
    ge_increment: wp.array2d(dtype=wp.float32)  # (n, N_CHANNELS) anchored at the top row
    k_ind: wp.array(dtype=wp.float32)         # (n,) induced-drag factor per height

    k_ind_free: wp.float32
    alpha_run: wp.float32                     # operating point [rad]
    de_run: wp.float32                        # operating point [rad]
    n_heights: wp.int32

@wp.struct
class PropulsionParametersStruct:
    td1: wp.vec3f
    td2: wp.vec3f
    tvi: wp.vec3f
    T_max: wp.float32
    k_m: wp.float32
    k_q: wp.float32
    k_o: wp.float32
    C_p: wp.float32
    Sp: wp.float32

# ==================== PYTHON DATACLASSES ====================

@dataclasses.dataclass
class WingParameters:
    span: float = 5.00
    area: float = 3.32
    mac: float = 0.620
    # aspect_ratio and taper_ratio no longer enter the physics: they fed the empirical
    # ground-effect correlation, which the measured OpenVSP height sweep replaced. Kept because
    # they describe the wing, not because anything reads them.
    aspect_ratio: float = 7.53
    taper_ratio: float = 0.39

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create from JSON config"""
        vehicle_params = config.get('vehicle_params', {})
        wing = vehicle_params.get('wing', {})

        return cls(
            span=wing.get('span', 5.00),
            area=wing.get('area', 3.32),
            mac=wing.get('mac', 0.620),
            aspect_ratio=wing.get('aspect_ratio', 7.53),
            taper_ratio=wing.get('taper_ratio', 0.39),
        )

    def as_warp_struct(self):
        s = WingParametersStruct()
        s.span = self.span
        s.area = self.area
        s.mac = self.mac
        s.aspect_ratio = self.aspect_ratio
        s.taper_ratio = self.taper_ratio
        return s
    

# ==================== OPENVSP DERIVATIVE SET ====================

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


@dataclasses.dataclass
class PropulsionParameters:
    td1: tuple = (0.078, -0.6, 0.004)  # thruster 1 distance from CoG [m]
    td2: tuple = (0.078, 0.6, 0.004)  # thruster 2 distance from CoG [m]
    tvi: tuple = (1.0, 0.0, 0.0)  # thruster vector
    T_max: float = 265  # maximum thrust [N] (take-off)

    # Propeller data
    k_m: float = 37.42  # Motor constant
    k_q: float = 1.1871e-6  # Propeller moment constant
    k_o: float = 797.1268  # K_omega: Propeller speed constant
    C_p: float = 0.57  # Efficiency factor
    Sp: float = np.pi * (0.635**2 / 4)  # Propeller disc area [m^2]

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create from JSON config. Generic over motor count.

        The model has two hard-wired thrusters (td1/td2). Dual-motor uses each motor's position.
        A SINGLE centreline motor is split across the two model thrusters, both at the motor's
        position with HALF the disc area each: per-engine thrust is linear in Sp
        (T = 0.5*rho*Sp*C_p*Vd*(Vd-Va)), so 2*(Sp/2) reproduces exactly one real motor. The two
        equal counter-torques (M_prop_left = -M_prop_right) cancel, i.e. single-prop torque is
        neglected -- fine for a docile trainer."""
        vehicle_params = config.get('vehicle_params', {})
        actuator_system = vehicle_params.get('actuator_system', {})
        motors = actuator_system.get('motors', {})
        motor_list = list(motors.values())

        if not motor_list:
            return cls()  # all defaults

        first = motor_list[0]
        thrust_vector = tuple(first.get('thrust_vector', [1.0, 0.0, 0.0]))
        motor_data = first.get('motor_propeller_data', {})
        Sp = motor_data.get('Sp', 0.3167)

        if len(motor_list) >= 2:
            td1 = tuple(motor_list[0].get('position', [0.078, -0.6, 0.004]))
            td2 = tuple(motor_list[1].get('position', [0.078, 0.6, 0.004]))
            total_max_thrust = sum(m.get('max_thrust', 132.5) for m in motor_list)
            Sp_eff = Sp
        else:                                       # single centreline motor
            pos = tuple(first.get('position', [0.0, 0.0, 0.0]))
            td1 = td2 = pos
            total_max_thrust = first.get('max_thrust', 132.5)
            Sp_eff = Sp / 2.0                        # two model-thrusters each half = one real motor

        return cls(
            td1 = td1,
            td2 = td2,
            tvi = thrust_vector,
            T_max = total_max_thrust,
            k_m = motor_data.get('k_m', 37.42),
            k_q = motor_data.get('k_q', 1.1871e-6),
            k_o = motor_data.get('k_o', 797.1268),
            C_p = motor_data.get('C_p', 0.57),
            Sp = Sp_eff
        )

    def as_warp_struct(self):
        s = PropulsionParametersStruct()
        s.td1 = wp.vec3f(*self.td1)
        s.td2 = wp.vec3f(*self.td2)
        s.tvi = wp.vec3f(*self.tvi)
        s.T_max = self.T_max
        s.k_m = self.k_m
        s.k_q = self.k_q
        s.k_o = self.k_o
        s.C_p = self.C_p
        s.Sp = self.Sp
        return s


@dataclasses.dataclass
class Inertia:
    L_xx: float = 39.71
    L_xy: float = 0.0
    L_xz: float = 8.97
    L_yx: float = 0.0
    L_yy: float = 85.51
    L_yz: float = 0.0
    L_zx: float = 8.97
    L_zy: float = 0.0
    L_zz: float = 114.39

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create from JSON config"""
        vehicle_params = config.get('vehicle_params', {})
        inertia_matrix = vehicle_params.get('inertia_matrix', [
            [39.71, 0.0, 8.97],
            [0.0, 85.51, 0.0],
            [8.97, 0.0, 114.39]
        ])
        
        return cls(
            L_xx=inertia_matrix[0][0],
            L_xy=inertia_matrix[0][1],
            L_xz=inertia_matrix[0][2],
            L_yx=inertia_matrix[1][0],
            L_yy=inertia_matrix[1][1],
            L_yz=inertia_matrix[1][2],
            L_zx=inertia_matrix[2][0],
            L_zy=inertia_matrix[2][1],
            L_zz=inertia_matrix[2][2]
        )

    def as_warp_struct(self):
        J_np = np.array([
            [self.L_xx, self.L_xy, self.L_xz],
            [self.L_yx, self.L_yy, self.L_yz],
            [self.L_zx, self.L_zy, self.L_zz]
        ], dtype=np.float32)

        return wp.mat33(*J_np.flatten())  # Flatten in row-major order


@dataclasses.dataclass
class VehicleParameters:
    """
    Vehicle parameters for the airship 1
    """
    m: float = 120.0  # mass (kg)
    inertia_matrix: list = None  # Inertia matrix (3x3)
    Mac: float = 0.620  # Mean Aerodynamic Chord (m)

    # Sub-components
    WP: WingParameters = None
    J: Inertia = None
    PP: PropulsionParameters = None
    
    # Subsystem configurations
    actuator_system: Dict[str, Any] = None
    sensor_system: Dict[str, Any] = None
    estimator_system: Dict[str, Any] = None

    # Control surface parameters (move from actuator_system extraction)
    T_s: float = 0.2  # time constant for the throttle dynamics
    omega_0: float = 10.0  # natural frequency for the control surfaces
    zeta: float = 1 / np.sqrt(2)  # damping ratio
    # Hard incidence termination [rad]. Not a stall angle: the derivative model has no stall, so
    # this is the edge of the envelope its coefficients were fitted in.
    alpha_max: float = 19.5 * np.pi / 180

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create from JSON config"""
        vehicle_params = config.get('vehicle_params', {})
        aero_params = config.get('aero_params', {})
        
        # Get mass and MAC
        mass = vehicle_params.get('mass', 120.0)
        wing = vehicle_params.get('wing', {})
        mac = wing.get('mac', 0.620)
        
        # Get inertia matrix
        inertia_matrix = vehicle_params.get('inertia_matrix', [
            [39.71, 0.0, 8.97],
            [0.0, 85.51, 0.0],
            [8.97, 0.0, 114.39]
        ])
        
        # Store complete subsystem configurations
        actuator_system = vehicle_params.get('actuator_system', {})
        sensor_system = vehicle_params.get('sensor_system', {})
        estimator_system = vehicle_params.get('estimator_system', {})
        
        # Get actuator parameters for backwards compatibility
        aero_surfaces = actuator_system.get('aero_surfaces', {})
        motors = actuator_system.get('motors', {})
        
        # Get control surface parameters (from elevator as reference)
        elevator = aero_surfaces.get('elevator', {})
        omega_0 = elevator.get('omega_0', 10.0)
        zeta = elevator.get('zeta', 0.70710678119)
        
        # Get motor time constant
        motor = next(iter(motors.values()), {}) if motors else {}
        T_s = motor.get('T', 0.2)
        
        # Hard incidence limit (see alpha_max above)
        alpha_max_rad = np.radians(aero_params.get('alpha_max_deg', 19.5))

        # Create sub-components
        wing_params = WingParameters.from_config(config)
        inertia = Inertia.from_config(config)
        propulsion = PropulsionParameters.from_config(config)
        
        return cls(
            m = mass,
            inertia_matrix = inertia_matrix, 
            Mac = mac,
            WP = wing_params,
            J = inertia,
            PP = propulsion,
            actuator_system = actuator_system,  
            sensor_system = sensor_system,      
            estimator_system = estimator_system, 
            T_s = T_s,
            omega_0 = omega_0,
            zeta = zeta,
            alpha_max = alpha_max_rad
        )


@dataclasses.dataclass
class EnvironmentParameters:
    """
    Environment parameters for the airship
    """
    g: float = 9.81  # (m/s^2)
    dt: float = 0.01  # time step (s)
    solver_type: str = "RK45"
    integration_atol: float = 1e-10
    integration_rtol: float = 1e-8
    seed: Optional[int] = None
    constant_wind: tuple = (0.0, 0.0, 0.0)  
    turbulence: Dict[str, Any] = None  

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create from JSON config"""
        env_params = config.get('environment_params', {})
        
        turbulence_config = env_params.get('turbulence', {'enable': False})
        
        constant_wind = env_params.get('constant_wind', [0.0, 0.0, 0.0])
        
        return cls(
            g=env_params.get('g', 9.81),
            dt=env_params.get('dt', 0.01),
            solver_type=env_params.get('solver_type', 'RK45'),
            integration_atol=env_params.get('integration_atol', 1e-10),
            integration_rtol=env_params.get('integration_rtol', 1e-8),
            seed=env_params.get('seed'),
            constant_wind=tuple(constant_wind),  
            turbulence=turbulence_config  
        )


@dataclasses.dataclass
class ControlLimits:
    """
    Control limits for the airship.
    These limits are used to scale the actions to the range of [-1, 1].
    """
    elevator_limits: wp.vec2 = wp.vec2(-20.0, 20.0)
    aileron_limits: wp.vec2 = wp.vec2(-15.0, 15.0)
    rudder_limits: wp.vec2 = wp.vec2(-15.0, 15.0)
    throttle_limits: wp.vec2 = wp.vec2(0.0, 1.0)

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create from JSON config"""
        vehicle_params = config.get('vehicle_params', {})
        actuator_system = vehicle_params.get('actuator_system', {})
        aero_surfaces = actuator_system.get('aero_surfaces', {})
        motors = actuator_system.get('motors', {})
        
        # Get control surface limits
        elevator = aero_surfaces.get('elevator', {})
        aileron = aero_surfaces.get('ailerons', {})
        rudder = aero_surfaces.get('rudder', {})
        
        # Get throttle limits from first motor
        motor = next(iter(motors.values()), {}) if motors else {}
        
        return cls(
            elevator_limits=wp.vec2(
                elevator.get('min_deflection', -20.0),
                elevator.get('max_deflection', 20.0)
            ),
            aileron_limits=wp.vec2(
                aileron.get('min_deflection', -15.0),
                aileron.get('max_deflection', 15.0)
            ),
            rudder_limits=wp.vec2(
                rudder.get('min_deflection', -15.0),
                rudder.get('max_deflection', 15.0)
            ),
            throttle_limits=wp.vec2(
                motor.get('min_throttle', 0.0),
                motor.get('max_throttle', 1.0)
            )
        )


@dataclasses.dataclass
class DefaultInitialState:
    """Default initial state configuration"""
    position: tuple = (0.0, 0.0, -1.0)
    linear_vel: tuple = (28.0, 0.0, 0.0)
    angular_vel: tuple = (0.0, 0.0, 0.0)
    orientation: tuple = (1.0, 0.0, 0.0, 0.0)  # [qw, qx, qy, qz]

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Create from JSON config"""
        default_state = config.get('default_initial_state', {})

        # Get quaternion from JSON [q0, q1, q2, q3] and convert to WARP [q1, q2, q3, q0]
        json_quat = default_state.get('orientation', [1.0, 0.0, 0.0, 0.0])
        warp_quat = (json_quat[1], json_quat[2], json_quat[3], json_quat[0])  # Reorder
    
        return cls(
            position=tuple(default_state.get('position', [0.0, 0.0, -1.0])),
            linear_vel=tuple(default_state.get('linear_vel', [28.0, 0.0, 0.0])),
            angular_vel=tuple(default_state.get('angular_vel', [0.0, 0.0, 0.0])),
            orientation=warp_quat  
        )

# ==================== FACTORY ====================

# Factory function to create all parameters from config
def load_params(aircraft_name: str):
    """All parameter structs for one aircraft, built from its packaged JSON + OpenVSP CSVs."""
    cfg = AircraftConfig(aircraft_name)
    config = cfg.load()
    return {
        'vehicle_params': VehicleParameters.from_config(config),
        'environment_params': EnvironmentParameters.from_config(config),
        'control_limits': ControlLimits.from_config(config),
        'aero_params': DerivativeAeroParameters.from_config(config),
        'default_initial_state': DefaultInitialState.from_config(config)
    }
