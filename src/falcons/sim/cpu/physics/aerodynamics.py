import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import pandas as pd
from .atmosphere import AtmosphereStateMixin

class AerodynamicsBase(ABC):
    """Base class for aerodynamic calculations"""
    
    def __init__(self, aerodynamic_params, vehicle_params, environment_params):
        self.AP = aerodynamic_params
        self.VP = vehicle_params
        self.EP = environment_params
    
    @abstractmethod
    def get_force_coefficients(self, alpha: float, beta: float, aero_action: np.ndarray) -> Tuple[float, float, float]:
        """Get forces coefficient (CD, CY, CL)"""
        pass
    
    @abstractmethod
    def get_moment_coefficients(self, alpha: float, beta: float, aero_action: np.ndarray) -> Tuple[float, float, float]:
        """Get moment coefficients (Cl, Cm, Cn)"""
        pass
    
    @abstractmethod
    def get_ground_effect_factors(self, state: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """Get ground effect factors for lift and drag"""
        pass


class PolynomialAerodynamics(AerodynamicsBase):
    """Aerodynamics using polynomial coefficient model (e.g., from OpenVSP)"""
    
    def __init__(self, aerodynamic_params, vehicle_params, environment_params, poly_params: pd.DataFrame):
        super().__init__(aerodynamic_params, vehicle_params, environment_params)
        self.poly_params = poly_params
        
        # Validate polynomial parameters
        self._validate_poly_params()
    
    def _validate_poly_params(self):
        """Validate that polynomial parameters contain required columns"""
        required_columns = ['alpha', 'beta', 'delta_e', 'delta_a', 'delta_r', 
                          'CL', 'CD', 'CY', 'CMx', 'CMy', 'CMz']
        
        for col in required_columns:
            if col not in self.poly_params.columns:
                raise ValueError(f"Missing required column '{col}' in polynomial parameters")
    
    def get_coefficient(self, coef_name: str, alpha: float, beta: float, aero_action: np.ndarray) -> float:
        """
        Compute aerodynamic coefficient using polynomial model
        
        Args:
            coef_name: Coefficient name (CL, CD, CY, CMx, CMy, CMz)
            alpha: Angle of attack (rad)
            beta: Sideslip angle (rad)
            aero_action: Aerodynamic control action vector [elevator, ailerons, rudder, ...]
            
        Returns:
            Coefficient value
        """
        if coef_name not in self.poly_params.columns:
            raise ValueError(f"Coefficient '{coef_name}' not found in polynomial parameters")
        
        # Get exponents from polynomial parameters
        alpha_exp = self.poly_params['alpha'].values
        beta_exp = self.poly_params['beta'].values
        elev_exp = self.poly_params['delta_e'].values
        aileron_exp = self.poly_params['delta_a'].values
        rudder_exp = self.poly_params['delta_r'].values
        coefs = self.poly_params[coef_name].values

        # Convert angles to degrees for polynomial (assuming polynomial was fitted in degrees)
        alpha_deg = alpha * 180 / np.pi
        beta_deg = beta * 180 / np.pi

        # Extract control surface deflections (assuming normalized inputs [-1, 1])
        # Handle different aero_action sizes gracefully
        elevator_deflection = aero_action[0] if len(aero_action) > 0 else 0.0
        aileron_deflection = aero_action[1] if len(aero_action) > 1 else 0.0
        rudder_deflection = aero_action[2] if len(aero_action) > 2 else 0.0

        # Compute polynomial terms
        terms = (
            (alpha_deg ** alpha_exp) *
            (beta_deg ** beta_exp) *
            (elevator_deflection ** elev_exp) *
            (aileron_deflection ** aileron_exp) *
            (rudder_deflection ** rudder_exp)
        )

        return np.sum(coefs * terms)
    
    def get_force_coefficients(self, alpha: float, beta: float, aero_action: np.ndarray) -> Tuple[float, float, float]:
        """
        Get force coefficients
        
        Returns:
            (CD, CY, CL): Drag, side force, lift coefficients
        """
        CD = self.get_coefficient("CD", alpha, beta, aero_action)
        CY = self.get_coefficient("CY", alpha, beta, aero_action)
        CL = self.get_coefficient("CL", alpha, beta, aero_action)

        return CD, CY, CL

    def get_moment_coefficients(self, alpha: float, beta: float, aero_action: np.ndarray) -> Tuple[float, float, float]:
        """
        Get moment coefficients
        
        Returns:
            (Cl, Cm, Cn): Roll, pitch, yaw moment coefficients
        """
        Cl = self.get_coefficient("CMx", alpha, beta, aero_action)  # Roll moment coefficient
        Cm = self.get_coefficient("CMy", alpha, beta, aero_action)  # Pitch moment coefficient
        Cn = self.get_coefficient("CMz", alpha, beta, aero_action)  # Yaw moment coefficient
        
        return Cl, Cm, Cn
    
    def get_ground_effect_factors(self, state: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """
        Compute ground effect factors for lift and drag using empirical models
        
        Args:
            state: Aircraft state dictionary containing position
            
        Returns:
            (mu_l, mu_d): Ground effect factors for lift and drag
        """
        # Get wing geometry parameters
        TR = self.VP['wing']['taper_ratio'] # Taper ratio
        AR = self.VP['wing']['aspect_ratio'] # Aspect ratio
        b = self.VP['wing']['span'] # Wing span
        
        # Calculate height of wing above ground
        height = -state['position'][2]  # Negative because z-axis points down in NED
        h = height + self.VP['wing']['cg_offset_vector'][2] # Height of wing above ground (z_w is wing offset from CG)
        
        # Height-to-span ratio
        h_over_b = np.abs(h / b)
        
        # Lift ground effect factor (based on empirical correlation)
        # This model accounts for wing geometry effects on ground effect
        geometric_factor = (1 - 2.25 * (TR ** 0.00273 - 0.997) * (AR ** 0.717 + 13.6))
        height_factor = (288 * h_over_b**0.787 * np.exp(-9.14 * h_over_b**0.327)) / (AR ** 0.882)
        mu_l = 1 + geometric_factor * height_factor
        
        # Drag ground effect factor (based on empirical correlation)
        # First term: base drag reduction
        term1 = 1 - (1 - 0.157 * np.maximum(0, (TR ** 0.775 - 0.373)) * np.maximum(0, (AR ** 0.417 - 1.27))) * np.exp(-4.74 * np.maximum(0, np.abs(h/b) ** 0.814))
        term2 = h_over_b ** 2 * np.exp(-3.88 * np.maximum(0, h_over_b ** 0.758))
        
        mu_d = term1 - term2
        
        return mu_l, mu_d
    
    def get_coefficients_with_ground_effect(self, alpha: float, beta: float, aero_action: np.ndarray, 
                                          state: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Get all aerodynamic coefficients including ground effect
        
        Args:
            alpha: Angle of attack (rad)
            beta: Sideslip angle (rad)
            aero_action: Aerodynamic control action vector
            state: Aircraft state dictionary
            
        Returns:
            Dictionary with all coefficients including ground effect
        """
        # Get out-of-ground-effect coefficients
        CD_oge, CY, CL_oge = self.get_force_coefficients(alpha, beta, aero_action)
        Cl, Cm, Cn = self.get_moment_coefficients(alpha, beta, aero_action)
        
        # Apply ground effect to lift and drag
        mu_l, mu_d = self.get_ground_effect_factors(state)
        CL = CL_oge * mu_l
        CD = CD_oge * mu_d * mu_l**2
        
        return {
            'CD': CD,
            'CY': CY,
            'CL': CL,
            'Cl': Cl,
            'Cm': Cm,
            'Cn': Cn,
            'CL_oge': CL_oge,
            'CD_oge': CD_oge,
            'mu_l': mu_l,
            'mu_d': mu_d
        }


class ClassicalAerodynamics(AerodynamicsBase):
    """THIS CLASS IS JUST A PLACEHOLDER"""
    """Classical aerodynamics using linear lift curve and drag polar"""
    def __init__(self, aerodynamic_params, vehicle_params, environment_params):
        super().__init__(aerodynamic_params, vehicle_params, environment_params)
        
        # Validate that required parameters exist
        self._validate_classical_params()
    
    def _validate_classical_params(self):
        """Validate that classical aerodynamic parameters are available"""
        required_attrs = ['CL_alpha', 'CD_0', 'CD_alpha2', 'alpha_0']
        for attr in required_attrs:
            if not hasattr(self.AP, attr):
                raise ValueError(f"Missing required aerodynamic parameter: {attr}")
    
    def get_lift_coefficient(self, alpha: float, beta: float, aero_action: np.ndarray) -> float:
        """Get lift coefficient using linear lift curve"""
        # Basic lift curve: CL = CL_alpha * (alpha - alpha_0) + control effects
        CL_basic = self.AP.CL_alpha * (alpha - self.AP.alpha_0)
        
        # Add elevator effect
        elevator_deflection = aero_action[0] if len(aero_action) > 0 else 0.0
        CL_elevator = self.AP.CL_delta_e * elevator_deflection if hasattr(self.AP, 'CL_delta_e') else 0
        
        return CL_basic + CL_elevator
    
    def get_drag_coefficient(self, alpha: float, beta: float, aero_action: np.ndarray) -> float:
        """Get drag coefficient using drag polar"""
        CL = self.get_lift_coefficient(alpha, beta, aero_action)
        
        # Drag polar: CD = CD_0 + CD_alpha2 * alpha^2 + K * CL^2
        CD_basic = self.AP.CD_0 + self.AP.CD_alpha2 * alpha**2
        
        # Induced drag
        if hasattr(self.AP, 'e') and hasattr(self.VP, 'AR_w'):
            K = 1 / (np.pi * self.AP.e * self.VP.AR_w)
            CD_induced = K * CL**2
        else:
            CD_induced = 0
        
        return CD_basic + CD_induced
    
    def get_side_force_coefficient(self, alpha: float, beta: float, aero_action: np.ndarray) -> float:
        """Get side force coefficient"""
        CY_beta = self.AP.CY_beta * beta if hasattr(self.AP, 'CY_beta') else 0
        
        # Add rudder effect
        rudder_deflection = aero_action[2] if len(aero_action) > 2 else 0.0
        CY_rudder = self.AP.CY_delta_r * rudder_deflection if hasattr(self.AP, 'CY_delta_r') else 0
        
        return CY_beta + CY_rudder
    
    def get_force_coefficients(self, alpha: float, beta: float, aero_action: np.ndarray) -> Tuple[float, float, float]:
        """Get force coefficients (CD, CY, CL)"""
        CL = self.get_lift_coefficient(alpha, beta, aero_action)
        CD = self.get_drag_coefficient(alpha, beta, aero_action)
        CY = self.get_side_force_coefficient(alpha, beta, aero_action)
        return CD, CY, CL
    
    def get_moment_coefficients(self, alpha: float, beta: float, aero_action: np.ndarray) -> Tuple[float, float, float]:
        """Get moment coefficients using classical approach"""
        # Roll moment
        Cl_beta = self.AP.Cl_beta * beta if hasattr(self.AP, 'Cl_beta') else 0
        aileron_deflection = aero_action[1] if len(aero_action) > 1 else 0.0
        Cl_aileron = self.AP.Cl_delta_a * aileron_deflection if hasattr(self.AP, 'Cl_delta_a') else 0
        Cl = Cl_beta + Cl_aileron
        
        # Pitch moment
        Cm_alpha = self.AP.Cm_alpha * alpha if hasattr(self.AP, 'Cm_alpha') else 0
        elevator_deflection = aero_action[0] if len(aero_action) > 0 else 0.0
        Cm_elevator = self.AP.Cm_delta_e * elevator_deflection if hasattr(self.AP, 'Cm_delta_e') else 0
        Cm = self.AP.Cm_0 + Cm_alpha + Cm_elevator if hasattr(self.AP, 'Cm_0') else Cm_alpha + Cm_elevator
        
        # Yaw moment
        Cn_beta = self.AP.Cn_beta * beta if hasattr(self.AP, 'Cn_beta') else 0
        rudder_deflection = aero_action[2] if len(aero_action) > 2 else 0.0
        Cn_rudder = self.AP.Cn_delta_r * rudder_deflection if hasattr(self.AP, 'Cn_delta_r') else 0
        Cn = Cn_beta + Cn_rudder
        
        return Cl, Cm, Cn
    
    def get_ground_effect_factors(self, state: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """Simple ground effect model for classical aerodynamics"""
        height = -state['position'][2]
        h_over_b = height / self.VP.b_w
        
        # Simple exponential ground effect model
        if h_over_b < 1.0:
            mu_l = 1 + 0.5 * np.exp(-2 * h_over_b)  # Lift increase
            mu_d = 1 - 0.3 * np.exp(-3 * h_over_b)  # Drag decrease
        else:
            mu_l = 1.0
            mu_d = 1.0
        
        return mu_l, mu_d
    
    def get_coefficients_with_ground_effect(self, alpha: float, beta: float, aero_action: np.ndarray, 
                                          state: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Get all aerodynamic coefficients including ground effect"""
        
        # Get force coefficients
        CD_oge, CY, CL_oge = self.get_force_coefficients(alpha, beta, aero_action)
        Cl, Cm, Cn = self.get_moment_coefficients(alpha, beta, aero_action)
        
        # Apply ground effect
        mu_l, mu_d = self.get_ground_effect_factors(state)
        CL = CL_oge * mu_l
        CD = CD_oge * mu_d
        
        return {
            'CD': CD,
            'CY': CY,
            'CL': CL,
            'Cl': Cl,
            'Cm': Cm,
            'Cn': Cn,
            'CL_oge': CL_oge,
            'CD_oge': CD_oge,
            'mu_l': mu_l,
            'mu_d': mu_d
        }


class AerodynamicsFactory:
    """Factory for creating aerodynamics models"""
    
    @staticmethod
    def create_aerodynamics(aero_type: str, aerodynamic_params, vehicle_params, 
                          environment_params, **kwargs) -> AerodynamicsBase:
        """
        Create aerodynamics model based on type
        
        Args:
            aero_type: Type of aerodynamics ('polynomial', 'classical')
            aerodynamic_params: Aerodynamic parameters
            vehicle_params: Vehicle parameters
            environment_params: Environment parameters
            **kwargs: Additional parameters (e.g., poly_params for polynomial)
            
        Returns:
            Aerodynamics model instance
        """
        if aero_type == 'polynomial':
            if 'poly_params' not in kwargs:
                raise ValueError("poly_params required for polynomial aerodynamics")
            return PolynomialAerodynamics(
                aerodynamic_params, vehicle_params, environment_params, 
                kwargs['poly_params']
            )
        elif aero_type == 'classical':
            return ClassicalAerodynamics(
                aerodynamic_params, vehicle_params, environment_params
            )
        else:
            raise ValueError(f"Unknown aerodynamics type: {aero_type}")


class AerodynamicsCalculatorMixin(AtmosphereStateMixin):
    """Mixin to add aerodynamics computation to aircraft classes"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize aerodynamics calculator based on available data
        self.aerodynamics = None
    
    def _initialize_aerodynamics_calculator(self):
        """Initialize aerodynamics calculator based on available parameters"""
        if self.aerodynamics is None:
            aero_type = self.AP.get('type', 'polynomial')

            if aero_type == 'polynomial' and self.poly_params is not None:
                self.aerodynamics = AerodynamicsFactory.create_aerodynamics(
                    'polynomial', self.AP, self.VP, self.EP, poly_params=self.poly_params
                )
            else:
                self.aerodynamics = AerodynamicsFactory.create_aerodynamics(
                    'classical', self.AP, self.VP, self.EP
                )

    def compute_aerodynamic_coeffs(self, alpha: float, beta: float, aero_action: np.ndarray, 
                           state: Dict[str, np.ndarray]):
        """
        Compute aerodynamic coefficients and store them as instance attributes
        
        Args:
            alpha: Angle of attack (rad)
            beta: Sideslip angle (rad)
            aero_action: Aerodynamic control action vector
            state: Aircraft state dictionary
        """
        # Ensure aerodynamics calculator is initialized
        self._initialize_aerodynamics_calculator()
        
        # Compute coefficients with ground effect
        coeffs = self.aerodynamics.get_coefficients_with_ground_effect(
            alpha, beta, aero_action, state
        )
        
        # Store coefficients as instance attributes
        self.coeffs = coeffs