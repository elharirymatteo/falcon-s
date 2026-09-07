import numpy as np
from typing import Dict, Any

class AtmosphereStateMixin:
    """Mixin for calculating aerodynamic state parameters"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Aerodynamic state variables
        self.Va: float = 0.0      # airspeed
        self.alpha: float = 0.0   # angle of attack  
        self.beta: float = 0.0    # sideslip angle
        self.Q: float = 0.0       # dynamic pressure
        self.hr: float = 0.0      # height ratio (for plotting)
    
    def set_aerodynamic_parameters(self) -> None:
        """
        Compute aerodynamic parameters from current state
        
        Updates: Va, alpha, beta, Q, hr 
        """
        # Calculate airspeed vector and angular velocity relative to gusts
        self.airspeed_vector = self.state['linear_vel'] - self.linear_gusts
        self.angular_vel_rel = self.state['angular_vel'] - self.angular_gusts

        # Airspeed magnitude
        self.Va = np.linalg.norm(self.airspeed_vector)
        
        # Angle of attack (atan2 handles all quadrants correctly)
        self.alpha = np.arctan2(self.airspeed_vector[2], self.airspeed_vector[0])
        
        # Sideslip angle (asin with safety for numerical stability)
        self.beta = np.arcsin(np.clip(self.airspeed_vector[1] / (self.Va + 1e-6), -1, 1))

        # Air density from standard atmosphere model
        self.rho = self.get_air_density()
        
        # Dynamic pressure
        self.Q = 0.5 * self.rho * self.Va**2

        # Height ratio (for plotting)
        self.hr = self.state['position'][2] / self.VP['wing']['span']
    
    def get_aerodynamic_vars(self) -> Dict[str, float]:
        """Get current aerodynamic state as dictionary"""
        return {
            'Va': self.Va,
            'alpha': self.alpha, 
            'beta': self.beta,
            'Q': self.Q,
            'rho': self.rho, 
        }
    
    def get_air_density(self) -> float:
        """Get air density as a function of altitude (ISA model up to 11km)"""
        # ISO 2533:1975 standard atmosphere parameters
        T_0 = 288.15                          # Sea level standard temperature [K]
        p_0 = 101325.0                        # Sea level standard pressure [Pa]
        L = 0.0065                            # Temperature lapse rate [K/m]
        g = 9.80665                           # Gravity used in ISA [m/s^2]
        R = 287.05287                         # Specific gas constant for dry air [J/(kg·K)]
        # Altitude is negative down in NED, so use -position[2]
        h = max(0.0, -self.state['position'][2])  # Clamp to >= 0

        if h > 11000.0:
            raise ValueError(f"Altitude {h:.1f} m exceeds ISA troposphere model limit (11,000 m).")

        # Troposphere
        T = T_0 - L * h
        p = p_0 * (T / T_0) ** (g / (R * L))
        rho = p / (R * T)

        return rho