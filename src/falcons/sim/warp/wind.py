import warp as wp
import numpy as np
from typing import Tuple, Optional
from .dryden import DrydenTurbulenceModelWARP

@wp.kernel
def update_wind_effects(
    # Aircraft state
    orientations: wp.array(dtype=wp.quatf),
    
    # Wind parameters
    constant_wind_inertial: wp.vec3f,
    
    # Turbulence inputs (from turbulence model)
    turbulent_linear_gusts: wp.array(dtype=wp.vec3f),
    turbulent_angular_gusts: wp.array(dtype=wp.vec3f),
    
    # Control flags
    enable_turbulence: bool,
    
    # Outputs
    total_linear_gusts: wp.array(dtype=wp.vec3f),
    total_angular_gusts: wp.array(dtype=wp.vec3f),
    constant_wind_body: wp.array(dtype=wp.vec3f)
):
    """
    Update wind effects for all environments in parallel
    Combines constant wind and turbulent gusts
    """
    tid = wp.tid()
    
    # 1. Convert constant wind from inertial to body frame
    orientation = orientations[tid]
    wind_body = wp.quat_rotate_inv(orientation, constant_wind_inertial)
    constant_wind_body[tid] = wind_body
    
    # 2. Get turbulent gusts (if enabled)
    if enable_turbulence:
        turbulent_linear = turbulent_linear_gusts[tid]
        turbulent_angular = turbulent_angular_gusts[tid]
    else:
        turbulent_linear = wp.vec3f(0.0, 0.0, 0.0)
        turbulent_angular = wp.vec3f(0.0, 0.0, 0.0)
    
    # 3. Combine wind components
    # Linear gusts = turbulent + constant wind (both in body frame)
    total_linear_gusts[tid] = turbulent_linear + wind_body
    
    # Angular gusts = only turbulent (constant wind doesn't create angular gusts)
    total_angular_gusts[tid] = turbulent_angular

class WindModelWARP:
    """
    WARP implementation of wind model managing constant wind and turbulence effects
    """
    
    def __init__(self, num_envs: int, device: str, EP, wingspan: float):
        """
        Initialize WARP wind model
        
        Args:
            num_envs: Number of parallel environments
            device: WARP device ("cuda" or "cpu")
            EP: EnvironmentParameters object from config
            wingspan: Aircraft wingspan for turbulence scaling
        """
        self.num_envs = num_envs
        self.device = device
        self.EP = EP
        self.wingspan = wingspan
        
        # Create arrays to hold wind effects
        self.total_linear_gusts = wp.zeros((num_envs,), dtype=wp.vec3f, device=device)
        self.total_angular_gusts = wp.zeros((num_envs,), dtype=wp.vec3f, device=device)
        self.constant_wind_body = wp.zeros((num_envs,), dtype=wp.vec3f, device=device)
        self.turbulent_linear_gusts = wp.zeros((num_envs,), dtype=wp.vec3f, device=device)
        self.turbulent_angular_gusts = wp.zeros((num_envs,), dtype=wp.vec3f, device=device)
        
        # Initialize random seed
        self.base_seed = None
        
        # Determine if wind is actually enabled
        self.wind_enabled = self._is_wind_enabled()
        
        if self.wind_enabled:
            # Extract constant wind
            self.constant_wind_inertial = wp.vec3f(
                EP.constant_wind[0], EP.constant_wind[1], EP.constant_wind[2]
            )
            
            # Setup turbulence directly in constructor
            if EP.turbulence and EP.turbulence.get('enable', False):
                model_type = EP.turbulence.get('model', 'dryden')
                
                if model_type == 'dryden':
                    intensity = EP.turbulence.get('intensity', 'light')
                    self.turbulence_model = DrydenTurbulenceModelWARP(
                        num_envs=num_envs,
                        device=device,
                        dt=EP.dt,
                        b=wingspan,
                        intensity=intensity
                    )
                else:
                    print(f"  Warning: Unknown turbulence model '{model_type}', turbulence disabled")
                    self.turbulence_model = None
            else:
                self.turbulence_model = None
                
            print(f"Wind model enabled - Constant: {list(EP.constant_wind)}, Turbulence: {self.turbulence_model is not None}")
        else:
            # Disabled - set all effects to zero
            self.constant_wind_inertial = wp.vec3f(0.0, 0.0, 0.0)
            self.turbulence_model = None
            # print("Wind model disabled - all effects set to zero")
    
    def _is_wind_enabled(self) -> bool:
        """Determine if any wind effects are enabled"""
        # Check constant wind
        has_constant_wind = (hasattr(self.EP, 'constant_wind') and 
                           any(abs(x) > 1e-6 for x in self.EP.constant_wind))
        
        # Check turbulence
        has_turbulence = (hasattr(self.EP, 'turbulence') and 
                         self.EP.turbulence and 
                         self.EP.turbulence.get('enable', False))
        
        return has_constant_wind or has_turbulence
    
    def update_wind(self, orientations: wp.array, airspeeds: wp.array, altitudes: wp.array):
        """
        Update wind effects
        
        Args:
            orientations: Aircraft orientations (quaternions)
            airspeeds: Aircraft airspeeds (m/s)
            altitudes: Aircraft altitudes (m)

        returns:
            total_linear_gusts: Combined linear wind gusts in body frame
            total_angular_gusts: Combined angular wind gusts in body frame    
        """
        if not self.wind_enabled:
            # Wind disabled - arrays stay zero, no computation
            return self.total_linear_gusts, self.total_angular_gusts
        
        # Wind enabled
        if self.turbulence_model is not None:
            self.turbulent_linear_gusts, self.turbulent_angular_gusts = self.turbulence_model.update_dryden(airspeeds, altitudes)
        
        # Combine effects
        wp.launch(
            kernel=update_wind_effects,
            dim=self.num_envs,
            inputs=[
                orientations, self.constant_wind_inertial,
                self.turbulent_linear_gusts, self.turbulent_angular_gusts,
                self.turbulence_model is not None,
                self.total_linear_gusts, self.total_angular_gusts,
                self.constant_wind_body
            ],
            device=self.device
        )
        
        return self.total_linear_gusts, self.total_angular_gusts
    
    def get_linear_gusts(self, positions: wp.array, orientations: wp.array, airspeeds: wp.array) -> wp.array:
        """
        Get current linear wind gusts for aerodynamic calculations
        
        Args:
            positions: Aircraft positions (for altitude extraction)
            orientations: Aircraft orientations 
            airspeeds: Aircraft airspeeds
            
        Returns:
            Linear wind gusts in body frame
        """
        # Extract altitudes from positions
        altitudes = wp.zeros((self.num_envs,), dtype=wp.float32, device=self.device)
        wp.launch(
            kernel=extract_altitude_kernel,
            dim=self.num_envs,
            inputs=[positions, altitudes],
            device=self.device
        )
        
        # Update wind effects
        self.update_wind(orientations, airspeeds, altitudes)
        
        return self.total_linear_gusts
    
    def get_constant_wind_body(self) -> wp.array:
        """Get constant wind in body frame for all environments"""
        return self.constant_wind_body
    
    def get_wind_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current wind values as numpy arrays"""
        return self.total_linear_gusts.numpy(), self.total_angular_gusts.numpy()
    
    def get_turbulence_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current turbulence values as numpy arrays (for debugging)"""
        if self.turbulence_model is not None:
            return self.turbulence_model.get_gusts_numpy()
        else:
            return np.zeros((self.num_envs, 3)), np.zeros((self.num_envs, 3))
    
    def reset(self):
        """Reset wind model"""
        if self.wind_enabled and self.turbulence_model is not None:
            self.turbulence_model.reset()
        
        # Reset output arrays
        self.total_linear_gusts.zero_()
        self.total_angular_gusts.zero_()
        self.constant_wind_body.zero_()
        self.turbulent_linear_gusts.zero_()
        self.turbulent_angular_gusts.zero_()
    
    def seed(self, seed=None):
        """Seed the wind model random number generators"""
        if seed is not None:
            self.base_seed = seed
            
            # Seed turbulence model if it exists
            if hasattr(self, 'turbulence_model') and self.turbulence_model is not None:
                self.turbulence_model.seed(seed + 100)

        else:
            self.base_seed = None
            if hasattr(self, 'turbulence_model') and self.turbulence_model is not None:
                self.turbulence_model.seed()

@wp.kernel
def extract_altitude_kernel(
    positions: wp.array(dtype=wp.vec3f),
    altitudes: wp.array(dtype=wp.float32)
):
    """Extract altitude from position array (convert NED to positive up)"""
    tid = wp.tid()
    altitudes[tid] = -positions[tid][2]  # Convert NED z to positive up altitude