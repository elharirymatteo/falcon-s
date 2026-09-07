import numpy as np
from .dryden import DrydenTurbulenceModel
from .quaternion_math import quat_rotate_vector

class WindModelMixin:
    """
    Mixin to manage all environmental effects, including constant wind and turbulence.
    This class should be inherited by the StatePropagationMixin.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turbulence_model = None
        self.linear_gusts = np.zeros(3)
        self.angular_gusts = np.zeros(3)

        # Initialize from configuration file
        env_config = self.config.get('environment_params', {})
        self.constant_wind_inertial = np.array(env_config.get('constant_wind', [0.0, 0.0, 0.0]))

        # --- Turbulence Setup ---
        turbulence_config = env_config.get('turbulence', {})
        if turbulence_config.get('enable', False):
            model_type = turbulence_config.get('model', 'dryden')
            if model_type == 'dryden':
                # Get initial conditions needed by the turbulence model
                wingspan = self.VP['wing']['span']

                self.turbulence_model = DrydenTurbulenceModel(
                    dt=self.get_time_step(),
                    b=wingspan,
                    intensity=turbulence_config.get('intensity', 'light')
                )

    def _get_constant_wind_body(self) -> np.ndarray:
        """
        Returns the constant wind vector from the inertial frame (NED) to the body frame.
        """
        return quat_rotate_vector(self.state['orientation'], self.constant_wind_inertial, i_to_b=True)

    def update_wind(self):
        """Updates the total wind gust vectors for the current time step."""
        turbulent_linear = np.zeros(3)
        turbulent_angular = np.zeros(3)

        # 1. Get turbulent gusts if the model is enabled
        if self.turbulence_model:
            # The turbulence model's update depends on the current airspeed (Va) and altitude (h)
            # Va is calculated in the AtmosphereStateMixin
            turbulent_linear, turbulent_angular = self.turbulence_model.update(self.Va, -self.state['position'][2])

        # 2. Get constant wind component
        constant_wind_body = self._get_constant_wind_body()
        
        # 3. Combine the wind components
        self.linear_gusts = turbulent_linear + constant_wind_body
        self.angular_gusts = turbulent_angular # Constant wind does not create angular gusts