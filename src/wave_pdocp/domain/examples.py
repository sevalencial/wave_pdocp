"""Physical fields for the single-parameter validation experiment."""

import numpy as np

from .parameters import WaveParameter


def variable_coefficient_parameter() -> WaveParameter:
    """One smooth, dissipative case on (0, 1), with distributed global control."""
    return WaveParameter(
        wave_speed=lambda x: 1.0 + 0.2 * np.sin(2.0 * np.pi * x),
        damping=lambda x: 0.1 + 0.05 * np.cos(2.0 * np.pi * x),
        control_center=0.5,
        control_length=1.0,
        target_amplitude=1.0,
        target_mode=1,
    )
