from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

SpatialCoefficient = (
    float | Callable[[NDArray[np.float64]], NDArray[np.float64] | float]
)


@dataclass(frozen=True)
class WaveParameter:
    """Coefficients may be constants or vectorized functions of spatial position.

    A callable receives a 1-D array of x coordinates and returns an array of
    the same shape, or a scalar for a constant field. Negative damping is
    allowed deliberately: it describes antidamping, not dissipation.
    """

    wave_speed: SpatialCoefficient = 1.0
    damping: SpatialCoefficient = 0.0

    control_center: float = 0.5
    control_length: float = 0.5

    target_amplitude: float = 1.0
    target_mode: int = 1
