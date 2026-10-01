from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..domain.grid import SpatialGrid
from ..domain.parameterization import ParameterFields

Array = NDArray[np.float64]


@dataclass(frozen=True)
class FOMSolution:
    grid: SpatialGrid
    fields: ParameterFields
    final_time: float
    beta: float
    final_adjoint: Array
    rhs: Array
    gramian: Array
    state_operator: Array
    control_operator: Array
    state_adjoint: Array
    control_adjoint: Array
    semigroup_T: Array
    residual: Array
    weighted_residual_norm: float
