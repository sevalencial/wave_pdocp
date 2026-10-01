"""Reconstruct the optimal control and terminal state in continuous time."""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from .solution import FOMSolution

Array = NDArray[np.float64]


def reconstruct_control(
    solution: FOMSolution, n_time: int = 200
) -> tuple[Array, Array]:
    """Sample u(t) = -B† exp((T-t) A†) P_T only for visualization.

    n_time does not change the FOM or its Gramian: no time-stepping is used.
    """
    if not isinstance(n_time, (int, np.integer)) or n_time < 2:
        raise ValueError("n_time debe ser un entero mayor o igual a 2")
    t = np.linspace(0.0, solution.final_time, n_time)
    control = np.empty((n_time, solution.control_adjoint.shape[0]))
    for i, ti in enumerate(t):
        adjoint = expm((solution.final_time - ti) * solution.state_adjoint)
        control[i] = -solution.control_adjoint @ (adjoint @ solution.final_adjoint)
    return t, control


def reconstruct_final_state(solution: FOMSolution) -> Array:
    """Y*(T) = S(T) Y0 - W(T) P_T, using the fields of the actual solve."""
    fields = solution.fields
    initial = np.concatenate([fields.initial_displacement, fields.initial_velocity])
    return solution.semigroup_T @ initial - solution.gramian @ solution.final_adjoint
