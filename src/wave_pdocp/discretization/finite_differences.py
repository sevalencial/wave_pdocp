import numpy as np
from numpy.typing import NDArray

from ..domain.grid import SpatialGrid

Array = NDArray[np.float64]


def build_diffusion_matrix(wave_speed_half: Array, grid: SpatialGrid) -> Array:
    """Conservative Dirichlet operator using c at the N+1 half-grid points."""
    n = grid.n
    h = grid.h

    if wave_speed_half.shape != (n + 1,):
        raise ValueError("wave_speed_half debe contener N+1 valores en semipuntos")

    a_half = wave_speed_half**2
    L = np.zeros((n, n))

    for k in range(n):
        a_left = a_half[k]
        a_right = a_half[k + 1]
        L[k, k] = -(a_left + a_right) / h**2

        if k > 0:
            L[k, k - 1] = a_left / h**2

        if k < n - 1:
            L[k, k + 1] = a_right / h**2
    return L


def build_damping_matrix(damping: Array, grid: SpatialGrid) -> Array:
    if damping.shape != (grid.n,):
        raise ValueError("damping debe contener N valores nodales")
    return np.diag(damping)


def build_control_matrix(control: Array, grid: SpatialGrid) -> Array:
    if control.shape != (grid.n,):
        raise ValueError("control debe contener N valores nodales")
    return np.diag(control)
