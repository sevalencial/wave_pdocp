import itertools

import numpy as np
from scipy.linalg import cholesky

from ..discretization.inner_products import build_Q_h_matrix
from ..domain.grid import SpatialGrid


def compare_adjoint_meshes(
    coarse: np.ndarray,
    coarse_grid: SpatialGrid,
    fine: np.ndarray,
    fine_grid: SpatialGrid,
    safe_epsilon=1e-15,
) -> float:
    """Linear interpolation with homogeneous endpoint values, norm on fine H_h.

    Both components use the homogeneous boundary convention of the smooth
    Dirichlet-wave snapshots in this campaign. Grids need not be nested.
    """
    if coarse_grid.length != fine_grid.length or coarse_grid.n >= fine_grid.n:
        raise ValueError("Se requiere el mismo dominio y una malla más fina")
    xp = np.r_[0.0, coarse_grid.interior_points, coarse_grid.length]
    interpolated = np.concatenate(
        [
            np.interp(fine_grid.interior_points, xp, np.r_[0.0, component, 0.0])
            for component in np.split(coarse, 2)
        ]
    )
    C = cholesky(build_Q_h_matrix(fine_grid))
    return float(
        np.linalg.norm(C @ (fine - interpolated))
        / (np.linalg.norm(C @ fine) + safe_epsilon)
    )


def errors_decrease(
    errors: list[float], factor_tolerance=1.05, absolute_floor=1e-8
) -> bool:
    if len(errors) < 2 or not np.all(np.isfinite(errors)):
        return False
    return all(
        b <= max(absolute_floor, factor_tolerance * a)
        for a, b in itertools.pairwise(errors)
    )
