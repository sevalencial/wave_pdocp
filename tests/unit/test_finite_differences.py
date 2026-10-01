import numpy as np
import pytest

from wave_pdocp.discretization.finite_differences import build_diffusion_matrix
from wave_pdocp.domain.grid import SpatialGrid


@pytest.mark.parametrize("n", [1, 7, 24])
def test_variable_flux_matches_manufactured_solution_including_boundary_nodes(n):
    # a(x)=1+x, y(x)=x(1-x): (a y')' = -1-4x exactly for this flux stencil.
    grid = SpatialGrid(1.0, n)
    x = grid.interior_points
    L = build_diffusion_matrix(np.sqrt(1.0 + grid.half_points), grid)
    np.testing.assert_allclose(L @ (x * (1.0 - x)), -1.0 - 4.0 * x, atol=1e-12)
    np.testing.assert_array_equal(L, L.T)
    assert np.max(np.linalg.eigvalsh(L)) < 0


def test_constant_speed_recovers_dirichlet_laplacian():
    grid = SpatialGrid(2.0, 6)
    L = build_diffusion_matrix(np.full(grid.n + 1, 1.7), grid)
    expected = np.diag(np.full(grid.n, -2.0))
    expected += np.diag(np.ones(grid.n - 1), 1) + np.diag(np.ones(grid.n - 1), -1)
    np.testing.assert_allclose(L, 1.7**2 / grid.h**2 * expected)


def test_nodal_values_cannot_be_mistaken_for_half_grid_values():
    grid = SpatialGrid(1.0, 6)
    with pytest.raises(ValueError, match="N\\+1"):
        build_diffusion_matrix(np.ones(grid.n), grid)
