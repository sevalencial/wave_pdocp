import numpy as np
import pytest

from wave_pdocp.discretization.finite_differences import (
    build_damping_matrix,
    build_diffusion_matrix,
)
from wave_pdocp.discretization.inner_products import (
    build_Q_h_matrix,
    build_R_h_matrix,
)
from wave_pdocp.domain.examples import variable_coefficient_parameter
from wave_pdocp.domain.grid import SpatialGrid
from wave_pdocp.domain.parameterization import evaluate_parameter
from wave_pdocp.domain.parameters import WaveParameter
from wave_pdocp.fom.solver import FOMSolver
from wave_pdocp.operators.adjoints import (
    A_adjoint,
)
from wave_pdocp.operators.assembly import (
    build_state_operator,
)


def test_weighted_state_adjoint():

    grid = SpatialGrid(
        length=1.0,
        n_interior=20,
    )

    mu = WaveParameter()

    fields = evaluate_parameter(
        mu,
        grid,
    )

    L = build_diffusion_matrix(
        fields.wave_speed_half,
        grid,
    )

    D = build_damping_matrix(
        fields.damping,
        grid,
    )

    A = build_state_operator(
        L,
        D,
    )

    Q = build_Q_h_matrix(
        grid,
    )

    A_adj = A_adjoint(
        A,
        Q,
    )

    rng = np.random.default_rng(42)

    Y = rng.standard_normal(2 * grid.n)
    Z = rng.standard_normal(2 * grid.n)

    lhs = (A @ Y).T @ Q @ Z
    rhs = Y.T @ Q @ (A_adj @ Z)

    assert np.allclose(
        lhs,
        rhs,
        rtol=1e-10,
        atol=1e-10,
    )


def test_conservative_full_control_adjoint_identities():
    grid = SpatialGrid(1.0, 15)
    solution = FOMSolver(grid, final_time=0.7, beta=1.0).solve(
        WaveParameter(control_length=1.0)
    )
    np.testing.assert_allclose(
        solution.state_adjoint, -solution.state_operator, atol=1e-11
    )
    expected = np.c_[np.zeros((grid.n, grid.n)), np.eye(grid.n)]
    np.testing.assert_allclose(solution.control_adjoint, expected, atol=1e-12)


@pytest.mark.parametrize("control_length", [0.5, 1.0])
def test_variable_coefficient_adjoint_duality_and_dissipation(control_length):
    from dataclasses import replace

    grid = SpatialGrid(1.0, 15)
    mu = replace(variable_coefficient_parameter(), control_length=control_length)
    solution = FOMSolver(grid, final_time=0.7, beta=3.0).solve(mu)
    A, B = solution.state_operator, solution.control_operator
    Q = build_Q_h_matrix(grid)
    R = build_R_h_matrix(grid)
    rng = np.random.default_rng(19)
    y, z = rng.normal(size=(2, 2 * grid.n))
    u = rng.normal(size=grid.n)
    np.testing.assert_allclose(
        (A @ y) @ Q @ z, y @ Q @ solution.state_adjoint @ z, rtol=1e-11
    )
    np.testing.assert_allclose(
        (B @ u) @ Q @ z, u @ R @ solution.control_adjoint @ z, rtol=1e-11
    )

    # Physical energy uses c(x); the optimization metric Q remains independent of c.
    H = np.zeros_like(Q)
    H[: grid.n, : grid.n] = -grid.h * A[grid.n :, : grid.n]
    H[grid.n :, grid.n :] = grid.h * np.eye(grid.n)
    expected = np.zeros_like(Q)
    expected[grid.n :, grid.n :] = -2 * grid.h * np.diag(solution.fields.damping)
    np.testing.assert_allclose(A.T @ H + H @ A, expected, atol=1e-12)
