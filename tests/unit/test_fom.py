from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from wave_pdocp.diagnostics.fom import compute_fom_diagnostics
from wave_pdocp.domain.examples import variable_coefficient_parameter
from wave_pdocp.domain.grid import SpatialGrid
from wave_pdocp.domain.parameterization import evaluate_parameter
from wave_pdocp.domain.parameters import WaveParameter
from wave_pdocp.fom.reconstruction import reconstruct_control, reconstruct_final_state
from wave_pdocp.fom.solver import FOMSolver


def test_single_mu_fom():

    grid = SpatialGrid(
        length=1.0,
        n_interior=20,
    )

    mu = WaveParameter()

    solver = FOMSolver(
        grid=grid,
        final_time=2.0,
        beta=1.0,
    )

    solution = solver.solve(mu)

    assert solution.final_adjoint.shape == (2 * grid.n,)

    assert solution.gramian.shape == (
        2 * grid.n,
        2 * grid.n,
    )

    assert solution.weighted_residual_norm < 1e-9


@pytest.mark.parametrize("beta", [0.1, 1.0, 10.0, 100.0])
@pytest.mark.parametrize("initial_amplitudes", [(0.0, 0.0), (0.3, -0.1)])
def test_fom_and_control_match_exact_single_mode_solution(
    beta, initial_amplitudes, monkeypatch
):
    # Independent closed-form Gramian of a single discrete sine oscillator.
    # Nonzero initial data also checks that only Y0 is propagated, never YT.
    grid = SpatialGrid(1.0, 12)
    mode, amplitude, T = 3, 1.7, 0.73
    shape = np.sin(mode * np.pi * grid.interior_points)
    omega = 2 / grid.h * np.sin(mode * np.pi / (2 * (grid.n + 1)))
    mu = WaveParameter(control_length=1.0, target_amplitude=amplitude, target_mode=mode)
    fields = replace(
        evaluate_parameter(mu, grid),
        initial_displacement=initial_amplitudes[0] * shape,
        initial_velocity=initial_amplitudes[1] * shape,
    )
    monkeypatch.setattr(
        "wave_pdocp.fom.solver.evaluate_parameter", lambda mu, grid: fields
    )
    solution = FOMSolver(grid, T, beta).solve(mu)

    sine, cosine = np.sin(omega * T), np.cos(omega * T)
    S_mode = np.array([[cosine, sine / omega], [-omega * sine, cosine]])
    W_mode = np.array(
        [
            [T / 2 - np.sin(2 * omega * T) / (4 * omega), sine**2 / (2 * omega**2)],
            [sine**2 / 2, T / 2 + np.sin(2 * omega * T) / (4 * omega)],
        ]
    )
    free = S_mode @ np.array(initial_amplitudes)
    p_mode = np.linalg.solve(
        np.eye(2) + beta * W_mode, beta * (free - [amplitude, 0.0])
    )
    final_mode = free - W_mode @ p_mode
    np.testing.assert_allclose(
        solution.final_adjoint, np.kron(p_mode, shape), atol=2e-10
    )
    np.testing.assert_allclose(
        reconstruct_final_state(solution), np.kron(final_mode, shape), atol=2e-10
    )

    t, control = reconstruct_control(solution, n_time=7)
    expected_u = -(
        omega * np.sin(omega * (T - t)) * p_mode[0]
        + np.cos(omega * (T - t)) * p_mode[1]
    )
    np.testing.assert_allclose(control, expected_u[:, None] * shape, atol=2e-10)


def test_variable_coefficient_fom_optimality_and_gramian_identity():
    solution = FOMSolver(SpatialGrid(1.0, 20), final_time=2.0, beta=100.0).solve(
        variable_coefficient_parameter()
    )
    diag = compute_fom_diagnostics(solution)
    assert diag.relative_residual_norm < 1e-10
    assert diag.terminal_optimality_error_norm < 1e-9
    assert diag.gramian_relative_asymmetry < 1e-9
    assert diag.gramian_min_eigenvalue > 0
    assert diag.system_condition_number < 10
    assert diag.state_max_real_eigenvalue < 0
    assert 0 < diag.control_energy
    assert 0 < diag.objective_value < diag.uncontrolled_objective_value
    assert 0 < diag.relative_terminal_error < 0.02

    A, Ad, W = solution.state_operator, solution.state_adjoint, solution.gramian
    G = solution.control_operator @ solution.control_adjoint
    expected = solution.semigroup_T @ G @ expm(solution.final_time * Ad) - G
    np.testing.assert_allclose(A @ W + W @ Ad, expected, atol=1e-8)


def test_zero_target_has_zero_control_and_well_defined_relative_diagnostics():
    solution = FOMSolver(SpatialGrid(1.0, 8), 2.0, 10.0).solve(
        WaveParameter(target_amplitude=0.0)
    )
    diag = compute_fom_diagnostics(solution)
    np.testing.assert_array_equal(solution.final_adjoint, 0)
    assert diag.relative_residual_norm == 0
    assert diag.relative_terminal_error == 0
    assert diag.control_energy == diag.objective_value == 0


def test_reconstructed_control_respects_local_actuator_support():
    solution = FOMSolver(SpatialGrid(1.0, 15), 1.0, 3.0).solve(
        WaveParameter(control_length=0.5)
    )
    _, control = reconstruct_control(solution, n_time=5)
    np.testing.assert_array_equal(control[:, solution.fields.control_mask == 0], 0)
