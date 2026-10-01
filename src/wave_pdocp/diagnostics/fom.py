from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky, eigvals, eigvalsh, solve_triangular

from ..discretization.inner_products import build_Q_h_matrix
from ..fom.reconstruction import reconstruct_final_state
from ..fom.solution import FOMSolution


@dataclass(frozen=True)
class FOMDiagnostics:
    weighted_residual_norm: float
    relative_residual_norm: float
    terminal_error_norm: float
    relative_terminal_error: float
    displacement_max_error: float
    velocity_max_error: float
    terminal_optimality_error_norm: float
    control_energy: float
    objective_value: float
    uncontrolled_objective_value: float
    gramian_relative_asymmetry: float
    gramian_min_eigenvalue: float
    gramian_condition_number: float
    system_condition_number: float
    state_max_real_eigenvalue: float
    terminal_optimality_relative_error: float
    gramian_weighted_symmetry_error: float
    gramian_max_eigenvalue: float
    state_spectral_radius: float
    adjoint_norm: float
    cost_reduction: float


def _relative(value: float, reference: float, safe_epsilon: float = 1e-15) -> float:
    return value / (reference + safe_epsilon)


def compute_fom_diagnostics(
    solution: FOMSolution, safe_epsilon: float = 1e-15
) -> FOMDiagnostics:
    """Use the fixed H_h metric, including for spectra and conditioning.

    Gramian eigenvalues refer to the symmetric part of C W C^{-1}; its
    asymmetry is reported separately, never silently corrected in the solve.
    Control energy is P_T^T Q W P_T, without temporal quadrature.
    """
    fields = solution.fields
    Q = build_Q_h_matrix(solution.grid)
    C = cholesky(Q)  # Q = C.T @ C
    target = np.concatenate([fields.target_displacement, fields.target_velocity])
    initial = np.concatenate([fields.initial_displacement, fields.initial_velocity])
    error = reconstruct_final_state(solution) - target
    terminal_error = float(np.linalg.norm(C @ error))
    residual_vector = (
        np.eye(Q.shape[0]) + solution.beta * solution.gramian
    ) @ solution.final_adjoint - solution.rhs
    residual = float(np.linalg.norm(C @ residual_vector))
    optimality_error = solution.final_adjoint - solution.beta * error
    control_energy = float(
        solution.final_adjoint @ Q @ solution.gramian @ solution.final_adjoint
    )
    free_error = solution.semigroup_T @ initial - target

    # Right solve for C W C^{-1}; W itself need not be Euclidean-symmetric.
    W_e = solve_triangular(C.T, (C @ solution.gramian).T, lower=True).T
    W_sym = (W_e + W_e.T) / 2.0
    system_e = np.eye(W_e.shape[0]) + solution.beta * W_e
    eigenvalues_W = eigvalsh(W_sym)
    eigenvalues_A = eigvals(solution.state_operator)
    adjoint_norm = float(np.linalg.norm(C @ solution.final_adjoint))
    optimality_norm = float(np.linalg.norm(C @ optimality_error))
    objective = 0.5 * control_energy + 0.5 * solution.beta * terminal_error**2
    zero_cost = 0.5 * solution.beta * float(np.linalg.norm(C @ free_error)) ** 2
    QW = Q @ solution.gramian
    return FOMDiagnostics(
        weighted_residual_norm=residual,
        relative_residual_norm=_relative(
            residual, float(np.linalg.norm(C @ solution.rhs)), safe_epsilon
        ),
        terminal_error_norm=terminal_error,
        relative_terminal_error=_relative(
            terminal_error, float(np.linalg.norm(C @ target)), safe_epsilon
        ),
        displacement_max_error=float(np.max(np.abs(error[: solution.grid.n]))),
        velocity_max_error=float(np.max(np.abs(error[solution.grid.n :]))),
        terminal_optimality_error_norm=optimality_norm,
        control_energy=control_energy,
        objective_value=objective,
        uncontrolled_objective_value=zero_cost,
        gramian_relative_asymmetry=_relative(
            float(np.linalg.norm(W_e - W_e.T)), float(np.linalg.norm(W_e)), safe_epsilon
        ),
        gramian_min_eigenvalue=float(eigenvalues_W[0]),
        gramian_condition_number=float(np.linalg.cond(W_e)),
        system_condition_number=float(np.linalg.cond(system_e)),
        state_max_real_eigenvalue=float(np.max(eigenvalues_A.real)),
        terminal_optimality_relative_error=_relative(
            optimality_norm,
            max(adjoint_norm, solution.beta * terminal_error),
            safe_epsilon,
        ),
        gramian_weighted_symmetry_error=_relative(
            float(np.linalg.norm(QW - QW.T)), float(np.linalg.norm(QW)), safe_epsilon
        ),
        gramian_max_eigenvalue=float(eigenvalues_W[-1]),
        state_spectral_radius=float(np.max(np.abs(eigenvalues_A))),
        adjoint_norm=adjoint_norm,
        cost_reduction=(zero_cost - objective) / (zero_cost + safe_epsilon),
    )
