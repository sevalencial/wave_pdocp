import numpy as np
from scipy.linalg import solve

from ..discretization.finite_differences import (
    build_control_matrix,
    build_damping_matrix,
    build_diffusion_matrix,
)
from ..discretization.inner_products import build_Q_h_matrix, build_R_h_matrix
from ..domain.grid import SpatialGrid
from ..domain.parameterization import evaluate_parameter
from ..domain.parameters import WaveParameter
from ..operators.adjoints import A_adjoint, B_adjoint
from ..operators.assembly import build_control_operator, build_state_operator
from ..operators.gramian import compute_gramian
from ..operators.semigroup import compute_semigroup
from .solution import FOMSolution


class FOMSolver:
    def __init__(self, grid: SpatialGrid, final_time: float, beta: float) -> None:
        if final_time <= 0:
            raise ValueError("tiempo final debe ser positivo")

        if beta <= 0:
            raise ValueError("beta debe ser positivo")

        self.grid = grid
        self.final_time = final_time
        self.beta = beta

    def solve(self, mu: WaveParameter) -> FOMSolution:
        fields = evaluate_parameter(mu, self.grid)

        L = build_diffusion_matrix(fields.wave_speed_half, self.grid)
        D = build_damping_matrix(fields.damping, self.grid)
        C = build_control_matrix(fields.control_mask, self.grid)

        A = build_state_operator(L, D)
        B = build_control_operator(C)

        # productos internos
        Q_h = build_Q_h_matrix(self.grid)
        R_h = build_R_h_matrix(self.grid)

        # adjuntas
        A_adj = A_adjoint(A, Q_h)
        B_adj = B_adjoint(B, Q_h, R_h)

        S_T = compute_semigroup(A, self.final_time)

        W = compute_gramian(A, B, A_adj, B_adj, self.final_time)

        # Condiciones
        Y0 = np.concatenate([fields.initial_displacement, fields.initial_velocity])

        YT = np.concatenate([fields.target_displacement, fields.target_velocity])

        rhs = self.beta * (S_T @ Y0 - YT)

        # solucion del sistema

        n_state = 2 * self.grid.n

        system = np.eye(n_state) + self.beta * W
        P_T = solve(system, rhs)
        residual = system @ P_T - rhs

        residual_norm = float(np.sqrt(residual.T @ Q_h @ residual))

        return FOMSolution(
            grid=self.grid,
            fields=fields,
            final_time=self.final_time,
            beta=self.beta,
            final_adjoint=P_T,
            rhs=rhs,
            gramian=W,
            state_operator=A,
            control_operator=B,
            state_adjoint=A_adj,
            control_adjoint=B_adj,
            semigroup_T=S_T,
            residual=residual,
            weighted_residual_norm=residual_norm,
        )
