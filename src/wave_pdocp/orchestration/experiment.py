from ..diagnostics.fom import FOMDiagnostics, compute_fom_diagnostics
from ..domain.examples import variable_coefficient_parameter
from ..domain.grid import SpatialGrid
from ..domain.parameters import WaveParameter
from ..fom.solution import FOMSolution
from ..fom.solver import FOMSolver


class SingleMuExperiment:
    def __init__(
        self,
        grid: SpatialGrid,
        solver: FOMSolver,
        parameter: WaveParameter,
    ) -> None:

        self.grid = grid
        self.solver = solver
        self.parameter = parameter
        self.diagnostics: FOMDiagnostics | None = None

    @classmethod
    def variable_coefficient_example(cls) -> "SingleMuExperiment":
        """Shared configuration for both executable scripts: one fixed mu."""
        grid = SpatialGrid(length=1.0, n_interior=100)
        solver = FOMSolver(grid=grid, final_time=2.0, beta=100.0)
        return cls(grid, solver, variable_coefficient_parameter())

    def run(self) -> FOMSolution:

        solution = self.solver.solve(self.parameter)

        self.diagnostics = compute_fom_diagnostics(solution)
        self._report(solution, self.diagnostics)

        return solution

    def _report(self, solution: FOMSolution, diagnostics: FOMDiagnostics) -> None:

        N = self.grid.n

        print("=" * 60)
        print("Validacion del FOM para un solo mu")
        print("=" * 60)

        print(f"N: {N}")
        print(f"T: {solution.final_time}, beta: {solution.beta}")

        print(
            "A shape: ",
            solution.state_operator.shape,
        )

        print(
            "B shape: ",
            solution.control_operator.shape,
        )

        print(
            "W shape: ",
            solution.gramian.shape,
        )

        print(
            "P_T shape: ",
            solution.final_adjoint.shape,
        )

        print(
            "Residual H_h norm :",
            solution.weighted_residual_norm,
        )
        print(f"Residual relativo H_h: {diagnostics.relative_residual_norm:.3e}")
        print(f"Error terminal H_h: {diagnostics.terminal_error_norm:.6e}")
        print(f"Error terminal relativo H_h: {diagnostics.relative_terminal_error:.3%}")
        print(f"Error max desplazamiento: {diagnostics.displacement_max_error:.6e}")
        print(f"Error max velocidad: {diagnostics.velocity_max_error:.6e}")
        print(
            f"Error P_T = beta (Y(T)-Y_T): {diagnostics.terminal_optimality_error_norm:.3e}"
        )
        print(f"Energia del control: {diagnostics.control_energy:.6e}")
        print(
            f"J(u*): {diagnostics.objective_value:.6e}; J(0): {diagnostics.uncontrolled_objective_value:.6e}"
        )
        print(
            f"Asimetria relativa de W en H_h: {diagnostics.gramian_relative_asymmetry:.3e}"
        )
        print(
            f"Autovalor minimo de W (parte simetrica en H_h): {diagnostics.gramian_min_eigenvalue:.6e}"
        )
        print(f"cond_H_h(W): {diagnostics.gramian_condition_number:.6e}")
        print(f"cond_H_h(I + beta W): {diagnostics.system_condition_number:.6e}")
        print(f"max Re(lambda(A)): {diagnostics.state_max_real_eigenvalue:.6e}")
