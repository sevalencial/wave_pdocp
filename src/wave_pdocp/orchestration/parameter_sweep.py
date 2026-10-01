import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..data.generator import snapshot_from_solution, stack_snapshots
from ..data.schema import METRIC_COLUMNS
from ..diagnostics.fom_validation import ValidationResult, validate_fom
from ..diagnostics.parametric import (
    add_local_diagnostics,
    select_representative_samples,
    summarize_campaign,
)
from ..diagnostics.refinement import compare_adjoint_meshes, errors_decrease
from ..domain.grid import SpatialGrid
from ..domain.parameter_space import ParameterSpace
from ..domain.parameterization import WaveSpeedAmplitudeFamily
from ..fom.solver import FOMSolver
from ..io.config import SweepConfig
from ..io.dataset_io import build_validated_dataset, write_dataset
from ..io.metrics_io import campaign_metadata, save_json, save_table


@dataclass(frozen=True)
class ParameterSweepResult:
    parameters: np.ndarray
    statuses: list[str]
    diagnostics: pd.DataFrame
    refinement: pd.DataFrame
    dataset_path: Path
    metrics_path: Path
    summary_path: Path
    snapshots_path: Path
    summary: dict


def evaluate_case(solver, parameter, validation):
    started = perf_counter()
    solution, caught = None, []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solution = solver.solve(parameter)
            result = validate_fom(solution, validation)
    except Exception as exc:  # noqa: BLE001 -- record a failed sample and continue the campaign
        result = ValidationResult("FAIL", [f"{type(exc).__name__}: {exc}"])
    elapsed = perf_counter() - started
    for warning in caught:
        result.reasons.append(f"{warning.category.__name__}: {warning.message}")
        if result.status == "PASS":
            result.status = "WARNING"
    if elapsed > validation.runtime_warning_seconds:
        result.reasons.append("runtime_above_warning_threshold")
        if result.status == "PASS":
            result.status = "WARNING"
    return solution, result, elapsed


class ParameterSweepExperiment:
    """Coordinate solves/diagnostics/refinement; mathematics stays in its layers."""

    def __init__(self, config: SweepConfig, root: Path):
        self.config, self.root = config, root
        p = config.parameter_space
        self.space = ParameterSpace((p.min,), (p.max,))
        self.family = WaveSpeedAmplitudeFamily(self.space, config.domain.length)
        self.grid = SpatialGrid(config.domain.length, config.discretization.n_interior)
        self.solver = FOMSolver(
            self.grid, config.domain.final_time, config.optimal_control.beta
        )
        self.paths = {
            name: root / value for name, value in config.outputs.model_dump().items()
        }
        if len({path.resolve() for path in self.paths.values()}) != len(self.paths):
            raise ValueError("Las rutas de salida deben ser diferentes")

    def run(self, overwrite=False, make_plots=True) -> ParameterSweepResult:
        for path in self.paths.values():
            if path.exists() and not overwrite:
                raise FileExistsError(
                    f"Ya existe {path}; no se regenerarán datos/splits sin --overwrite"
                )
        started = perf_counter()
        parameters = self.space.sample_uniform_grid(
            self.config.parameter_space.n_samples
        )
        rows, snapshots = [], {}
        for i, mu in enumerate(parameters):
            solution, result, elapsed = evaluate_case(
                self.solver, self.family(mu), self.config.validation
            )
            row = dict(
                sample_id=i,
                mu=float(mu[0]),
                status=result.status,
                reasons="; ".join(result.reasons),
                runtime_seconds=elapsed,
                **result.metrics,
            )
            rows.append(row)
            if solution is not None:
                snapshot = snapshot_from_solution(solution)
                if all(np.all(np.isfinite(v)) for v in snapshot.values()):
                    snapshots[i] = snapshot
            print(
                f"[{i + 1:03d}/{len(parameters)}] mu={mu[0]:.6f} | {result.status} | r_rel={result.metrics.get('relative_residual', float('nan')):.2e} | e_T={result.metrics.get('terminal_error', float('nan')):.3e} | kappa={result.metrics.get('system_condition_number', float('nan')):.3e} | time={elapsed:.3f}s",
                flush=True,
            )
            if result.reasons:
                print("  " + "; ".join(result.reasons), flush=True)
        table = pd.DataFrame(rows).reindex(
            columns=[
                "sample_id",
                "mu",
                "status",
                "reasons",
                "runtime_seconds",
                *METRIC_COLUMNS,
            ]
        )
        add_local_diagnostics(table, snapshots, self.grid, self.config.validation)
        selected = select_representative_samples(
            table, self.config.mesh_refinement.representative_samples
        )
        refinement = (
            self._refine(selected, table, snapshots, parameters)
            if self.config.mesh_refinement.enabled
            else pd.DataFrame()
        )
        samples = stack_snapshots(snapshots, table, parameters, self.grid)
        metadata = campaign_metadata(self.config, self.root)
        metadata["representative_sample_ids"] = selected
        metadata["mesh_refinement_performed"] = not refinement.empty
        metadata["mesh_all_converged"] = (
            bool(refinement.convergence_decreases.all())
            if not refinement.empty
            else None
        )
        save_table(table, self.paths["metrics"])
        save_table(refinement, self.paths["refinement"])
        write_dataset(self.paths["snapshots"], samples, metadata, overwrite=overwrite)
        split_counts = build_validated_dataset(
            self.paths["snapshots"], self.paths["dataset"], overwrite=overwrite
        )
        if make_plots:
            from ..visualization.parametric_plots import plot_campaign

            plot_campaign(
                table, samples, self.grid, refinement, selected, self.paths["figures"]
            )
        summary = summarize_campaign(table, refinement, perf_counter() - started)
        summary.update(
            split_counts=split_counts,
            config=metadata["config"],
            dataset_path=str(self.paths["dataset"]),
        )
        save_json(summary, self.paths["summary"])
        return ParameterSweepResult(
            parameters,
            table.status.tolist(),
            table,
            refinement,
            self.paths["dataset"],
            self.paths["metrics"],
            self.paths["summary"],
            self.paths["snapshots"],
            summary,
        )

    def _refine(self, selected, table, snapshots, parameters):
        config = self.config.mesh_refinement
        rows = []
        for i in selected:
            previous = None
            errors, sample_rows = [], []
            for n in config.sizes:
                grid = SpatialGrid(self.grid.length, n)
                if n == self.grid.n:
                    snapshot = snapshots[i]
                    result = ValidationResult(
                        table.at[i, "status"],
                        [],
                        {key: table.at[i, key] for key in METRIC_COLUMNS},
                    )
                    elapsed = 0.0
                else:
                    solver = FOMSolver(grid, self.solver.final_time, self.solver.beta)
                    solution, result, elapsed = evaluate_case(
                        solver, self.family(parameters[i]), self.config.validation
                    )
                    snapshot = (
                        snapshot_from_solution(solution)
                        if solution is not None and result.status != "FAIL"
                        else None
                    )
                row = dict(
                    sample_id=i,
                    mu=float(parameters[i, 0]),
                    n_interior=n,
                    status=result.status,
                    reasons="; ".join(result.reasons),
                    runtime_seconds=elapsed,
                    reused_baseline=n == self.grid.n,
                    **result.metrics,
                )
                row.update(previous_n=None, relative_adjoint_difference=None)
                if snapshot is not None and previous is not None:
                    coarse, coarse_grid = previous
                    error = compare_adjoint_meshes(
                        coarse,
                        coarse_grid,
                        snapshot["final_adjoint"],
                        grid,
                        self.config.validation.safe_epsilon,
                    )
                    row.update(
                        previous_n=coarse_grid.n, relative_adjoint_difference=error
                    )
                    errors.append(error)
                previous = (
                    (snapshot["final_adjoint"], grid) if snapshot is not None else None
                )
                sample_rows.append(row)
                print(
                    f"[mesh] sample={i:03d} mu={parameters[i, 0]:.6f} N={n} | {result.status} | error={row['relative_adjoint_difference']}"
                    + (" | " + row["reasons"] if row["reasons"] else ""),
                    flush=True,
                )
            converged = len(errors) == len(config.sizes) - 1 and errors_decrease(
                errors, config.decrease_factor_tolerance, config.absolute_error_floor
            )
            all_pass = all(row["status"] == "PASS" for row in sample_rows)
            if not converged or not all_pass:
                if table.at[i, "status"] == "PASS":
                    table.at[i, "status"] = "WARNING"
                reason = (
                    "mesh_errors_not_decreasing"
                    if not converged
                    else "mesh_validation_warning_or_failure"
                )
                table.at[i, "reasons"] = (table.at[i, "reasons"] + "; " + reason).strip(
                    "; "
                )
            for row in sample_rows:
                row["convergence_decreases"] = converged
            rows.extend(sample_rows)
        return pd.DataFrame(rows)
