import numpy as np
import pandas as pd
from scipy.linalg import cholesky

from ..discretization.inner_products import build_Q_h_matrix
from ..domain.grid import SpatialGrid
from .fom_validation import ValidationConfig


def add_local_diagnostics(
    table: pd.DataFrame, snapshots: dict, grid: SpatialGrid, config: ValidationConfig
) -> None:
    """Compare consecutive grid points only; never bridge a failed sample.

    The two interval metrics are attached to the right-hand sample. The first
    row and intervals touching failures are missing, not artificial zeros.
    """
    C = cholesky(build_Q_h_matrix(grid))
    sensitivity = np.full(len(table), np.nan)
    change = sensitivity.copy()
    for j in range(1, len(table)):
        left, right = table.iloc[j - 1], table.iloc[j]
        a, b = int(left.sample_id), int(right.sample_id)
        if (
            a not in snapshots
            or b not in snapshots
            or "FAIL" in (left.status, right.status)
        ):
            continue
        p, q = snapshots[a]["final_adjoint"], snapshots[b]["final_adjoint"]
        distance = abs(float(right.mu) - float(left.mu))
        if distance <= 0:
            raise ValueError("Los parámetros deben estar ordenados y ser distintos")
        delta = float(np.linalg.norm(C @ (q - p)))
        sensitivity[j] = delta / distance
        change[j] = delta / (float(np.linalg.norm(C @ p)) + config.safe_epsilon)
    table["local_parameter_lipschitz_indicator"] = sensitivity
    table["relative_snapshot_change"] = change
    finite = sensitivity[np.isfinite(sensitivity)]
    if finite.size:
        limit = config.sensitivity_outlier_factor * max(
            float(np.median(finite)), config.safe_epsilon
        )
        for j in np.flatnonzero(sensitivity > limit):
            if table.at[j, "status"] == "PASS":
                table.at[j, "status"] = "WARNING"
            table.at[j, "reasons"] = (
                table.at[j, "reasons"] + "; local_sensitivity_outlier"
            ).strip("; ")


def select_representative_samples(table: pd.DataFrame, count: int) -> list[int]:
    usable = table[table.status != "FAIL"]
    if usable.empty:
        return []
    chosen = []

    def add(sample_id):
        sample_id = int(sample_id)
        if sample_id not in chosen and len(chosen) < count:
            chosen.append(sample_id)

    for idx in [usable.index[0], usable.index[-1], usable.index[len(usable) // 2]]:
        add(usable.at[idx, "sample_id"])
    for metric in [
        "system_condition_number",
        "terminal_error",
        "control_energy",
        "local_parameter_lipschitz_indicator",
    ]:
        values = usable[metric].dropna()
        if not values.empty:
            add(usable.at[values.idxmax(), "sample_id"])
    for sample_id in usable.loc[usable.status == "WARNING", "sample_id"]:
        add(sample_id)
    # Fill duplicate extrema with points farthest from those already selected.
    while len(chosen) < min(count, len(usable)):
        candidates = usable[~usable.sample_id.isin(chosen)]
        selected_mu = usable.loc[usable.sample_id.isin(chosen), "mu"].to_numpy()
        distances = np.min(
            np.abs(candidates.mu.to_numpy()[:, None] - selected_mu), axis=1
        )
        add(candidates.iloc[int(np.argmax(distances))].sample_id)
    return sorted(chosen)


def summarize_campaign(
    table: pd.DataFrame, refinement: pd.DataFrame, runtime: float
) -> dict:
    def aggregate(column, method):
        values = table[column].dropna()
        return float(getattr(values, method)()) if not values.empty else None

    summary = {
        "number_of_samples": len(table),
        "status_counts": {
            name: int((table.status == name).sum())
            for name in ("PASS", "WARNING", "FAIL")
        },
        "total_runtime_seconds": runtime,
        "mean_fom_runtime_seconds": aggregate("runtime_seconds", "mean"),
        "max_relative_residual": aggregate("relative_residual", "max"),
        "median_relative_residual": aggregate("relative_residual", "median"),
        "max_optimality_error": aggregate("optimality_error", "max"),
        "max_optimality_relative_error": aggregate("optimality_relative_error", "max"),
        "max_terminal_error": aggregate("terminal_error", "max"),
        "min_relative_terminal_error": aggregate("relative_terminal_error", "min"),
        "max_relative_terminal_error": aggregate("relative_terminal_error", "max"),
        "min_gramian_eigenvalue": aggregate("gramian_min_eigenvalue", "min"),
        "max_gramian_eigenvalue": aggregate("gramian_max_eigenvalue", "max"),
        "max_gramian_condition_number": aggregate("gramian_condition_number", "max"),
        "max_system_condition_number": aggregate("system_condition_number", "max"),
        "max_local_parameter_sensitivity": aggregate(
            "local_parameter_lipschitz_indicator", "max"
        ),
        "max_control_energy": aggregate("control_energy", "max"),
        "min_cost_reduction": aggregate("cost_reduction", "min"),
        "mesh_representative_count": int(refinement.sample_id.nunique())
        if not refinement.empty
        else 0,
        "mesh_all_converged": bool(refinement.convergence_decreases.all())
        if not refinement.empty
        else None,
        "flagged_samples": table.loc[
            table.status != "PASS", ["sample_id", "mu", "status", "reasons"]
        ].to_dict("records"),
    }
    summary["campaign_validated"] = bool(
        (table.status == "PASS").all() and summary["mesh_all_converged"]
    )
    return summary
