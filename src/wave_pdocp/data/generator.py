from dataclasses import asdict

import numpy as np

from ..domain.grid import SpatialGrid
from ..fom.reconstruction import reconstruct_final_state
from ..fom.solution import FOMSolution
from .schema import FIELD_NAMES


def snapshot_from_solution(solution: FOMSolution) -> dict:
    snapshot = asdict(solution.fields)
    snapshot["final_adjoint"] = solution.final_adjoint.copy()
    snapshot["final_state"] = reconstruct_final_state(solution)
    return snapshot


def stack_snapshots(
    snapshots: dict, table, parameters: np.ndarray, grid: SpatialGrid
) -> dict:
    # Preserve finite failed/warning snapshots in the audit archive, not the split.
    ids = sorted(
        i
        for i, s in snapshots.items()
        if all(np.all(np.isfinite(v)) for v in s.values())
    )
    samples = {
        "sample_id": np.asarray(ids, dtype=np.int64),
        "mu": parameters[ids].reshape(len(ids), parameters.shape[1]),
        "status": np.asarray(
            [table.loc[table.sample_id == i, "status"].iloc[0] for i in ids], dtype=str
        ),
    }
    for name in (*FIELD_NAMES, "final_adjoint", "final_state"):
        width = (
            2 * grid.n
            if name in ("final_adjoint", "final_state")
            else grid.n + 1
            if name == "wave_speed_half"
            else grid.n
        )
        samples[name] = (
            np.stack([snapshots[i][name] for i in ids]) if ids else np.empty((0, width))
        )
    return samples
