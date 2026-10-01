import json
from dataclasses import asdict
from pathlib import Path

from ..diagnostics.fom import FOMDiagnostics
from ..fom.solution import FOMSolution


def save_fom_report(
    solution: FOMSolution, diagnostics: FOMDiagnostics, path: Path
) -> None:
    """Save the sampled inputs alongside diagnostics so the case is identifiable."""
    report = {
        "grid": {"length": solution.grid.length, "n_interior": solution.grid.n},
        "final_time": solution.final_time,
        "beta": solution.beta,
        "x": solution.grid.interior_points.tolist(),
        "x_half": solution.grid.half_points.tolist(),
        "fields": {
            name: value.tolist() for name, value in asdict(solution.fields).items()
        },
        "diagnostics": asdict(diagnostics),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
