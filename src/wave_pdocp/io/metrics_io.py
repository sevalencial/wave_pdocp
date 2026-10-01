import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def save_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def save_json(value: dict, path: Path) -> None:
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(v) for key, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        if isinstance(item, np.generic):
            return item.item()
        return item

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def campaign_metadata(config, root: Path) -> dict:
    def git(*args):
        process = subprocess.run(
            ["git", *args], cwd=root, text=True, capture_output=True, check=False
        )
        return process.stdout.strip() if process.returncode == 0 else None

    source_hash = hashlib.sha256()
    for path in sorted(
        [*root.joinpath("src").rglob("*.py"), *root.joinpath("scripts").glob("*.py")]
    ):
        source_hash.update(str(path.relative_to(root)).encode())
        source_hash.update(path.read_bytes())
    raw = config.model_dump(mode="json")
    amplitude = max(abs(config.parameter_space.min), abs(config.parameter_space.max))
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_sha256": source_hash.hexdigest(),
        "config_sha256": hashlib.sha256(
            json.dumps(raw, sort_keys=True).encode()
        ).hexdigest(),
        "config": raw,
        "grid_size": config.discretization.n_interior,
        "domain_length": config.domain.length,
        "final_time": config.domain.final_time,
        "beta": config.optimal_control.beta,
        "parameter_dimension": 1,
        "parameter_space": {
            "lower": [config.parameter_space.min],
            "upper": [config.parameter_space.max],
        },
        "parameterization_name": config.parameter_space.parameterization,
        "parameterization_formulas": {
            "c": "1 + mu[0]*sin(2*pi*x/L)",
            "d": "0.1 + 0.05*cos(2*pi*x/L)",
            "target": "(sin(pi*x/L), 0)",
            "initial": "(0, 0)",
            "control": "(0, L)",
        },
        "physical_bounds": {
            "c_min": 1 - amplitude,
            "c_max": 1 + amplitude,
            "d_min": 0.05,
            "d_max": 0.15,
            "lipschitz_c": 1.0,
        },
        "dataset_version": config.dataset.version,
        "role": "validated_snapshot_archive",
        "relative_metrics_epsilon": config.validation.safe_epsilon,
        "split_interpretation": "random interpolation within parameter interval; no extrapolation claim",
    }
