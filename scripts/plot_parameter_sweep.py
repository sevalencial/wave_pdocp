"""Regenerate figures from saved snapshots without solving or changing splits."""

import argparse
from pathlib import Path

import pandas as pd

from wave_pdocp.domain.grid import SpatialGrid
from wave_pdocp.io.dataset_io import read_dataset
from wave_pdocp.visualization.parametric_plots import plot_campaign


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=root / "outputs/parameter_sweep/validated_snapshots.h5",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    samples, metadata, _ = read_dataset(args.archive)
    paths = metadata["config"]["outputs"]
    table = pd.read_csv(root / paths["metrics"])
    refinement_path = root / paths["refinement"]
    try:
        refinement = pd.read_csv(refinement_path)
    except pd.errors.EmptyDataError:
        refinement = pd.DataFrame()
    grid = SpatialGrid(metadata["domain_length"], metadata["grid_size"])
    output_dir = args.output_dir or root / paths["figures"]
    plot_campaign(
        table,
        samples,
        grid,
        refinement,
        metadata["representative_sample_ids"],
        output_dir,
    )
    print(f"Figuras regeneradas desde snapshots en {output_dir}")


if __name__ == "__main__":
    main()
