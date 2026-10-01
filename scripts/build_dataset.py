import argparse
from pathlib import Path

from wave_pdocp.io.dataset_io import build_validated_dataset


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Construir splits solo de PASS, a partir del archivo validado"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=root / "outputs/parameter_sweep/validated_snapshots.h5",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "data/datasets/wave_pdocp_fom.h5"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    counts = build_validated_dataset(args.archive, args.output, args.overwrite)
    print(f"Splits persistidos: {counts}\nDataset: {args.output}")


if __name__ == "__main__":
    main()
