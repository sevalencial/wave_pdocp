import argparse
import json
from pathlib import Path

from wave_pdocp.io.config import load_sweep_config
from wave_pdocp.orchestration.parameter_sweep import ParameterSweepExperiment


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Barrido FOM, validación, refinamiento y dataset"
    )
    parser.add_argument(
        "--config", type=Path, default=root / "configs/parameter_sweep.yaml"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reemplazar explícitamente resultados y splits existentes",
    )
    args = parser.parse_args()
    result = ParameterSweepExperiment(load_sweep_config(args.config), root).run(
        overwrite=args.overwrite
    )
    print(json.dumps(result.summary, indent=2, default=str))
    print(
        f"Dataset: {result.dataset_path}\nDiagnósticos: {result.metrics_path}\nResumen: {result.summary_path}"
    )
    if "FAIL" in result.statuses:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
