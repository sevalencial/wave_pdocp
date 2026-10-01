from pathlib import Path

from wave_pdocp.io.reports import save_fom_report
from wave_pdocp.orchestration.experiment import SingleMuExperiment


def main():
    experiment = SingleMuExperiment.variable_coefficient_example()
    solution = experiment.run()
    out_path = (
        Path(__file__).resolve().parents[1] / "outputs/single_mu_variable/metrics.json"
    )
    assert experiment.diagnostics is not None
    save_fom_report(solution, experiment.diagnostics, out_path)
    print(f"Diagnosticos guardados en {out_path}")


if __name__ == "__main__":
    main()
