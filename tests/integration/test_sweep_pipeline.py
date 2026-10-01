import numpy as np
import pytest

from wave_pdocp.io.config import SweepConfig
from wave_pdocp.io.dataset_io import build_validated_dataset, read_dataset
from wave_pdocp.orchestration.parameter_sweep import ParameterSweepExperiment


def small_config():
    return SweepConfig.model_validate(
        {
            "discretization": {"n_interior": 20},
            "parameter_space": {"n_samples": 5},
            "mesh_refinement": {"sizes": [10, 20, 40], "representative_samples": 3},
        }
    )


def test_five_fom_campaign_roundtrip_refinement_and_split(tmp_path):
    result = ParameterSweepExperiment(small_config(), tmp_path).run(make_plots=False)
    assert result.statuses == ["PASS"] * 5
    assert result.summary["mesh_all_converged"]
    samples, metadata, splits = read_dataset(result.dataset_path)
    archive, _, _ = read_dataset(result.snapshots_path)
    assert samples["mu"].shape == (5, 1)
    assert samples["final_adjoint"].shape == (5, 40)
    assert samples["wave_speed_half"].shape == (5, 21)
    assert metadata["physical_bounds"]["c_min"] == 0.7
    for key in samples:
        np.testing.assert_array_equal(samples[key], archive[key])
    merged = np.concatenate(list(splits.values()))
    np.testing.assert_array_equal(np.sort(merged), np.arange(5))
    assert [len(v) for v in splits.values()] == [1, 3, 1]  # HDF5 names are alphabetical
    with pytest.raises(FileExistsError):
        ParameterSweepExperiment(small_config(), tmp_path).run(make_plots=False)
    with pytest.raises(FileExistsError):
        build_validated_dataset(result.snapshots_path, result.dataset_path)
    copy_path = tmp_path / "copy.h5"
    build_validated_dataset(result.snapshots_path, copy_path)
    _, _, copied_splits = read_dataset(copy_path)
    for key in splits:
        np.testing.assert_array_equal(splits[key], copied_splits[key])


def test_warning_samples_are_archived_but_not_split(tmp_path):
    config = small_config().model_dump()
    config["validation"]["system_condition_warning"] = 1.0001
    config["mesh_refinement"]["enabled"] = False
    result = ParameterSweepExperiment(SweepConfig.model_validate(config), tmp_path).run(
        make_plots=False
    )
    assert result.statuses == ["WARNING"] * 5
    archive, _, _ = read_dataset(result.snapshots_path)
    samples, metadata, splits = read_dataset(result.dataset_path)
    assert archive["final_adjoint"].shape == (5, 40)
    assert samples["final_adjoint"].shape == (0, 40)
    assert metadata["split_created"] is False
    assert splits == {}


def test_solve_exception_is_recorded_and_does_not_abort_remaining_cases(tmp_path):
    config = small_config().model_dump()
    config["mesh_refinement"]["enabled"] = False
    experiment = ParameterSweepExperiment(SweepConfig.model_validate(config), tmp_path)
    original = experiment.solver.solve
    calls = []

    def injected_failure(parameter):
        calls.append(parameter)
        if len(calls) == 2:
            raise np.linalg.LinAlgError("injected solver failure")
        return original(parameter)

    experiment.solver.solve = injected_failure
    result = experiment.run(make_plots=False)
    assert len(calls) == 5
    assert result.statuses == ["PASS", "FAIL", "PASS", "PASS", "PASS"]
    assert "injected solver failure" in result.diagnostics.reasons.iloc[1]
    assert (
        result.diagnostics.local_parameter_lipschitz_indicator.iloc[[1, 2]].isna().all()
    )
    samples, _, _ = read_dataset(result.dataset_path)
    np.testing.assert_array_equal(samples["sample_id"], [0, 2, 3, 4])
