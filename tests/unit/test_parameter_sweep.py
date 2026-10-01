import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from wave_pdocp.data.splitting import split_indices
from wave_pdocp.diagnostics.fom_validation import (
    ValidationConfig,
    classify_metrics,
    validate_fom,
)
from wave_pdocp.diagnostics.parametric import (
    add_local_diagnostics,
    select_representative_samples,
)
from wave_pdocp.diagnostics.refinement import compare_adjoint_meshes, errors_decrease
from wave_pdocp.domain.grid import SpatialGrid
from wave_pdocp.domain.parameter_space import ParameterSpace
from wave_pdocp.domain.parameterization import (
    WaveSpeedAmplitudeFamily,
    evaluate_parameter,
)
from wave_pdocp.fom.solver import FOMSolver
from wave_pdocp.io.config import SweepConfig, load_sweep_config


def test_parameter_space_bounds_shape_and_future_dimension():
    space = ParameterSpace((0.0,), (0.3,))
    mu = space.sample_uniform_grid(100)
    assert mu.shape == (100, 1)
    assert mu[0, 0] == 0 and mu[-1, 0] == 0.3
    assert np.all(np.diff(mu[:, 0]) > 0)
    assert space.contains(0) and space.contains([0.3])
    assert not space.contains(-1e-6) and not space.contains([np.nan])
    assert not space.contains([0.1, 0.2])
    box = ParameterSpace((0, 1), (1, 2))
    assert box.contains([0.5, 1.5])
    with pytest.raises(NotImplementedError):
        box.sample_uniform_grid(5)


@pytest.mark.parametrize(
    "lower, upper", [((0,), (0,)), ((1,), (0,)), ((0,), (np.inf,)), ((0, 1), (2,))]
)
def test_parameter_space_rejects_invalid_bounds(lower, upper):
    with pytest.raises(ValueError):
        ParameterSpace(lower, upper)


def test_only_wave_speed_varies_and_uniform_physical_bounds_hold():
    grid = SpatialGrid(2.0, 21)
    family = WaveSpeedAmplitudeFamily(ParameterSpace((0,), (0.3,)), length=2.0)
    left = evaluate_parameter(family(0), grid)
    right = evaluate_parameter(family([0.3]), grid)
    for name in (
        "damping",
        "control_mask",
        "initial_displacement",
        "initial_velocity",
        "target_displacement",
        "target_velocity",
    ):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    assert np.min(right.wave_speed_half) >= 0.7
    assert np.max(right.wave_speed_half) <= 1.3
    assert np.max(abs(right.wave_speed_half - left.wave_speed_half)) <= 0.3 + 1e-14
    assert np.min(right.damping) >= 0.05 and np.max(right.damping) <= 0.15
    np.testing.assert_array_equal(right.control_mask, 1.0)
    with pytest.raises(ValueError):
        family([0.31])


@pytest.fixture
def valid_solution():
    family = WaveSpeedAmplitudeFamily(ParameterSpace((0,), (0.3,)))
    return FOMSolver(SpatialGrid(1.0, 12), 2.0, 100.0).solve(family(0.2))


def test_validation_recomputes_residual_and_rejects_nonfinite_arrays(valid_solution):
    assert validate_fom(valid_solution, ValidationConfig()).status == "PASS"
    corrupted = replace(
        valid_solution, final_adjoint=valid_solution.final_adjoint + 0.1
    )
    result = validate_fom(corrupted, ValidationConfig())
    assert result.status == "FAIL" and "relative_residual" in result.reasons
    A = valid_solution.state_operator.copy()
    A[0, 0] = np.nan
    assert (
        validate_fom(
            replace(valid_solution, state_operator=A), ValidationConfig()
        ).status
        == "FAIL"
    )


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"relative_residual": 1e-3}, "FAIL"),
        ({"optimality_relative_error": 1e-3}, "FAIL"),
        ({"gramian_weighted_symmetry_error": 1e-3}, "FAIL"),
        ({"optimal_cost": 1e9}, "FAIL"),
        ({"gramian_min_eigenvalue": -0.1}, "FAIL"),
        ({"gramian_min_eigenvalue": -1e-13}, "WARNING"),
        ({"gramian_min_eigenvalue": -1e-8, "gramian_max_eigenvalue": 1e6}, "WARNING"),
        ({"gramian_condition_number": np.inf}, "WARNING"),
        ({"gramian_condition_number": np.nan}, "FAIL"),
        ({"system_condition_number": 1e9}, "WARNING"),
        ({"state_spectral_abscissa": 1.0}, "FAIL"),
    ],
)
def test_validation_thresholds_are_scale_aware(valid_solution, changes, expected):
    metrics = validate_fom(valid_solution, ValidationConfig()).metrics
    assert (
        classify_metrics({**metrics, **changes}, ValidationConfig()).status == expected
    )


@pytest.mark.parametrize(
    "n, counts", [(100, [75, 15, 10]), (5, [3, 1, 1]), (3, [1, 1, 1])]
)
def test_split_is_disjoint_complete_and_reproducible(n, counts):
    first = split_indices(n)
    second = split_indices(n)
    assert [len(a) for a in first.values()] == counts
    np.testing.assert_array_equal(
        np.sort(np.concatenate(list(first.values()))), np.arange(n)
    )
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_mesh_comparison_on_smooth_known_fields_decreases():
    def state(grid):
        x = grid.interior_points
        return np.r_[x * (1 - x), np.sin(np.pi * x)]

    grids = [SpatialGrid(1.0, n) for n in (20, 40, 80)]
    errors = [
        compare_adjoint_meshes(state(a), a, state(b), b)
        for a, b in itertools.pairwise(grids)
    ]
    assert errors_decrease(errors)
    assert errors[1] < 0.7 * errors[0]
    assert not errors_decrease([0.1, 0.2])


def test_sensitivity_has_no_fake_zero_or_bridge_across_failure():
    grid = SpatialGrid(1.0, 5)
    table = pd.DataFrame(
        {
            "sample_id": range(5),
            "mu": np.arange(5) * 0.1,
            "status": ["PASS", "PASS", "FAIL", "PASS", "PASS"],
            "reasons": [""] * 5,
        }
    )
    snapshots = {i: {"final_adjoint": np.full(10, i * 0.1)} for i in range(5)}
    add_local_diagnostics(table, snapshots, grid, ValidationConfig())
    assert table.local_parameter_lipschitz_indicator.iloc[[0, 2, 3]].isna().all()
    assert table.local_parameter_lipschitz_indicator[1] == pytest.approx(
        table.local_parameter_lipschitz_indicator[4]
    )
    for name in ("system_condition_number", "terminal_error", "control_energy"):
        table[name] = np.arange(5)
    assert select_representative_samples(table, 4) == [0, 1, 3, 4]


def test_invalid_yaml_config_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("parameter_space:\n  min: 0\n  max: 1.5\n")
    with pytest.raises(ValueError):
        load_sweep_config(path)
    with pytest.raises(ValueError):
        SweepConfig.model_validate({"mesh_refinement": {"sizes": [100, 50, 200]}})
