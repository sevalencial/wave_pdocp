import numpy as np
import pytest

from wave_pdocp.domain.grid import SpatialGrid
from wave_pdocp.domain.parameterization import evaluate_parameter
from wave_pdocp.domain.parameters import WaveParameter


def test_constants_and_constant_callables_agree():
    grid = SpatialGrid(length=2.0, n_interior=12)
    constant = evaluate_parameter(WaveParameter(wave_speed=1.4, damping=0.2), grid)
    callable_fields = evaluate_parameter(
        WaveParameter(wave_speed=lambda x: 1.4, damping=lambda x: np.full_like(x, 0.2)),
        grid,
    )
    for name in ("wave_speed", "wave_speed_half", "damping"):
        np.testing.assert_array_equal(
            getattr(constant, name), getattr(callable_fields, name)
        )


def test_variable_coefficients_are_evaluated_at_the_actual_locations():
    grid = SpatialGrid(length=2.0, n_interior=4)
    seen = []

    def speed(x):
        seen.append(x.copy())
        return 1.0 + x**2

    fields = evaluate_parameter(
        WaveParameter(wave_speed=speed, damping=lambda x: x / 10), grid
    )
    np.testing.assert_allclose(grid.half_points, [0.2, 0.6, 1.0, 1.4, 1.8])
    np.testing.assert_array_equal(seen[0], grid.interior_points)
    np.testing.assert_array_equal(seen[1], grid.half_points)
    np.testing.assert_allclose(fields.wave_speed_half, 1.0 + grid.half_points**2)
    np.testing.assert_allclose(fields.damping, grid.interior_points / 10)


def test_target_amplitude_and_mode_are_independent():
    grid = SpatialGrid(length=2.0, n_interior=19)
    fields = evaluate_parameter(
        WaveParameter(target_amplitude=2.5, target_mode=3), grid
    )
    np.testing.assert_allclose(
        fields.target_displacement, 2.5 * np.sin(3 * np.pi * grid.interior_points / 2.0)
    )


@pytest.mark.parametrize(
    "coefficient",
    [
        0.0,
        -1.0,
        np.nan,
        lambda x: np.zeros_like(x),
        lambda x: np.full_like(x, np.inf),
        lambda x: np.ones((x.size, 1)),
        lambda x: np.ones(x.size + 1),
        lambda x: 1j,
        lambda x: np.where(x < 0.1, -1.0, 1.0),
    ],
)
def test_invalid_wave_speed_is_rejected(coefficient):
    with pytest.raises(ValueError, match="wave_speed"):
        evaluate_parameter(WaveParameter(wave_speed=coefficient), SpatialGrid(1.0, 5))


@pytest.mark.parametrize(
    "coefficient", [np.nan, lambda x: np.full_like(x, np.inf), lambda x: x[:, None]]
)
def test_invalid_damping_is_rejected(coefficient):
    with pytest.raises(ValueError, match="damping"):
        evaluate_parameter(WaveParameter(damping=coefficient), SpatialGrid(1.0, 5))


@pytest.mark.parametrize("mode", [0, -1, 1.5, True])
def test_non_dirichlet_target_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="target_mode"):
        evaluate_parameter(WaveParameter(target_mode=mode), SpatialGrid(1.0, 5))
