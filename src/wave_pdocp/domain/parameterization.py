from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .grid import SpatialGrid
from .parameter_space import ParameterSpace
from .parameters import SpatialCoefficient, WaveParameter

Array = NDArray[np.float64]


@dataclass(frozen=True)
class WaveSpeedAmplitudeFamily:
    """c_mu=1+mu sin(2pi x/L); all other physical data are fixed.

    ||c_mu-c_nu||_infinity <= |mu-nu|. For |mu| <= a < 1,
    1-a <= c_mu <= 1+a and 0.05 <= d <= 0.15 on the entire domain.
    """

    space: ParameterSpace
    length: float = 1.0
    name: str = "wave_speed_sine_amplitude_v1"

    def __post_init__(self):
        if (
            self.space.dimension != 1
            or max(abs(self.space.lower[0]), abs(self.space.upper[0])) >= 1
        ):
            raise ValueError("Esta familia requiere un intervalo escalar con |mu| < 1")
        if not np.isfinite(self.length) or self.length <= 0:
            raise ValueError("La longitud debe ser positiva y finita")

    def __call__(self, mu) -> WaveParameter:
        if not self.space.contains(mu):
            raise ValueError("mu está fuera del espacio paramétrico")
        amplitude = float(np.atleast_1d(mu)[0])
        return WaveParameter(
            wave_speed=lambda x: 1.0 + amplitude * np.sin(2 * np.pi * x / self.length),
            damping=lambda x: 0.1 + 0.05 * np.cos(2 * np.pi * x / self.length),
            control_center=self.length / 2,
            control_length=self.length,
            target_amplitude=1.0,
            target_mode=1,
        )


@dataclass(frozen=True)
class ParameterFields:
    wave_speed: Array
    wave_speed_half: Array
    damping: Array
    control_mask: Array
    initial_displacement: Array
    initial_velocity: Array
    target_displacement: Array
    target_velocity: Array


def evaluate_coefficient(value: SpatialCoefficient, x: Array, name: str) -> Array:
    """Evaluate a constant/vectorized callable without inventing boundary data."""
    raw = value(x) if callable(value) else value
    if not callable(value) and np.ndim(raw) != 0:
        raise ValueError(f"{name} debe ser una constante o una función de x")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} debe tomar valores reales")
    field = np.asarray(raw, dtype=np.float64)
    if field.ndim == 0:
        field = np.full(x.shape, field.item(), dtype=np.float64)
    elif field.shape != x.shape:
        raise ValueError(
            f"{name}(x) debe devolver un escalar o un vector de forma {x.shape}"
        )
    if not np.all(np.isfinite(field)):
        raise ValueError(f"{name} debe tomar valores finitos")
    return field.copy()


def evaluate_parameter(mu: WaveParameter, grid: SpatialGrid) -> ParameterFields:
    x = grid.interior_points
    L = grid.length

    c = evaluate_coefficient(mu.wave_speed, x, "wave_speed")
    c_half = evaluate_coefficient(mu.wave_speed, grid.half_points, "wave_speed")
    d = evaluate_coefficient(mu.damping, x, "damping")
    if np.any(c <= 0) or np.any(c_half <= 0):
        raise ValueError("wave_speed debe ser positiva en nodos y semipuntos")
    if (
        not isinstance(mu.target_mode, (int, np.integer))
        or isinstance(mu.target_mode, (bool, np.bool_))
        or mu.target_mode < 1
    ):
        raise ValueError("target_mode debe ser un entero positivo")
    if not np.isfinite(mu.target_amplitude):
        raise ValueError("target_amplitude debe ser finita")

    left = mu.control_center - mu.control_length / 2
    right = mu.control_center + mu.control_length / 2

    control_mask = ((x >= left) & (x <= right)).astype(float)

    y0 = np.zeros(grid.n)
    v0 = np.zeros(grid.n)

    yT = mu.target_amplitude * np.sin(mu.target_mode * np.pi * x / L)
    vT = np.zeros(grid.n)

    return ParameterFields(
        wave_speed=c,
        wave_speed_half=c_half,
        damping=d,
        control_mask=control_mask,
        initial_displacement=y0,
        initial_velocity=v0,
        target_displacement=yT,
        target_velocity=vT,
    )
