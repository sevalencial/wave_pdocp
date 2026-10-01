from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParameterSpace:
    """Closed parameter box; parameters and samples always retain a p axis."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self):
        lo, hi = np.atleast_1d(self.lower), np.atleast_1d(self.upper)
        if lo.ndim != 1 or lo.size == 0 or lo.shape != hi.shape:
            raise ValueError("Los límites deben tener la misma dimensión positiva")
        if not np.all(np.isfinite([lo, hi])) or np.any(lo >= hi):
            raise ValueError("Los límites deben ser finitos y lower < upper")
        object.__setattr__(self, "lower", tuple(float(x) for x in lo))
        object.__setattr__(self, "upper", tuple(float(x) for x in hi))

    @property
    def dimension(self) -> int:
        return len(self.lower)

    def contains(self, mu) -> bool:
        value = np.atleast_1d(np.asarray(mu, dtype=float))
        return bool(
            value.shape == (self.dimension,)
            and np.all(np.isfinite(value))
            and np.all(value >= self.lower)
            and np.all(value <= self.upper)
        )

    def sample_uniform_grid(self, n_samples: int) -> np.ndarray:
        if (
            not isinstance(n_samples, int)
            or isinstance(n_samples, bool)
            or n_samples < 2
        ):
            raise ValueError("Se requieren al menos dos muestras")
        if self.dimension != 1:
            raise NotImplementedError("La primera campaña usa una malla paramétrica 1D")
        return np.linspace(self.lower[0], self.upper[0], n_samples)[:, None]
