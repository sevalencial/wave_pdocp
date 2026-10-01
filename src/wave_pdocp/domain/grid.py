from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SpatialGrid:
    length: float
    n_interior: int

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("La longitud debe ser un valor positivo")

        if self.n_interior < 1:
            raise ValueError("n_interior debe ser mayor a 1")

    @property
    def h(self) -> float:
        return self.length / (self.n_interior + 1)

    @property
    def interior_points(self) -> NDArray[np.float64]:
        return np.arange(1, self.n_interior + 1) * self.h

    @property
    def half_points(self) -> NDArray[np.float64]:
        """The N+1 flux locations, including both boundary-adjacent faces."""
        return (np.arange(self.n_interior + 1) + 0.5) * self.h

    @property
    def n(self) -> int:
        return self.n_interior
