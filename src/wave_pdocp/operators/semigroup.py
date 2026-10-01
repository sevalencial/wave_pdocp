import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

Array = NDArray[np.float64]

def compute_semigroup(A: Array, time: float) -> Array:
    
    if time < 0:
        raise ValueError("El tiempo no puede ser negativo")

    return expm(A * time)