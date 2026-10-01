import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve

Array = NDArray[np.float64]

def A_adjoint(A: Array, Q: Array) -> Array:
    return solve(
        Q,
        A.T @ Q,
        assume_a="pos"
    )

def B_adjoint(B: Array, Q: Array, R: Array) -> Array:
    return solve(
        R,
        B.T @ Q,
        assume_a="pos"
    )