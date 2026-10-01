import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

def build_state_operator(L: Array, D: Array) -> Array:
    
    n = L.shape[0]
    
    zero = np.zeros((n, n))
    identity = np.eye(n)
    
    A = np.block([
        [zero, identity],
        [L, -D]
    ])
    return A

def build_control_operator(C: Array) -> Array:
    n = C.shape[0]
    zero = np.zeros((n, n))
    
    return np.vstack([
        zero,
        C
    ])