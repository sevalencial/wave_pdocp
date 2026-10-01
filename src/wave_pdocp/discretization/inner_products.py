import numpy as np
from numpy.typing import NDArray

from ..domain.grid import SpatialGrid

Array = NDArray[np.float64]

def build_K_h_matrix(grid: SpatialGrid) -> Array:
    n = grid.n
    h = grid.h
    K = np.zeros((n, n))
    
    np.fill_diagonal(K, 2.0)
    k = np.arange(n-1)
    K[k, k + 1] = -1.0
    K[k + 1, k] = -1.0
    
    return K / h 
        
def build_Q_h_matrix(grid: SpatialGrid) -> Array:
    n = grid.n
    h = grid.h
    
    K = build_K_h_matrix(grid)
    M = h * np.eye(n)
    
    Q = np.block([
        [K, np.zeros((n,n))],
        [np.zeros((n,n)), M]
    ])
    return Q

def build_R_h_matrix(grid: SpatialGrid) -> Array:
    return grid.h * np.eye(grid.n)