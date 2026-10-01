import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

Array = NDArray[np.float64]

def compute_gramian(A: Array, B: Array, A_adj: Array, B_adj: Array, final_time: float) -> Array:
    n_state = A.shape[0]
    
    G = B @ B_adj
    zero = np.zeros_like(A)
    
    block_matrix = np.block([
        [A, G],
        [zero, -A_adj]
    ])
    
    E = expm(final_time * block_matrix)
    
    upper_right = E[:n_state, n_state:]
    exp_adj = expm(final_time * A_adj)
    
    W = upper_right @ exp_adj
    
    return W