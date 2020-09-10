"""
    helper_funs
    ~~~~~~~~~~~

    Helper functions
"""
from typing import List, Union
from warnings import warn
import numpy as np
import scipy.optimize as spopt


def split_by_sizes(array: np.ndarray, sizes: List[int], axis: int = 0) -> List[np.ndarray]:
    assert array.shape[axis] == sum(sizes)
    return np.split(array, np.cumsum(sizes)[:-1], axis=axis)


def project_to_capped_simplex(weights: np.ndarray, cap: Union[int, float],
                              active_index: np.ndarray = None) -> np.ndarray:
    # process active index
    if active_index is None:
        active_index = np.array([True]*weights.size)
    else:
        active_index = np.array([i in active_index for i in range(weights.size)])

    active_cap = cap - (~active_index).sum()
    weights = weights.copy()
    weights[~active_index] = 1.0

    if active_cap <= 0:
        warn(f"Cap ({cap}) is too small for the number of inactive weights ({(~active_index).sum()})."
             f"Set all inactive weights to zero and all active weights to one.")
        weights[active_index] = 0.0
    else:
        active_weights = weights[active_index]
        lb = np.min(active_weights) - 1.0
        ub = np.max(active_weights) - 0.0
        x = spopt.bisect(lambda x: np.sum(np.maximum(np.minimum(active_weights - x, 1.0), 0.0)) - active_cap,
                         lb, ub)
        weights[active_index] = np.maximum(np.minimum(active_weights - x, 1.0), 0.0)
    return weights
