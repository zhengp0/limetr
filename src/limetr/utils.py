"""
    utils
    ~~~~~

    Helper functions.
"""
from typing import List
import numpy as np


def split_by_sizes(array: np.ndarray, sizes: List[int], axis: int = 0) -> List[np.ndarray]:
    assert array.shape[axis] == sum(sizes)
    return np.split(array, np.cumsum(sizes)[:-1], axis=axis)
