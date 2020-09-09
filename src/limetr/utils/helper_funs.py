"""
    helper_funs
    ~~~~~~~~~~~

    Helper functions
"""
from typing import List
import numpy as np


def split_by_sizes(vec: np.ndarray, sizes: List[int]) -> List[np.ndarray]:
    assert len(vec) == sum(sizes)
    return np.split(vec, np.cumsum(sizes)[:-1])
