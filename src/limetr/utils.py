"""
    utils
    ~~~~~

    Helper functions.
"""
from typing import List, Any
from collections.abc import Iterable
import numpy as np


def split_by_sizes(array: np.ndarray, sizes: List[int], axis: int = 0) -> List[np.ndarray]:
    assert array.shape[axis] == sum(sizes)
    return np.split(array, np.cumsum(sizes)[:-1], axis=axis)


def empty_array():
    return np.array([])


def default_attr_factory(attr: Any,
                         size: int,
                         default_value: float,
                         attr_name: str = 'attr') -> np.ndarray:
    if not attr:
        attr = np.repeat(default_value, size)
    elif np.isscalar(attr):
        attr = np.repeat(attr, size)
    else:
        attr = np.asarray(attr)
        check_size(attr, size, attr_name=attr_name)

    return attr


def check_size(attr: Any, size: int, attr_name: str = 'attr'):
    assert len(attr) == size, f"{attr_name} must length {size}."


def iterable(__obj: object) -> bool:
    return isinstance(__obj, Iterable)


def has_no_repeat(array: np.ndarray) -> bool:
    return array.size == np.unique(array).size


def sizes_to_slices(sizes: np.array) -> List[slice]:
    ends = np.cumsum(sizes)
    starts = np.insert(ends, 0, 0)[:-1]
    return [slice(*pair) for pair in zip(starts, ends)]
