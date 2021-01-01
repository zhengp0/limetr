"""
    utils
    ~~~~~

    Helper functions.
"""
from typing import List, Any, Union
from numbers import Number
from collections.abc import Iterable
import numpy as np
from spmat.dlmat import DLMat, BDLMat


def split_by_sizes(array: np.ndarray,
                   sizes: List[int],
                   axis: int = 0) -> List[np.ndarray]:
    assert array.shape[axis] == sum(sizes)
    return np.split(array, np.cumsum(sizes)[:-1], axis=axis)


def empty_array():
    return np.array([])


def default_vec_factory(vec: Union[Number, Iterable],
                        size: int,
                        default_value: float,
                        vec_name: str = 'attr') -> np.ndarray:
    if np.isscalar(vec):
        vec = np.repeat(vec, size)
    elif len(vec) == 0:
        vec = np.repeat(default_value, size)
    else:
        vec = np.asarray(vec)
        check_size(vec, size, vec_name=vec_name)

    return vec


def check_size(vec: Any, size: int, vec_name: str = 'attr'):
    assert len(vec) == size, f"{vec_name} must length {size}."


def iterable(__obj: object) -> bool:
    return isinstance(__obj, Iterable)


def has_no_repeat(array: np.ndarray) -> bool:
    return array.size == np.unique(array).size


def sizes_to_slices(sizes: np.array) -> List[slice]:
    ends = np.cumsum(sizes)
    starts = np.insert(ends, 0, 0)[:-1]
    return [slice(*pair) for pair in zip(starts, ends)]


def get_varmat(gamma: np.ndarray,
               obsvar: List[np.ndarray],
               remat: List[np.ndarray]) -> BDLMat:
    assert len(obsvar) == len(remat)
    sqrt_gamma = np.sqrt(gamma)
    dlmats = [
        DLMat(obsvar[i], remat[i]*sqrt_gamma)
        for i in range(len(obsvar))
    ]
    return BDLMat(dlmats)
