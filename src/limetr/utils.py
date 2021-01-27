"""
    utils
    ~~~~~

    Helper functions.
"""
from collections.abc import Iterable
from numbers import Number
from typing import Any, List, Union

import numpy as np
from spmat.dlmat import BDLMat, DLMat


def split_by_sizes(array: np.ndarray,
                   sizes: List[int],
                   axis: int = 0) -> List[np.ndarray]:
    """
    Function that split an array into a list of arrays, provided the size for
    each sub-array size.

    Parameters
    ----------
    array : ndarray
        The array need to be splitted.
    sizes : List[int]
        A list of sizes for each sub-array.
    axis: int, optional
        Along which axis, array will be splitted, default is 0.

    Raises
    ------
    AssertionError
        If the sum of the ``sizes`` does not equal to the shape of ``array``
        along ``axis``.

    Returns
    -------
    List[ndarray]
        A list of splitted array. 
    """
    assert array.shape[axis] == sum(sizes)
    return np.split(array, np.cumsum(sizes)[:-1], axis=axis)


def empty_array() -> np.ndarray:
    """
    Function used for 'default_factory', creates and returns empty array.

    Returns
    -------
    ndarray
        An empty array with ``dtype`` being ``float``.
    """
    return np.array([])


def default_vec_factory(vec: Union[Number, Iterable],
                        size: int,
                        default_value: Any = None,
                        vec_name: str = 'vector') -> np.ndarray:
    """
    Function that automatically create and fill values of a vector.

    Parameters
    ----------
    vec : Union[Number, Iterable]
        A vector or number that need to be checked or expand.
    size : int
        The desired size of the vector.
    default_value : Any, optional
        Default value of the vector, will be used when ``vec`` is empty.
        Default is ``None``.
    vec_name : str, optional
        Name of the vector, for more informative error message.
        Default to be ``'vector'``.

    Raises
    ------
    AssertionError
        If ``vec`` is empty and ``default_value`` is ``None``.
    AssertionError
        If ``vec`` is ``Iterable`` and length does not equal to ``size``.

    Returns
    -------
    ndarray:
        Final processed array.
    """
    if np.isscalar(vec):
        vec = np.repeat(vec, size)
    elif len(vec) == 0:
        assert default_value is not None, \
            "Must provide `default_value` when `vec` is empty."
        vec = np.repeat(default_value, size)
    else:
        vec = np.asarray(vec)
        check_size(vec, size, vec_name=vec_name)

    return vec


def check_size(vec: Iterable, size: int, vec_name: str = 'vector'):
    """
    Function that check the size consistency.

    Parameters
    ----------
    vec : Iterable
        Iterable vector which length will be checked.
    size : int
        Desired size of the vector.
    vec_name : str, optional
        Name of the vector, for more informative error message.
        Default to be ``'vector'``.

    Raises
    ------
    ValueError
        If vector length does not equal to provided ``size``.
    """
    if len(vec) != size:
        raise ValueError(f"{vec_name} must be length {size}.")


def iterable(__obj: object) -> bool:
    """
    Function that check if an object is iterable.

    Parameters
    ----------

    __obj : object
        Object to be examed.

    Returns
    -------
    bool
        ``True`` if object is iterable, and ``False`` otherwise.
    """
    return isinstance(__obj, Iterable)


def has_no_repeat(array: np.ndarray) -> bool:
    """
    Function that check if an array have no repeat values.

    Parameters
    ----------
    array : ndarray
        Array that need to be examed.

    Returns
    -------
    bool
        ``True`` if array has no repeat values, and ``False`` otherwise.
    """
    return array.size == np.unique(array).size


def sizes_to_slices(sizes: Iterable) -> List[slice]:
    """
    Function that convert sizes of sub-arrays to corresponding slices in the
    original array.

    Parameters
    ----------
    sizes : Iterable[int]
        Iterable object contains positive integers as the sizes of the arrays.

    Returns
    -------
    List[slice]
        A list of ``slice`` to access each sub-array in the original array.
    """
    ends = np.cumsum(sizes)
    starts = np.insert(ends, 0, 0)[:-1]
    return [slice(*pair) for pair in zip(starts, ends)]


def get_varmat(gamma: np.ndarray,
               obsvar: List[np.ndarray],
               remat: List[np.ndarray]) -> BDLMat:
    """
    Function that compute the variance-covariance matrix for the mixed effects
    model, used in optimization interface.

    Parameters
    ----------
    gamma : ndarray
        Variance vector of the random effects.
    obsvar : List[ndarray]
        Variance vectors of the obervation errors. Each array in the list
        corresponding to a group.
    remat : List[ndarray]
        Random effects design matrices. Each matrix in the list corresponding to
        a group.

    Raises
    ------
    AssertionError
        If length of ``obsvar`` does not equal to length of ``remat``. Both
        lengths should agree with number of groups.

    Returns
    -------
    BDLMat
        Block diagonal matrix object.
    """
    assert len(obsvar) == len(remat)
    sqrt_gamma = np.sqrt(gamma)
    dlmats = [
        DLMat(obsvar[i], remat[i]*sqrt_gamma)
        for i in range(len(obsvar))
    ]
    return BDLMat(dlmats)


def get_maxlen(objs: List[Any]) -> int:
    """
    Get the maximum len of a list of objects.

    Parameters
    ----------
    objs : List[Any]
        A list of objects.

    Returns
    -------
    int
        Maximum length among objects.
    """
    return max([len(obj) if iterable(obj) else 1 for obj in objs])


def broadcast(objs: List[Any],
              size: int) -> np.ndarray:
    """
    Broadcast a list of objects.

    Parameters
    ----------
    objs : List[Any]
        A list of objects.
    size : int
        Size for the broadcasting.

    Raises
    ------
    ValueError
        If there is iterable object in the list whose size is not 1 and not
        agree with broadcast size.

    Returns
    -------
    np.ndarray
        Two dimensional array that stores the squared objects.
    """
    size = int(size)
    assert size >= 0, "Size has to be a non-negative integer."
    if size == 0:
        vecs = np.empty(shape=(len(objs), 0))
    else:
        for i, obj in enumerate(objs):
            if np.isscalar(obj) or len(obj) == 1:
                objs[i] = np.repeat(obj, size)
            elif len(obj) == size:
                objs[i] = np.asarray(obj)
            else:
                raise ValueError("Object size not consistent with broadcast "
                                 "size.")
        vecs = np.vstack(objs)
    return vecs
