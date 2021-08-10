"""
    utils
    ~~~~~

    Helper functions.
"""
from collections.abc import Iterable
from numbers import Number
from typing import Any, Dict, List, Union

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


def reduce_by_sizes(array: np.ndarray,
                    group_sizes: np.ndarray,
                    ufunc: np.ufunc = np.add,
                    axis: int = 0,
                    **kwargs: Dict) -> np.ndarray:
    """Reduce array by group, this is a wrapper for np.reduceat.

    Parameters
    ----------
    array : np.ndarray
        Numpy array to be reduced.
    group_sizes : np.ndarray
        The group sizes of the array. Requires the sum of group sizes equal to
        the size of the array.
    ufunc : np.ufunc, optional
        Numpy ufunc, default to ``np.add``.
    axis : int, optional
        Axis to be reduced, default to be 0.
    kwargs : Dict
        Additional keyword arguments.

    Returns
    -------
    Reduced array with size equal to length of ``group_sizes``.
    """
    array = np.asarray(array)
    group_sizes = np.asanyarray(group_sizes).astype(int)
    if not isinstance(ufunc, np.ufunc):
        raise TypeError("ufunc must be a instance of numpy.ufunc.")
    if any(group_sizes <= 0) or sum(group_sizes) != array.shape[axis]:
        raise ValueError("group sizes must be an array of positive number with "
                         "sum equals to length of the array.")

    # convert the group_sizes to indicies
    indices = np.cumsum(np.hstack([0, group_sizes]))[:-1]

    return ufunc.reduceat(array, indices, axis=axis, **kwargs)
