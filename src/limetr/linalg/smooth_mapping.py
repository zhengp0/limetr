"""
    smooth_mapping
    ~~~~~~~~~~~~~~

    Smooth mapping module.
"""
from typing import Callable, Tuple

import numpy as np


class SmoothMapping:
    """
    Smooth mapping class, including function and Jacobian function.

    Attributes
    ----------
    shape: Tuple[int, int]
        Shape of the mapping.
    fun: Callable
        Mapping function.
    jac: Callable
        Jacobian function of the mapping.
    """

    def __init__(self,
                 shape: Tuple[int, int],
                 fun: Callable,
                 jac: Callable):
        """
        Parameters
        ----------
        shape : Tuple[int, int]
            Shape of the mapping.
        fun : Callable
            Mapping function.
        jac : Callable
            Jacobian function of the mapping.

        Raises
        ------
        AssertionError
            If ``shape`` is not a tuple with two positive intergers.
        AssertionError
            If ``fun`` is not callable.
        AssertionError
            If ``jac`` is not callable.
        """
        assert (isinstance(shape, tuple) and
                shape == 2 and
                all([isinstance(size, int) and size > 0 for size in shape])), \
            "`shape` has to be tuple with two positive integers."
        assert callable(fun), "`fun` must be callable."
        assert callable(jac), "`jac` must be callable."
        self.shape = shape
        self.fun = self._validate_input(fun)
        self.jac = self._validate_input(jac)

    def _validate_input(self, fun: Callable) -> Callable:
        """
        Decorator to validate the input before pass into the function.

        Parameters
        ----------
        fun : Callable
            Function to be decorated

        Returns
        -------
        Callable
            Function that valide the input.
        """
        def decorated_fun(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            if x.size != self.shape[1]:
                raise ValueError("Input size not matching with mapping shape.")
            return fun(x)
        return decorated_fun

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.fun(x)

    def __repr__(self) -> str:
        return f"SmoothMapping(shape={self.shape})"


# pylint:disable=too-few-public-methods
class LinearMapping(SmoothMapping):
    """
    Linear mapping class, construct smooth mapping from a matrix.

    Attributes
    ----------
    mat: ndarray
        Matrix as the linear mapping.
    """

    def __init__(self, mat: np.ndarray):
        mat = np.asarray(mat)
        assert mat.ndim == 2, "`mat` must be a matrix."
        self.mat = mat

        # pylint: disable=unused-argument
        def fun(x): return mat.dot(x)
        def jac(x): return mat

        super().__init__(mat.shape, fun, jac)

    def __repr__(self) -> str:
        return f"LinearMapping(shape={self.shape})"
