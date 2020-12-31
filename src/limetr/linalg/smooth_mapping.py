"""
    smooth_mapping
    ~~~~~~~~~~~~~~

    Smooth mapping module.
"""
from typing import Tuple, Callable
import numpy as np


class SmoothMapping:
    """
    Smooth mapping class, callable and including Jacobian function.
    """

    def __init__(self,
                 shape: Tuple[int, int],
                 fun: Callable,
                 jac: Callable):
        self.shape = shape
        self._fun = fun
        self._jac = jac
        self.check_attr()

    def check_attr(self):
        assert callable(self._fun), "`fun` must be callable."
        assert callable(self._jac), "`jac` must be callable."

        assert isinstance(self.shape, tuple), "shape must be a tuple."
        assert len(self.shape) == 2, "shape must contains two numbers."
        assert all([size > 0 and isinstance(size, int) for size in self.shape]), \
            "Both numbers in shape must be positive integer."

    def check_input(self, x: np.ndarray):
        if x.size != self.shape[1]:
            raise ValueError("Input size not matching with mapping shape.")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.check_input(x)
        return self._fun(x)

    def jac(self, x: np.ndarray) -> np.ndarray:
        self.check_input(x)
        return self._jac(x)

    def __repr__(self) -> str:
        return f"SmoothMapping(shape={self.shape})"


class LinearMapping(SmoothMapping):
    """
    Linear mapping class, construct smooth mapping from a matrix.
    """

    def __init__(self, mat: np.ndarray):
        self.mat = np.asarray(mat)
        if self.mat.ndim != 2:
            raise ValueError("`mat` must be a matrix.")

        #pylint: disable=unused-argument
        def fun(x): return mat.dot(x)
        def jac(x): return mat

        super().__init__(mat.shape, fun, jac)

    def __repr__(self) -> str:
        return f"LinearMapping(shape={self.shape})"
