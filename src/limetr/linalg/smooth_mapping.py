"""
    smooth_mapping
    ~~~~~~~~~~~~~~

    Smooth mapping module.
"""
from typing import Tuple, Callable
import numpy as np


class SmoothMapping:
    def __init__(self, shape: Tuple, fun: Callable, jac_fun: Callable):
        self.shape = shape
        self.fun = fun
        self.jac_fun = jac_fun
        self.check_attr()

    def check_attr(self):
        assert callable(self.fun), "fun must be callable."
        assert callable(self.jac_fun), "jac_fun must be callable."

        assert isinstance(self.shape, tuple), "shape must be a tuple."
        assert len(self.shape) == 2, "shape must contains two numbers."
        assert all([size > 0 and isinstance(size, int) for size in self.shape]), \
            "Both numbers in shape must be positive integer."


class LinearMapping(SmoothMapping):
    def __init__(self, mat: np.ndarray):
        self.mat = np.asarray(mat)
        assert self.mat.ndim == 2
        super().__init__(mat.shape,
                         lambda x: mat.dot(x),
                         lambda x: mat)
