"""
Prior Module
"""
from typing import List, Any
from dataclasses import dataclass, field

import numpy as np

from limetr.utils import default_vec_factory, iterable, empty_array


@dataclass
class Prior:
    """
    Prior class, need to be inherited.
    """
    size: int = None

    def process_size(self, vecs: List[Any]):
        if self.size is None:
            self.size = max([1] + [len(vec) for vec in vecs if iterable(vec)])

        if self.size <= 0 or not isinstance(self.size, int):
            raise ValueError("Size of the prior must be a positive number.")


@dataclass
class GaussianPrior(Prior):
    mean: np.ndarray = field(default_factory=empty_array, repr=False)
    sd: np.ndarray = field(default_factory=empty_array, repr=False)

    def __post_init__(self):
        self.process_size([self.mean, self.sd])
        self.mean = default_vec_factory(self.mean, self.size, 0.0, vec_name="mean")
        self.sd = default_vec_factory(self.sd, self.size, 0.0, vec_name="sd")

        if any(self.sd <= 0.0):
            raise ValueError("Standard deviation must be all positive.")

    # optimization interface
    def objective(self, var: np.ndarray) -> float:
        return 0.5*np.sum((var - self.mean)**2/self.sd**2)

    def gradient(self, var: np.ndarray) -> np.ndarray:
        return (var - self.mean)/self.sd**2

    # pylint: disable=unused-argument
    def hessian(self, var: np.ndarray) -> np.ndarray:
        return np.diag(1/self.sd**2)


@dataclass
class UniformPrior(Prior):
    lb: np.ndarray = field(default_factory=empty_array, repr=False)
    ub: np.ndarray = field(default_factory=empty_array, repr=False)

    def __post_init__(self):
        self.process_size([self.lb, self.ub])
        self.lb = default_vec_factory(self.lb, self.size, -np.inf, vec_name="lb")
        self.ub = default_vec_factory(self.ub, self.size, np.inf, vec_name="ub")

        if any(self.lb > self.ub):
            raise ValueError("Lower bounds must be less or equal than upper bounds.")


@dataclass
class LinearPrior:
    mat: np.ndarray = field(default_factory=lambda: np.empty(shape=(0, 1)), repr=False)
    size: int = None

    def __post_init__(self):
        if self.size is None:
            self.size = self.mat.shape[0]

        if self.size != self.mat.shape[0]:
            raise ValueError("`mat` and `size` not matching.")

    def is_empty(self) -> bool:
        return self.mat.size == 0


@dataclass
class LinearGaussianPrior(LinearPrior, GaussianPrior):
    def __post_init__(self):
        LinearPrior.__post_init__(self)
        GaussianPrior.__post_init__(self)

    # optimization interface
    def objective(self, var: np.ndarray) -> float:
        trans_var = self.mat.dot(var)
        return super().objective(trans_var)

    def gradient(self, var: np.ndarray) -> np.ndarray:
        trans_var = self.mat.dot(var)
        return self.mat.T.dot(super().gradient(trans_var))

    def hessian(self, var: np.ndarray) -> np.ndarray:
        trans_var = self.mat.dot(var)
        return self.mat.T.dot(super().hessian(trans_var).dot(self.mat))


@dataclass
class LinearUniformPrior(LinearPrior, UniformPrior):
    def __post_init__(self):
        LinearPrior.__post_init__(self)
        UniformPrior.__post_init__(self)
