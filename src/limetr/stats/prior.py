"""
Prior Module
"""
from typing import List, Any
from dataclasses import dataclass, field

import numpy as np

from limetr.utils import default_attr_factory, iterable


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
    mean: List = field(default_factory=list, repr=False)
    sd: List = field(default_factory=list, repr=False)

    def __post_init__(self):
        self.process_size([self.mean, self.sd])
        self.mean = default_attr_factory(self.mean, self.size, 0.0, attr_name="mean")
        self.sd = default_attr_factory(self.sd, self.size, 0.0, attr_name="sd")

        if any(self.sd <= 0.0):
            raise ValueError("Standard deviation must be all positive.")


@dataclass
class UniformPrior(Prior):
    lb: List = field(default_factory=list, repr=False)
    ub: List = field(default_factory=list, repr=False)

    def __post_init__(self):
        self.process_size([self.lb, self.ub])
        self.lb = default_attr_factory(self.lb, self.size, -np.inf, attr_name="lb")
        self.ub = default_attr_factory(self.ub, self.size, np.inf, attr_name="ub")

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


@dataclass
class LinearUniformPrior(LinearPrior, UniformPrior):
    def __post_init__(self):
        LinearPrior.__post_init__(self)
        UniformPrior.__post_init__(self)
