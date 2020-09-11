"""
    distribution
    ~~~~~~~~~~~~

    Distribution module.
"""
from typing import Any, List
from dataclasses import dataclass, field
import numpy as np
from limetr.utils import isiterable, default_attr_factory


@dataclass
class Distribution:
    size: int = field(default=None, repr=False)

    def process_size(self, attrs: List[Any]):
        if self.size is None:
            sizes = [len(attr) for attr in attrs if isiterable(attr)]
            sizes.append(1)
            self.size = max(sizes)

        if self.size <= 0 or not isinstance(self.size, int):
            raise ValueError("Size of the distribution must be a positive integer.")


@dataclass
class Gaussian(Distribution):
    mean: np.ndarray = None
    sd: np.ndarray = None

    def __post_init__(self):
        self.process_size([self.mean, self.sd])
        self.mean = default_attr_factory(self.mean, self.size, 0.0, attr_name='mean')
        self.sd = default_attr_factory(self.sd, self.size, np.inf, attr_name='sd')
        assert all(self.sd > 0.0), "Standard deviation must be all positive."


@dataclass
class Uniform(Distribution):
    lb: np.ndarray = None
    ub: np.ndarray = None

    def __post_init__(self):
        self.process_size([self.lb, self.ub])
        self.lb = default_attr_factory(self.lb, self.size, -np.inf, attr_name='lb')
        self.ub = default_attr_factory(self.ub, self.size, np.inf, attr_name='ub')
        assert all(self.lb <= self.ub), "Lower bounds must be less or equal than upper bounds."


@dataclass
class Laplace(Distribution):
    mean: np.ndarray = None
    sd: np.ndarray = None

    def __post_init__(self):
        self.process_size([self.mean, self.sd])
        self.mean = default_attr_factory(self.mean, self.size, 0.0, attr_name='mean')
        self.sd = default_attr_factory(self.sd, self.size, np.inf, attr_name='sd')
        assert all(self.sd > 0.0), "Standard deviation must be all positive."
