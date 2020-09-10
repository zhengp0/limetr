"""
    distribution
    ~~~~~~~~~~~~

    Distribution module.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Distribution:
    size: int = 1

    def __post_init__(self):
        if self.size <= 0 or not isinstance(self.size, int):
            raise ValueError("Size of the distribution must be a positive integer.")

    def _check_attr(self):
        raise NotImplementedError("Please do not directly use Distribution class.")


@dataclass
class Gaussian(Distribution):
    mean: np.ndarray = None
    sd: np.ndarray = None

    def __post_init__(self):
        super().__post_init__()
        self.mean = np.zeros(self.size) if self.mean is None else self.mean
        self.sd = np.repeat(np.inf, self.size) if self.sd is None else self.sd
        self._check_attr()

    def _check_attr(self):
        assert len(self.mean) == self.size, f"mean should have length {self.size}"
        assert len(self.sd) == self.size, f"sd should have length {self.size}"
        assert all(self.sd > 0.0), "SD should be all positive."


@dataclass
class Uniform(Distribution):
    lb: np.ndarray = None
    ub: np.ndarray = None

    def __post_init__(self):
        super().__post_init__()
        self.lb = np.repeat(-np.inf, self.size) if self.lb is None else self.lb
        self.ub = np.repeat(np.inf, self.size) if self.ub is None else self.ub
        self._check_attr()

    def _check_attr(self):
        assert len(self.lb) == self.size, f"lb should have length {self.size}"
        assert len(self.ub) == self.size, f"ub should have length {self.size}"
        assert all(self.lb <= self.ub), f"lb should be less or equal than ub."


@dataclass
class Laplace(Distribution):
    mean: np.ndarray = None
    sd: np.ndarray = None

    def __post_init__(self):
        super().__post_init__()
        self.mean = np.zeros(self.size) if self.mean is None else self.mean
        self.sd = np.repeat(np.inf, self.size) if self.sd is None else self.sd
        self._check_attr()

    def _check_attr(self):
        assert len(self.mean) == self.size, f"mean should have length {self.size}"
        assert len(self.sd) == self.size, f"sd should have length {self.size}"
        assert all(self.sd > 0.0), "SD should be all positive."
