"""
Variable Module
"""
from typing import List, Union
from dataclasses import dataclass, field
import numpy as np
from limetr.linalg import SmoothMapping, LinearMapping
from limetr.stats import (Prior,
                          GaussianPrior,
                          UniformPrior,
                          LinearGaussianPrior,
                          LinearUniformPrior)


@dataclass
class Variable:
    mapping: SmoothMapping
    priors: List[Prior] = field(default_factory=list, repr=False)
    name: str = field(default="unknown")

    gprior: GaussianPrior = field(default=None, init=False, repr=False)
    uprior: UniformPrior = field(default=None, init=False, repr=False)
    linear_gpriors: List[LinearGaussianPrior] = field(default_factory=list, init=False, repr=False)
    linear_upriors: List[LinearUniformPrior] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        self.process_priors()

    @property
    def size(self) -> int:
        return self.mapping.shape[1]

    def process_priors(self, priors: List[Prior] = None):
        priors = self.priors if priors is None else priors
        for prior in priors:
            if isinstance(prior, LinearGaussianPrior):
                self.linear_gpriors.append(prior)
                if prior.mat.shape[1] != self.size:
                    raise ValueError("Linear Gaussian prior size not matching.")
            elif isinstance(prior, LinearUniformPrior):
                self.linear_upriors.append(prior)
                if prior.mat.shape[1] != self.size:
                    raise ValueError("Linear Uniform prior size not matching.")
            elif isinstance(prior, GaussianPrior):
                if self.gprior is not None and self.gprior != prior:
                    raise ValueError("Can only provide one Gaussian prior.")
                self.gprior = prior
                if prior.size != self.size:
                    raise ValueError("Gaussian prior size not matching.")
            elif isinstance(prior, UniformPrior):
                if self.uprior is not None and self.uprior != prior:
                    raise ValueError("Can only provide one Uniform prior.")
                self.uprior = prior
                if prior.size != self.size:
                    raise ValueError("Uniform prior size not matching.")

    def reset_priors(self):
        self.gprior = None
        self.uprior = None
        self.linear_gpriors = list()
        self.linear_upriors = list()

    def add_priors(self, priors: Union[Prior, List[Prior]]):
        if not isinstance(priors, list):
            priors = [priors]
        self.priors.extend(priors)
        self.process_priors(priors)

    def get_prior_objective(self, var: np.ndarray) -> float:
        val = 0.0 if self.gprior is None else self.gprior.objective(var)
        for prior in self.linear_gpriors:
            val += prior.objective(var)
        return val

    def get_prior_gradient(self, var: np.ndarray) -> np.ndarray:
        val = np.zeros(self.size) if self.gprior is None else \
            self.gprior.gradient(var)
        for prior in self.linear_gpriors:
            val += prior.gradient(var)
        return val

    def get_prior_hessian(self, var: np.ndarray) -> np.ndarray:
        val = np.zeros((self.size, self.size)) if self.gprior is None else \
            self.gprior.hessian(var)
        for prior in self.linear_gpriors:
            val += prior.hessian(var)
        return val

    def get_uvec(self) -> np.ndarray:
        if self.uprior is None:
            vec = np.array([[-np.inf]*self.size, [np.inf]*self.size])
        else:
            vec = np.vstack([self.uprior.lb, self.uprior.ub])
        return vec

    def get_linear_umat(self) -> np.ndarray:
        if not self.linear_upriors:
            umat = np.empty((0, self.size))
        else:
            umat = np.vstack([
                prior.mat for prior in self.linear_upriors
            ])
        return umat

    def get_linear_uvec(self) -> np.ndarray:
        if not self.linear_upriors:
            uvec = np.empty((2, 0))
        else:
            uvec = np.hstack([
                np.vstack([prior.lb, prior.ub])
                for prior in self.linear_upriors
            ])
        return uvec


@dataclass
class FEVariable(Variable):
    name: str = "fixed effects"


@dataclass
class REVariable(Variable):
    name: str = "random effects"

    def __post_init__(self):
        if not isinstance(self.mapping, LinearMapping):
            raise ValueError("Random effect design mapping has to be linear.")
        super().__post_init__()

    def process_priors(self, priors: List[Prior] = None):
        super().process_priors(priors)
        self.check_uprior()

    def check_uprior(self):
        if self.uprior is None:
            self.add_priors(UniformPrior(lb=0.0, ub=np.inf, size=self.size))
        else:
            if any(self.uprior.lb < 0):
                print(self.uprior.lb)
                raise ValueError("Random effects variance must have positive lower bounds.")
