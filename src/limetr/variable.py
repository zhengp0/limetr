"""
Variable Module
"""
from typing import Any, Iterable, Type

import numpy as np
from numpy import ndarray

from limetr.linalg import LinearMapping, SmoothMapping
from limetr.stats import (GaussianPrior, LinearGaussianPrior, LinearPrior,
                          LinearUniformPrior, Prior, UniformPrior)


class Variable:
    """
    Variable class, contains mapping and prior information

    Attributes
    ----------
    mapping : SmoothMapping
        Smooth mapping, map the variable to predict the data.
    priors : Iterable[Prior], optional
        Priors for the variable.
    name : Any
        Name of the variable.
    gprior : GaussianPrior
        Gaussian prior of the variable.
    uprior : UniformPrior
        Uniform prior of the variable.
    linear_gpriors : List[LinearGaussianPrior]
        List of linear Gaussian priors.
    linear_upriors : List[LinearUniformPrior]
        List of linear Uniform priors.

    Methods
    -------
    update_gprior(prior)
        Update the Gaussian prior.
    update_uprior(prior)
        Update the Uniform prior.
    update_linear_gpriors(prior)
        Update linear Gaussian prior, append to the list.
    update_linear_upriors(prior)
        Update linear Uniform prior, append to the list.
    update_priors(priors)
        Update all priors providing a list of priors.
    reset_priors()
        Reset priors to default settings.
    prior_objective(var)
        Objective function from Gaussian prior and linear Gaussian priors.
    prior_gradient(var)
        Gradient function from Gaussian prior and linear Gaussian priors.
    prior_hessian(var)
        Hessian function from Gaussian prior and linear Gaussian priors.
    get_uprior_info()
        Return Uniform prior upper and lower bounds.
    get_linear_upriors_mat()
        Return linear Uniform prior linear mapping
    get_linear_upriors_info()
        Return linear Uniform upper and lower bounds.
    """

    def __init__(self,
                 mapping: SmoothMapping,
                 priors: Iterable[Prior] = (),
                 name: Any = "unknown"):
        """
        Parameters
        ----------
        mapping : SmoothMapping
            Smooth mapping, map the variable to predict the data.
        priors : Iterable[Prior], optional
            Priors for the variable, by default tuple with zero length.
        name : Any, optional
            Name of the variable, default by ``'unknown'``.

        Raises
        ------
        TypeError
            If ``mapping`` is not ``SmoothMapping``.
        """
        if not isinstance(mapping, SmoothMapping):
            raise TypeError("`mapping` has to be SmoothMapping.")
        self.mapping = mapping
        self.name = name

        self.gprior = GaussianPrior(size=self.size)
        self.uprior = UniformPrior(size=self.size)
        self.linear_gpriors = []
        self.linear_upriors = []
        self.update_priors(priors)

    @property
    def size(self) -> int:
        """Size of the variable"""
        return self.mapping.shape[1]

    def _validate_prior(self,
                        prior: Prior,
                        prior_type: Type) -> Prior:
        if not isinstance(prior, prior_type):
            raise TypeError(f"`prior` must be type {prior_type.__name__}.")
        size_matching = prior.mat.shape[1] == self.size \
            if issubclass(prior_type, LinearPrior) else prior.size == self.size
        if not size_matching:
            raise ValueError("`prior` size not match with variable.")

        return prior

    def update_gprior(self, prior: GaussianPrior):
        """
        Update Gaussian prior.

        Parameters
        ----------
        prior : GaussianPrior
        """
        self.gprior = self._validate_prior(prior, GaussianPrior)

    def update_uprior(self, prior: UniformPrior):
        """
        Update Uniform prior

        Parameters
        ----------
        prior : UniformPrior
        """
        self.uprior = self._validate_prior(prior, UniformPrior)

    def update_linear_gpriors(self, prior: LinearGaussianPrior):
        """
        Update linear Gaussian priors

        Parameters
        ----------
        prior : LinearGaussianPrior
        """
        self.linear_gpriors.append(self._validate_prior(
            prior, LinearGaussianPrior
        ))

    def update_linear_upriors(self, prior: LinearUniformPrior):
        """
        Update linear Uniform priors

        Parameters
        ----------
        prior : LinearUniformPrior
        """
        self.linear_upriors.append(self._validate_prior(
            prior, LinearUniformPrior
        ))

    def update_priors(self, priors: Iterable[Prior]):
        """
        Update priors

        Parameters
        ----------
        priors : Iterable[Prior]

        Raises
        ------
        TypeError
            When prior type is not recognizable.
        """
        for prior in priors:
            if isinstance(prior, LinearGaussianPrior):
                self.update_linear_gpriors(prior)
            elif isinstance(prior, LinearUniformPrior):
                self.update_linear_upriors(prior)
            elif isinstance(prior, GaussianPrior):
                self.update_gprior(prior)
            elif isinstance(prior, UniformPrior):
                self.update_uprior(prior)
            else:
                raise TypeError(f"Unrecognize prior type {type(prior).__name__}")

    def reset_priors(self):
        """
        Reset prior to default settings
        """
        self.gprior = GaussianPrior(size=self.size)
        self.uprior = UniformPrior(size=self.size)
        self.linear_gpriors = []
        self.linear_upriors = []

    def prior_objective(self, var: ndarray) -> float:
        """
        Objective function from Gaussian prior and linear Gaussian priors.

        Parameters
        ----------
        var : ndarray
            Variables.

        Returns
        -------
        float
            Objective function value.
        """
        val = self.gprior.objective(var)
        for prior in self.linear_gpriors:
            val += prior.objective(var)
        return val

    def prior_gradient(self, var: ndarray) -> ndarray:
        """
        Gradient function from Gaussian prior and linear Gaussian priors.

        Parameters
        ----------
        var : ndarray
            Variables.

        Returns
        -------
        ndarray
            Gradient at given value.
        """
        val = self.gprior.gradient(var)
        for prior in self.linear_gpriors:
            val += prior.gradient(var)
        return val

    def prior_hessian(self, var: ndarray) -> ndarray:
        """
        Hessian function from Gaussian prior and linear Gaussian priors.

        Parameters
        ----------
        var : ndarray
            Variables.

        Returns
        -------
        ndarray
            Hessian at given value.
        """
        val = self.gprior.hessian(var)
        for prior in self.linear_gpriors:
            val += prior.hessian(var)
        return val

    def get_uprior_info(self) -> ndarray:
        """
        Get Uniform prior information

        Returns
        -------
        ndarray
            Lower and upper bounds of the prior.
        """
        if self.uprior is None:
            uprior = UniformPrior(size=self.size)
        else:
            uprior = self.uprior
        return uprior.info

    def get_linear_upriors_mat(self) -> ndarray:
        """
        Get linear Uniform prior linear mapping

        Returns
        -------
        ndarray
            Linear mapping of the prior.
        """
        if len(self.linear_upriors) == 0:
            mat = np.empty((0, self.size))
        else:
            mat = np.vstack([prior.mat for prior in self.linear_upriors])
        return mat

    def get_linear_upriors_info(self) -> ndarray:
        """
        Get linear Unform prior information

        Returns
        -------
        ndarray
            Lower and upper bounds of the prior.
        """
        if len(self.linear_upriors) == 0:
            info = np.empty((2, 0))
        else:
            info = np.hstack([prior.info for prior in self.linear_upriors])
        return info

    def __repr__(self) -> str:
        return f"Variable(name={self.name}, size={self.size})"


class FeVariable(Variable):
    """
    Fixed effects variable.
    """

    def __init__(self,
                 mapping: SmoothMapping,
                 priors: Iterable[Prior] = (),
                 name: Any = "fixed effects"):
        """
        Parameters
        ----------
        name : Any, optional
            Name of the variable, by default "fixed effects"
        """
        super().__init__(mapping, priors, name=name)

    def __repr__(self) -> str:
        return f"FeVariable(size={self.size})"


class ReVariable(Variable):
    """
    Random effects variable.
    """

    def __init__(self,
                 mapping: SmoothMapping,
                 priors: Iterable[Prior] = (),
                 name: Any = "random effects"):
        """
        Parameters
        ----------
        name : Any, optional
            Name of the variable, by default "random effects"
        """
        if not isinstance(mapping, LinearMapping):
            raise TypeError("Random effect design mapping has to be linear.")

        priors = list(priors)
        if not any([isinstance(prior, UniformPrior) for prior in priors]):
            priors.append(UniformPrior(lb=0.0, ub=np.inf, size=mapping.mat.shape[1]))

        super().__init__(mapping, priors, name=name)

    def update_uprior(self, prior: UniformPrior):
        prior = self._validate_prior(prior, UniformPrior)
        if any(prior.lb < 0):
            raise ValueError("Random effects variance must have non-negative lower bounds.")
        self.uprior = prior

    def reset_priors(self):
        self.gprior = GaussianPrior(size=self.size)
        self.uprior = UniformPrior(lb=0.0, ub=np.inf, size=self.size)
        self.linear_gpriors = []
        self.linear_upriors = []

    def __repr__(self) -> str:
        return f"ReVariable(size={self.size})"
