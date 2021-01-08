"""
    model
    ~~~~~

    Main model module.
"""
from typing import Tuple

import numpy as np
from numpy import ndarray
from scipy.linalg import block_diag
from scipy.optimize import LinearConstraint, minimize
from spmat import BDLMat

from limetr.data import Data
from limetr.utils import get_varmat, split_by_sizes
from limetr.variable import FeVariable, ReVariable


class LimeTr:
    """
    LimeTr model class

    Attributes
    ----------
    data : Data
        Data object contains the observations
    fevar : FeVariable
        Fixed effects variable
    revar : ReVariable
        Random effects variable
    inlier_pct : float
        Inlier percentage

    Methods
    -------
    get_vars(var)
        Split array into variables beta and gamma.
    get_residual(beta)
        Compute trimming weighted residual.
    get_femat(beta)
        Compute trimming weighted Jacobian matrix of fixed effects.
    get_remat()
        Compute trimming weighted design matrix of random effects.
    get_obsvar()
        Compute trimming weighted observation variance.
    get_varmat(gamma)
        Compute trimming weighted variance covariance matrix of the likelihood
    objective(var)
        Objective function of the optimiation problem
    gradient(var)
        Gradient function of the optimization problem
    hessian(var)
        Hessian function of the optimization problem, approximated by the Fisher
        information matrix.
    fit_model(var, options)
        Run optimization algorithm to get solution.
    """

    def __init__(self,
                 data: Data,
                 fevar: FeVariable,
                 revar: ReVariable,
                 inlier_pct: float = 1.0):
        """
        Parameters
        ----------
        data : Data
            Data object contains the observations.
        fevar : FeVariable
            Fixed effects variable.
        revar : ReVariable
            Random effects variable.
        inlier_pct : float, optional
            Inlier percentage, by default 1

        Raises
        ------
        ValueError
            When fixed effects shape not matching with the data.
        ValueError
            When random effects shape not matching with the data.
        ValueError
            When the inlier percentage is outside zero to one interval.
        """

        if data.num_obs != fevar.mapping.shape[0]:
            raise ValueError("Fixed effects shape not matching with data.")
        if data.num_obs != revar.mapping.shape[0]:
            raise ValueError("Random effects shape not matching with data.")
        if inlier_pct < 0 or inlier_pct > 1:
            raise ValueError("`inlier_pct` must be between 0 and 1.")

        self.data = data
        self.fevar = fevar
        self.revar = revar
        self.inlier_pct = inlier_pct
        self.result = None

    # pylint:disable=unbalanced-tuple-unpacking
    def get_vars(self, var: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Split array into variables beta and gamma.

        Parameters
        ----------
        var : ndarray

        Returns
        -------
        Tuple[ndarray, ndarray]
            Return beta and gamma.
        """
        beta, gamma = tuple(split_by_sizes(var, [self.fevar.size,
                                                 self.revar.size]))
        gamma[gamma < 0] = 0.0
        return beta, gamma

    def get_residual(self, beta: ndarray) -> ndarray:
        """
        Compute trimming weighted residual

        Parameters
        ----------
        beta : ndarray

        Returns
        -------
        ndarray
            Weighted residual
        """
        return self.data.weight*(self.data.obs -
                                 self.fevar.mapping(beta))

    def get_femat(self, beta: ndarray) -> ndarray:
        """
        Compute trimming weighted Jacobian matrix of fixed effects.

        Parameters
        ----------
        beta : ndarray

        Returns
        -------
        ndarray
            Weighted Jacobian matrix of fixed effects.
        """
        return self.data.weight[:, None]*self.fevar.mapping.jac(beta)

    def get_remat(self) -> ndarray:
        """
        Compute trimming weighted design matrix of random effects.

        Returns
        -------
        ndarray
            Weighted design matrix of random effects.
        """
        return self.data.weight[:, None]*self.revar.mapping.mat

    def get_obsvar(self) -> ndarray:
        """
        Compute trimming weighted observation variance

        Returns
        -------
        ndarray
            Weighted observation variance
        """
        return self.data.obs_se**(2*self.data.weight)

    def get_varmat(self, gamma: ndarray) -> BDLMat:
        """
        Compute trimming weighted variance covariance matrix of the likelihood

        Parameters
        ----------
        gamma : ndarray

        Returns
        -------
        BDLMat
            Weighted variance covariance matrix of the likelihood.
        """
        return get_varmat(gamma, self.get_obsvar(), self.get_remat())

    def objective(self, var: ndarray) -> float:
        """
        Objective function of the optimization problem.

        Parameters
        ----------
        var : ndarray

        Returns
        -------
        float
            Objective function value.
        """
        beta, gamma = self.get_vars(var)
        r = self.get_residual(beta)
        d = self.get_varmat(gamma)

        val = 0.5*(d.logdet() + r.dot(d.invdot(r)))
        val += self.fevar.prior_objective(beta)
        val += self.revar.prior_objective(gamma)

        return val

    def gradient(self, var: ndarray) -> ndarray:
        """
        Gradient function of the optimization problem

        Parameters
        ----------
        var : ndarray

        Returns
        -------
        ndarray
            Gradient at given variable.
        """
        beta, gamma = self.get_vars(var)
        r = self.get_residual(beta)
        d = self.get_varmat(gamma)
        femat = self.get_femat(beta)
        remat = self.get_remat()

        dr = d.invdot(r)
        split_index = np.cumsum(np.insert(self.data.group_sizes, 0, 0))[:-1]

        grad_beta = -femat.T.dot(dr) + self.fevar.prior_gradient(beta)
        grad_gamma = 0.5*(
            np.sum(remat*(d.invdot(remat)), axis=0) -
            np.sum(np.add.reduceat(remat.T*dr, split_index, axis=1)**2, axis=1)
        ) + self.revar.prior_gradient(gamma)

        return np.hstack([grad_beta, grad_gamma])

    def hessian(self, var: ndarray) -> ndarray:
        """
        Hessian function of the optimization problem, approximated by the Fisher
        information matrix.

        Parameters
        ----------
        var : ndarray

        Returns
        -------
        ndarray
            Hessian at given variable.
        """
        beta, gamma = self.get_vars(var)
        d = self.get_varmat(gamma)
        femat = self.get_femat(beta)
        remat = split_by_sizes(self.get_remat(), self.data.group_sizes)

        beta_fisher = femat.T.dot(d.invdot(femat))
        beta_fisher += self.fevar.prior_hessian(beta)

        gamma_fisher = np.zeros((self.revar.size, self.revar.size))
        for i, dlmat in enumerate(d.dlmats):
            gamma_fisher += 0.5*(remat[i].T.dot(dlmat.invdot(remat[i])))**2
        gamma_fisher += self.revar.prior_hessian(gamma)

        return block_diag(beta_fisher, gamma_fisher)

    def fit_model(self,
                  var: ndarray = None,
                  options: dict = None):
        """
        Fit model function

        Parameters
        ----------
        var : ndarray, optional
            Initial guess of the variable, by default None
        options : dict, optional
            scipy optimizer options, by default None
        """
        var = np.zeros(self.fevar.size + self.revar.size) if var is None else var

        bounds = np.hstack([self.fevar.get_uprior_info(),
                            self.revar.get_uprior_info()]).T
        constraints_mat = block_diag(self.fevar.get_linear_upriors_mat(),
                                     self.revar.get_linear_upriors_mat())
        constraints_vec = np.hstack([self.fevar.get_linear_upriors_info(),
                                     self.revar.get_linear_upriors_info()])
        constraints = [LinearConstraint(
            constraints_mat,
            constraints_vec[0],
            constraints_vec[1]
        )] if constraints_mat.size > 0 else []

        self.result = minimize(self.objective, var,
                               method="trust-constr",
                               jac=self.gradient,
                               hess=self.hessian,
                               constraints=constraints,
                               bounds=bounds,
                               options=options)
