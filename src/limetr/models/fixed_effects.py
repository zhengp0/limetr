"""
Fixed-Effects Regression Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simple linear regression model.
"""
from typing import Dict

import numpy as np
from limetr.data import Data
from limetr.variable import FeVariable
from numpy import ndarray
from scipy.optimize import LinearConstraint, minimize
from scipy.stats import norm


class FeModel:
    """ Simple linear regression model class that allows observation
    (co)variance matrix.

    Attributes
    ----------
    data : Data
        Data object contains the observations
    fevar : FeVariable
        Fixed effects variable
    inlier_pct : float
        Inlier percentage

    Methods
    -------
    get_residual(beta)
        Compute trimming weighted residual.
    get_femat(beta)
        Compute trimming weighted Jacobian matrix of fixed effects.
    get_obs_varmat()
        Compute trimming weighted observation variance.
    objective(beta)
        Objective function of the optimiation problem
    gradient(beta)
        Gradient function of the optimization problem
    hessian(beta)
        Hessian function of the optimization problem, approximated by the Fisher
        information matrix.
    detect_outliers(beta)
        Detect outlier based on the statistical model and given varaible.
    get_model_init()
        Compute model initialization.
    fit_model(var=None, num_tr_steps=3, options=None)
        Run optimization algorithm to get solution.
    """

    def __init__(self,
                 data: Data,
                 fevar: FeVariable,
                 inlier_pct: float = 1.0):
        """
        Parameters
        ----------
        data : Data
            Data object contains the observations.
        fevar : FeVariable
            Fixed effects variable.
        inlier_pct : float, optional
            Inlier percentage, by default 1

        Raises
        ------
        ValueError
            When fixed effects shape not matching with the data.
        ValueError
            When the inlier percentage is outside zero to one interval.
        """

        if data.num_obs != fevar.mapping.shape[0]:
            raise ValueError("Fixed effects shape not matching with data.")
        if inlier_pct < 0 or inlier_pct > 1:
            raise ValueError("`inlier_pct` must be between 0 and 1.")

        self.data = data
        self.fevar = fevar
        self.inlier_pct = inlier_pct
        self.result = None

    # pylint:disable=unbalanced-tuple-unpacking
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

    def get_obs_varmat(self) -> ndarray:
        """
        Compute trimming weighted observation (co)variance matrix

        Returns
        -------
        ndarray
            Weighted observation (co)variance matrix
        """
        weight = self.data.weight
        sqrt_weight = np.sqrt(weight)
        varmat = self.data.obs_varmat
        varmat_diag = np.diag(varmat)**weight
        varmat = sqrt_weight*varmat*sqrt_weight[:, None]
        np.fill_diagonal(varmat, varmat_diag)
        return varmat

    def objective(self, beta: ndarray) -> float:
        """
        Objective function of the optimization problem.

        Parameters
        ----------
        beta : ndarray

        Returns
        -------
        float
            Objective function value.
        """
        r = self.get_residual(beta)
        d = self.get_obs_varmat()

        val = 0.5*r.dot(np.linalg.solve(d, r))
        val += self.fevar.prior_objective(beta)

        return val

    def gradient(self, beta: ndarray) -> ndarray:
        """
        Gradient function of the optimization problem

        Parameters
        ----------
        beta : ndarray

        Returns
        -------
        ndarray
            Gradient at given variable.
        """
        r = self.get_residual(beta)
        d = self.get_obs_varmat()
        femat = self.get_femat(beta)

        grad_beta = -femat.T.dot(np.linalg.solve(d, r))
        grad_beta += self.fevar.prior_gradient(beta)

        return grad_beta

    def hessian(self, beta: ndarray) -> ndarray:
        """
        Hessian function of the optimization problem, approximated by the Fisher
        information matrix.

        Parameters
        ----------
        beta : ndarray

        Returns
        -------
        ndarray
            Hessian at given variable.
        """
        d = self.get_obs_varmat()
        femat = self.get_femat(beta)

        beta_fisher = femat.T.dot(np.linalg.solve(d, femat))
        beta_fisher += self.fevar.prior_hessian(beta)

        return beta_fisher

    def detect_outliers(self, beta: ndarray) -> ndarray:
        """
        Detect outlier based on the statistical model and given varaible

        Parameters
        ----------
        beta : ndarray

        Returns
        -------
        ndarray
            Indices of outliers.
        """
        r = self.data.obs - self.fevar.mapping(beta)
        s = np.sqrt(np.diag(self.data.obs_varmat))
        a = norm.ppf(0.5 + 0.5*self.inlier_pct)
        return np.abs(r) > a*s

    def get_model_init(self) -> ndarray:
        """
        Get model initializations

        Returns
        -------
        ndarray
            Return the initialization variables.
        """
        beta = np.zeros(self.fevar.size)
        grad_beta = self.gradient(beta)
        hess_beta = self.hessian(beta)
        beta = beta - np.linalg.solve(
            hess_beta + np.identity(self.fevar.size),
            grad_beta
        )
        return beta

    def _fit_model(self,
                   beta: ndarray = None,
                   options: Dict = None):
        """
        (Inner) Fit model function

        Parameters
        ----------
        beta : ndarray, optional
            Initial guess of the variable, by default None
        options : dict, optional
            scipy optimizer options, by default None
        """
        beta = self.get_model_init() if beta is None else beta.copy()

        bounds = self.fevar.get_uprior_info().T
        constraints_mat = self.fevar.get_linear_upriors_mat()
        constraints_vec = self.fevar.get_linear_upriors_info()
        constraints = [LinearConstraint(
            constraints_mat,
            constraints_vec[0],
            constraints_vec[1]
        )] if constraints_mat.size > 0 else []

        self.result = minimize(self.objective, beta,
                               method="trust-constr",
                               jac=self.gradient,
                               hess=self.hessian,
                               constraints=constraints,
                               bounds=bounds,
                               options=options)

    def fit_model(self,
                  beta: ndarray = None,
                  trim_steps: int = 3,
                  options: Dict = None):
        """
        Fit model function

        Parameters
        ----------
        beta : ndarray, optional
            Initial guess of the variable, by default None
        trim_steps : int, optional
            Number of trimming steps, by default 3
        options : Dict, optional
            scipy optimizer options, by default None

        Raises
        ------
        ValueError
            When ``num_tr_steps`` is strictly less than 2.
        """

        trim_steps = int(trim_steps)
        if trim_steps < 2:
            raise ValueError("At least two trimming steps.")

        self._fit_model(beta=beta, options=options)
        if self.inlier_pct < 1.0:
            index = self.detect_outliers(self.result.x)
            if index.sum() > 0:
                for weight in np.linspace(1.0, 0.0, trim_steps)[1:]:
                    self.data.weight.fill(1.0)
                    self.data.weight[index] = weight
                    self._fit_model(beta=self.result.x, options=options)
                    index = self.detect_outliers(self.result.x)

    @property
    def soln(self) -> Dict[str, ndarray]:
        """Solution summary"""
        if self.result is None:
            raise ValueError("Please fit the mdoel first.")
        beta = self.result.x
        beta_sd = 1.0/np.sqrt(np.diag(self.hessian(beta)))
        return {
            "beta": beta,
            "beta_sd": beta_sd,
        }

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(\n"
                f"    data={self.data},\n"
                f"    fevar={self.fevar},\n"
                f"    inlier_pct={self.inlier_pct}\n"
                f")")
