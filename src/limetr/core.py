"""
    model
    ~~~~~

    Main model module.
"""
from typing import Tuple
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from spmat import BDLMat
from limetr.variable import FeVariable, ReVariable
from limetr.data import Data
from limetr.utils import split_by_sizes, get_varmat


class LimeTr:
    """
    LimeTr model class
    """

    def __init__(self,
                 data: Data,
                 fevar: FeVariable,
                 revar: ReVariable,
                 inlier_pct: float = 1.0):

        self.data = data
        self.fevar = fevar
        self.revar = revar
        self.inlier_pct = inlier_pct

        self.check_attr()

        self.result = None

    def check_attr(self):
        # check size
        if self.data.num_obs != self.fevar.mapping.shape[0]:
            raise ValueError("Fixed effects shape not matching data.")
        if self.data.num_obs != self.revar.mapping.shape[0]:
            raise ValueError("Random effects shape not matching data.")
        # check inlier percentage
        if self.inlier_pct < 0 or self.inlier_pct > 1:
            raise ValueError("`inlier_pct` must be between 0 and 1.")

    def get_vars(self, var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        variables = split_by_sizes(var, [self.fevar.size, self.revar.size])
        beta = variables[0]
        gamma = variables[1]
        gamma[gamma < 0] = 0.0
        return beta, gamma

    def get_residual(self, beta: np.ndarray, split: bool = False) -> np.ndarray:
        residual = self.data.weight*(self.data.obs - self.fevar.mapping(beta))
        if split:
            residual = split_by_sizes(residual, self.data.group_sizes)
        return residual

    def get_femat(self, beta: np.ndarray, split: bool = False) -> np.ndarray:
        femat = self.data.weight[:, None]*self.fevar.mapping.jac(beta)
        if split:
            femat = split_by_sizes(femat, self.data.group_sizes)
        return femat

    def get_remat(self, split: bool = True) -> np.ndarray:
        remat = self.data.weight[:, None]*self.revar.mapping.mat
        if split:
            remat = split_by_sizes(remat, self.data.group_sizes)
        return remat

    def get_obsvar(self, split: bool = True) -> np.ndarray:
        obsvar = self.data.obs_se**(2*self.data.weight)
        if split:
            obsvar = split_by_sizes(obsvar, self.data.group_sizes)
        return obsvar

    def get_varmat(self, gamma) -> BDLMat:
        return get_varmat(gamma, self.get_obsvar(), self.get_remat())

    def get_beta_fisher(self,
                        beta: np.ndarray,
                        gamma: np.ndarray = None,
                        d: BDLMat = None) -> np.ndarray:
        d = self.get_varmat(gamma) if d is None else d
        femat = self.get_femat(beta, split=False)
        return femat.T.dot(d.invdot(femat))

    def get_gamma_fisher(self,
                         gamma: np.ndarray = None,
                         d: BDLMat = None) -> np.ndarray:
        d = self.get_varmat(gamma) if d is None else d
        remat = self.get_remat(split=True)
        gamma_fisher = np.zeros((self.revar.size, self.revar.size))
        for i in range(self.data.num_groups):
            gamma_fisher += 0.5*(remat[i].T.dot(d.dlmats[i].invdot(remat[i])))**2
        return gamma_fisher

    def objective(self, var: np.ndarray) -> float:
        beta, gamma = self.get_vars(var)
        r = self.get_residual(beta)
        d = self.get_varmat(gamma)

        val = 0.5*(d.logdet() + r.dot(d.invdot(r)))
        val += self.fevar.prior_objective(beta)
        val += self.revar.prior_objective(gamma)

        return val

    def gradient(self, var: np.ndarray) -> np.ndarray:
        beta, gamma = self.get_vars(var)
        r = self.get_residual(beta)
        d = self.get_varmat(gamma)
        femat = self.get_femat(beta, split=False)
        remat = self.get_remat(split=False)

        dr = d.invdot(r)
        split_index = np.cumsum(np.insert(self.data.group_sizes, 0, 0))[:-1]

        grad_beta = -femat.T.dot(dr) + self.fevar.prior_gradient(beta)
        grad_gamma = 0.5*(
            np.sum(remat*(d.invdot(remat)), axis=0) -
            np.sum(np.add.reduceat(remat.T*dr, split_index, axis=1)**2, axis=1)
        ) + self.revar.prior_gradient(gamma)

        return np.hstack([grad_beta, grad_gamma])

    def hessian(self, var: np.ndarray) -> np.ndarray:
        beta, gamma = self.get_vars(var)
        d = self.get_varmat(gamma)

        hess_beta = self.get_beta_fisher(beta, d=d) + \
            self.fevar.prior_hessian(beta)
        hess_gamma = self.get_gamma_fisher(d=d) + \
            self.revar.prior_hessian(gamma)

        return block_diag(hess_beta, hess_gamma)

    def fit_model(self,
                  var: np.ndarray = None,
                  options: dict = None):
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
