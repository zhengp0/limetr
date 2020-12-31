"""
    model
    ~~~~~

    Main model module.
"""
from typing import Tuple, List
import numpy as np
from limetr.linalg import SquareBlockDiagMat
from limetr.variable import FEVariable, REVariable
from limetr.data import Data
from limetr.utils import sizes_to_slices


class LimeTr:
    """
    LimeTr model class
    """

    def __init__(self,
                 data: Data,
                 fevar: FEVariable,
                 revar: REVariable,
                 inlier_pct: float = 1.0):

        self.data = data
        self.fevar = fevar
        self.revar = revar
        self.inlier_pct = inlier_pct

        self.check_attr()

        self.sizes = [self.fevar.size, self.revar.size]
        self.size = sum(self.sizes)
        self.slices = sizes_to_slices(self.sizes)

    def check_attr(self):
        # check size
        if self.data.num_obs != self.fevar.mapping.shape[0]:
            raise ValueError("Fixed effects shape not matching data.")
        if self.data.num_obs != self.revar.mapping.shape[0]:
            raise ValueError("Random effects shape not matching data.")
        # check inlier percentage
        if self.inlier_pct < 0 or self.inlier_pct > 1:
            raise ValueError("`inlier_pct` must be between 0 and 1.")

    def split_var(self, var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return var[self.slices[0]], var[self.slices[1]]

    def split_data(self, data: np.ndarray, axis: int = 0) -> List[np.ndarray]:
        return np.split(data, np.cumsum(self.data.group_sizes)[:-1], axis=axis)

    def objective(self, var: np.ndarray) -> float:
        beta, gamma = self.split_var(var)
        gamma[gamma <= 0.0] = 0.0

        w = self.data.weights
        r = w*(self.data.obs - self.fevar.mapping(beta))
        z = self.split_data(w[:, None]*self.revar.mapping.mat)
        v = self.split_data(self.data.obs_se**(2*w))

        d = SquareBlockDiagMat([(z[i]*gamma).dot(z[i].T) + np.diag(v[i])
                                for i in range(self.data.num_groups)])

        val = 0.5*(d.logdet() + r.dot(d.invdot(r)))

        # TODO: add objective from prior

        return val

    def gradient(self, var: np.ndarray) -> np.ndarray:
        # TODO: add gradient
        return np.zeros(self.size)
