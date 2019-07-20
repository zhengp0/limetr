# experiment of operate varmat with singular value decomposition
import numpy as np


class VarMat:
    """
    Covariates matrix class with form
        Diag(v_i) + Z_i Diag(gamma) Z_i^T
    with i is refered as the id of the block.

    provide functions
    - varMat
    - invVarMat
    - dot
    - invDot
    - logDet
    - trace

    and variables
    - s_vals
    - s_vecs
    """
    def __init__(self, v, z, gamma, n):
        """
        initialize the object
        """
        # dimensions
        self.m = len(n)
        self.N = sum(n)
        self.k = gamma.size

        # check the dimension and the value of the input
        assert v.shape == (self.N,)
        assert np.all(v > 0.0)

        assert z.shape == (self.N, self.k)

        assert np.all(gamma > 0.0)

        # pass in the variables
        self.v = v
        self.z = z
        self.n = np.array(n)
        self.gamma = gamma

        # processed variables
        self.v_sqrt = np.sqrt(self.v)
        self.gamma_sqrt = np.sqrt(self.gamma)
        self.z_scaled = (self.z*self.gamma_sqrt)/self.v_sqrt.reshape(self.N, 1)

        # reserved variables
        self.n_s = np.minimum(self.n, self.k)
        self.n_u = self.n*self.n_s
        s_vals = np.zeros(s_n.sum())
        s_vecs = np.zeros(self.n.dot(s_n))

    def varMat(self):
        """
        returns the covariate matrix
        """

    def invVarMat(self):
        """
        returns the inverse covariate matrix
        """

    def dot(self, x):
        """
        dot product with the covariate matrix
        """

    def invDot(self, x):
        """
        inverse dot product with the covariate matrix
        """

    def logDet(self):
        """
        returns the log determinant of the covariate matrix
        """
        return np.sum(np.log(self.eig_vals))

    def trace(self):
        """
        returns the trace of the covariate matrix
        """
        return np.sum(self.eig_vals)
