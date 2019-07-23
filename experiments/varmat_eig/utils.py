# experiment of operate varmat with singular value decomposition
import numpy as np
from special_mat import izmat


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
    - diag
    - invDiag
    - logDet

    and variables
    - eig_vals
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
        self.sqrt_v = np.sqrt(self.v)
        self.sqrt_gamma = np.sqrt(self.gamma)
        self.scaled_z = (self.z*self.sqrt_gamma)/self.sqrt_v.reshape(self.N, 1)

        # decomposition of scaled z
        self.scaled_z_ns = np.minimum(self.n, self.k)
        self.scaled_z_nu = self.n*self.scaled_z_ns
        self.scaled_z_s = np.zeros(self.scaled_ns.sum())
        self.scaled_z_u = np.zeros(self.n.dot(self.scaled_z_ns))

        izmat.zdecomp(self.n, self.scaled_z_nu, self.scaled_z_ns,
                      self.scaled_z, self.scaled_z_u, self.scaled_z_s)

        self.scaled_z_nd = self.scaled_z_ns
        self.scaled_z_d = self.scaled_z_s**2

        # inverse and eigenvalues
        self.inv_scaled_z_d = 1.0/(1.0 + self.scaled_z_d) - 1.0
        self.scaled_e = izmat.izeig(self.N, self.n,
                                    self.scaled_z_nd, self.scaled_z_d)


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
        if x.ndim == 1:
            func = izmat.izmv
            sqrt_v = self.sqrt_v
        elif x.ndim == 2:
            func = izmat.izmm
            sqrt_v = self.sqrt_v.reshape(self.N, 1)
        else:
            print('unsupported dim of x')
            return None

        return func(self.scaled_z_nu, self.scaled_z_nd, self.n,
                    self.scaled_z_u, self.scaled_z_d, x*sqrt_v)*sqrt_v

    def invDot(self, x):
        """
        inverse dot product with the covariate matrix
        """
        if x.ndim == 1:
            func = izmat.izmv
            sqrt_v = self.sqrt_v
        elif x.ndim == 2:
            func = izmat.izmm
            sqrt_v = self.sqrt_v.reshape(self.N, 1)
        else:
            print('unsupported dim of x')
            return None

        return func(self.scaled_z_nu, self.scaled_z_nd, self.n,
                    self.scaled_z_u, self.scaled_z_d, x/sqrt_v)/sqrt_v

    def diag(self):
        """
        return the diagonal of the matrix
        """
        scaled_diag = izmat.izdiag(
            self.N, self.scaled_z_nu, self.scaled_z_nd,
            self.n, self.scaled_z_u, self.scaled_z_d
            )

        return scaled_diag*self.v


    def invDiag(self):
        """
        return the diagonal of the inverse covariate matrix
        """
        inv_scaled_diag = izmat.izdiag(
            self.N, self.scaled_z_nu, self.scaled_z_nd,
            self.n, self.scaled_z_u, self.inv_scaled_z_d
            )

        return inv_scaled_diag/self.v

    def logDet(self):
        """
        returns the log determinant of the covariate matrix
        """
        return np.sum(np.log(self.eig_vals)) + np.sum(np.log(self.v))

    @classmethod
    def testProblem(cls):
        n = [3, 4, 5]
        N = sum(n)
        k = 4

        v = np.random.rand(N) + 1e-2
        z = np.random.randn(N, k)
        gamma = np.random.rand(k)

        return cls(v, z, gamma, n)
