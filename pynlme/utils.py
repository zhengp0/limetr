# utility classes and functions
import numpy as np
from scipy.linalg import block_diag
from pynlme.futils import varmat


class VarMat:
    def __init__(self, v, z, gamma, study_sizes):
        # check the dimension of the input
        assert v.ndim == 1
        assert v.size == sum(study_sizes)
        assert np.all(v > 0.0)

        assert z.ndim == 2

        assert gamma.ndim == 1
        assert gamma.size == z.shape[1]
        assert np.all(gamma >= 0.0)

        assert study_sizes.dtype == int
        assert np.all(study_sizes > 0)

        # pass in variables
        self.v = v
        self.z = np.asfortranarray(z)
        self.gamma = gamma
        self.study_sizes = study_sizes

        self.k_gamma = gamma.size
        self.num_studies = len(study_sizes)
        self.num_data = sum(study_sizes)

    # public functions
    # -------------------------------------------------------------------------
    def varMat(self):
        split_idx = np.cumsum(self.study_sizes)[:-1]
        v_study = np.split(self.v, split_idx)
        z_study = np.split(self.z, split_idx, axis=0)

        diag_blocks = [self._blockVarMat(v_study[i], z_study[i], self.gamma)
                       for i in range(self.num_studies)]

        return block_diag(*diag_blocks)

    def invVarMat(self):
        split_idx = np.cumsum(self.study_sizes)[:-1]
        v_study = np.split(self.v, split_idx)
        z_study = np.split(self.z, split_idx, axis=0)

        diag_blocks = [self._blockInvVarMat(v_study[i], z_study[i], self.gamma)
                       for i in range(self.num_studies)]

        return block_diag(*diag_blocks)

    def dot(self, x):
        if x.ndim == 1:
            return varmat.dot_mv(self.v, self.z, self.gamma,
                                 self.study_sizes, x, self.num_studies,
                                 self.num_data, self.k_gamma)
        if x.ndim == 2:
            x = np.asfortranarray(x)
            return varmat.dot_mm(self.v, self.z, self.gamma,
                                 self.study_sizes, x, self.num_studies,
                                 self.num_data, self.k_gamma, x.shape[1])

    def invDot(self, x):
        if x.ndim == 1:
            return varmat.invdot_mv(self.v, self.z, self.gamma,
                                    self.study_sizes, x, self.num_studies,
                                    self.num_data, self.k_gamma)
        if x.ndim == 2:
            x = np.asfortranarray(x)
            return varmat.invdot_mm(self.v, self.z, self.gamma,
                                    self.study_sizes, x, self.num_studies,
                                    self.num_data, self.k_gamma, x.shape[1])

    def logDet(self):
        return varmat.logdet(self.v, self.z, self.gamma,
                             self.study_sizes, self.num_studies,
                             self.num_data, self.k_gamma)

    @classmethod
    def testProblem(cls):
        num_studies = 5
        study_sizes = np.array([10]*num_studies)
        num_data = np.sum(study_sizes)

        k = 2
        s = np.random.rand(num_data)*0.09 + 0.01
        v = s**2
        z = np.random.randn(num_data, k)
        gamma = np.random.rand(k)*0.1

        return cls(v, z, gamma, study_sizes)

    # internal functions
    # -------------------------------------------------------------------------
    @staticmethod
    def _blockVarMat(v, z, gamma):
        return np.diag(v) + (z*gamma).dot(z.T)

    @staticmethod
    def _blockInvVarMat(v, z, gamma):
        if v.size <= gamma.size:
            return np.linalg.inv(np.diag(v) + (z*gamma).dot(z.T))
        else:
            w = 1.0/v
            y = z*w.reshape(w.size, 1)
            d = np.diag(1.0/gamma) + y.T.dot(z)
            return np.diag(w) - y.dot(np.linalg.inv(d).dot(y.T))
