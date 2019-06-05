# utility classes and functions
import numpy as np
from scipy.linalg import block_diag
from limetr.futils import varmat


class VarMat:
    def __init__(self, v, z, gamma, study_sizes):
        # pass in the dimensions
        self.k_gamma = gamma.size
        self.num_studies = len(study_sizes)
        self.num_data = sum(study_sizes)

        # check the dimension of the input
        assert v.shape == (self.num_data,)
        assert np.all(v > 0.0)

        assert z.shape[1] == self.k_gamma
        self.z_rank_1 = (z.shape[0] == self.num_studies)
        if not self.z_rank_1:
            assert z.shape[0] == self.num_data

        assert gamma.shape == (self.k_gamma,)
        assert np.all(gamma >= 0.0)

        assert study_sizes.dtype == int
        assert np.all(study_sizes > 0)

        # pass in variables
        self.v = v
        self.z = np.asfortranarray(z)
        self.gamma = gamma
        self.study_sizes = study_sizes

    # public functions
    # -------------------------------------------------------------------------
    def varMat(self):
        split_idx = np.cumsum(self.study_sizes)[:-1]
        v_study = np.split(self.v, split_idx)
        if not self.z_rank_1:
            z_study = np.split(self.z, split_idx, axis=0)
        else:
            z_study = self.z

        diag_blocks = [self._blockVarMat(v_study[i], z_study[i], self.gamma)
                       for i in range(self.num_studies)]

        return block_diag(*diag_blocks)

    def invVarMat(self):
        split_idx = np.cumsum(self.study_sizes)[:-1]
        v_study = np.split(self.v, split_idx)
        if not self.z_rank_1:
            z_study = np.split(self.z, split_idx, axis=0)
        else:
            z_study = self.z

        diag_blocks = [self._blockInvVarMat(v_study[i], z_study[i], self.gamma)
                       for i in range(self.num_studies)]

        return block_diag(*diag_blocks)

    def dot(self, x):
        if x.ndim == 1:
            if self.z_rank_1:
                return varmat.dot_mv_rank_1(
                    self.v, self.z, self.gamma,
                    self.study_sizes, x, self.num_studies,
                    self.num_data, self.k_gamma
                    )
            else:
                return varmat.dot_mv(
                    self.v, self.z, self.gamma,
                    self.study_sizes, x, self.num_studies,
                    self.num_data, self.k_gamma
                    )
        if x.ndim == 2:
            x = np.asfortranarray(x)
            if self.z_rank_1:
                return varmat.dot_mm_rank_1(
                    self.v, self.z, self.gamma,
                    self.study_sizes, x, self.num_studies,
                    self.num_data, self.k_gamma, x.shape[1]
                    )
            else:
                return varmat.dot_mm(
                    self.v, self.z, self.gamma,
                    self.study_sizes, x, self.num_studies,
                    self.num_data, self.k_gamma, x.shape[1]
                    )

    def invDot(self, x):
        if x.ndim == 1:
            if self.z_rank_1:
                return varmat.invdot_mv_rank_1(
                    self.v, self.z, self.gamma,
                    self.study_sizes, x, self.num_studies,
                    self.num_data, self.k_gamma
                    )
            else:
                return varmat.invdot_mv(
                    self.v, self.z, self.gamma,
                    self.study_sizes, x, self.num_studies,
                    self.num_data, self.k_gamma
                    )
        if x.ndim == 2:
            x = np.asfortranarray(x)
            if self.z_rank_1:
                return varmat.invdot_mm_rank_1(
                    self.v, self.z, self.gamma,
                    self.study_sizes, x, self.num_studies,
                    self.num_data, self.k_gamma, x.shape[1]
                    )
            else:
                return varmat.invdot_mm(
                    self.v, self.z, self.gamma,
                    self.study_sizes, x, self.num_studies,
                    self.num_data, self.k_gamma, x.shape[1]
                    )

    def logDet(self):
        if self.z_rank_1:
            return varmat.logdet_rank_1(
                self.v, self.z, self.gamma,
                self.study_sizes, self.num_studies,
                self.num_data, self.k_gamma
                )
        else:
            return varmat.logdet(
                self.v, self.z, self.gamma,
                self.study_sizes, self.num_studies,
                self.num_data, self.k_gamma
                )

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

    @classmethod
    def testProblemRank1(cls):
        num_studies = 5
        study_sizes = np.array([10]*num_studies)
        num_data = np.sum(study_sizes)

        k = 2
        s = np.random.rand(num_data)*0.09 + 0.01
        v = s**2
        z = np.random.randn(num_studies, k)
        gamma = np.random.rand(k)*0.1

        return cls(v, z, gamma, study_sizes)

    # internal functions
    # -------------------------------------------------------------------------
    @staticmethod
    def _blockVarMat(v, z, gamma):
        return np.diag(v) + (z*gamma).dot(z.T)

    @staticmethod
    def _blockInvVarMat(v, z, gamma):
        return np.linalg.inv(np.diag(v) + (z*gamma).dot(z.T))
