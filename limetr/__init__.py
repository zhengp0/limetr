# nonlinear mixed effects model
import numpy as np
import ipopt
from limetr.utils import VarMat


class LimeTr:
    def __init__(self, n, k_beta, k_gamma, Y, F, JF, Z, S,
                 lin_uprior_mat=None, lin_gprior_mat=None,
                 lin_uprior_val=None, lin_gprior_val=None,
                 dir_uprior_val=None, dir_gprior_val=None):
        # pass in the dimension
        self.n = np.array(n)
        self.m = len(n)
        self.N = sum(n)
        self.k_beta = k_beta
        self.k_gamma = k_gamma
        self.k = self.k_beta + self.k_gamma

        self.idx_beta = slice(0, self.k_beta)
        self.idx_gamma = slice(self.k_beta, self.k)
        self.idx_split = np.cumsum(np.insert(n, 0, 0))[:-1]

        # pass in the data
        self.Y = Y
        self.F = F
        self.JF = JF
        self.Z = Z
        self.S = S
        self.V = S**2

        # pass in the priors
        self.use_lin_uprior = (lin_uprior_val is not None)
        self.use_lin_gprior = (lin_gprior_val is not None)
        self.use_dir_uprior = (dir_uprior_val is not None)
        self.use_dir_gprior = (dir_gprior_val is not None)

        if self.use_lin_uprior:
            self.lin_uprior_mat = lin_uprior_mat
            self.lin_uprior_val = lin_uprior_val
            self.num_lin_uprior = lin_uprior_mat.shape[0]
        else:
            self.num_lin_uprior = 0

        if self.use_lin_gprior:
            self.lin_gprior_mat = lin_gprior_mat
            self.lin_gprior_val = lin_gprior_val
            self.num_lin_gprior = lin_gprior_mat.shape[0]
        else:
            self.num_lin_gprior = 0

        if self.use_dir_uprior:
            self.dir_uprior_val = dir_uprior_val
        else:
            self.dir_uprior_val = np.array([
                [-np.inf]*self.k_beta + [0.0]*self.k_gamma,
                [np.inf]*self.k_beta + [np.inf]*self.k_gamma
                ])
            self.use_dir_uprior = True

        if self.use_dir_gprior:
            self.dir_gprior_val = dir_gprior_val

        # check the input
        self.check()

    def check(self):
        assert self.Y.shape == (self.N,)
        assert self.S.shape == (self.N,)
        assert self.Z.shape == (self.N, self.k_gamma)
        assert np.all(self.S > 0.0)

        if self.use_lin_uprior:
            assert self.lin_uprior_mat.shape == (self.num_lin_uprior, self.k)
            assert self.lin_uprior_val.shape == (2, self.num_lin_uprior)
            assert np.all(self.lin_uprior_val[0] <= self.lin_uprior_val[1])

        if self.use_lin_gprior:
            assert self.lin_gprior_mat.shape == (self.num_lin_gprior, self.k)
            assert self.lin_gprior_val.shape == (2, self.num_lin_gprior)
            assert np.all(self.lin_gprior_val[1] > 0.0)

        if self.use_dir_uprior:
            assert self.dir_uprior_val.shape == (2, self.k)
            assert np.all(self.dir_uprior_val[0] <= self.dir_uprior_val[1])
            assert np.all(self.dir_uprior_val[0, self.idx_gamma] >= 0.0)

        if self.use_dir_gprior:
            assert self.dir_gprior_val.shape == (2, self.k)
            assert np.all(self.dir_gprior_val[1] > 0.0)

    def objective(self, x, use_ad=False):
        # unpack variable
        beta = x[self.idx_beta]
        gamma = x[self.idx_gamma]

        # residual and variance
        R = self.Y - self.F(beta)
        D = VarMat(self.V, self.Z, gamma, self.n)

        val = 0.5*self.N*np.log(2.0*np.pi)

        if use_ad:
            # should only use when testing
            varmat = D
            D = varmat.varMat()
            inv_D = varmat.invVarMat()

            val += 0.5*np.log(np.linalg.det(D))
            val += 0.5*R.dot(inv_D.dot(R))
        else:
            val += 0.5*D.logDet()
            val += 0.5*R.dot(D.invDot(R))

        # add gpriors
        if self.use_lin_gprior:
            val += 0.5*np.sum(
                ((self.lin_gprior_mat.dot(x) - self.lin_gprior_val[0])/
                 self.lin_gprior_val[1])**2
                )

        if self.use_dir_gprior:
            val += 0.5*np.sum(
                ((x - self.dir_gprior_val[0])/self.dir_gprior_val[1])**2
                )

        return val

    def gradient(self, x, use_ad=False, eps=1e-12):
        if use_ad:
            # should only use when testing
            g = np.zeros(self.k)
            z = x + 0j
            for i in range(self.k):
                z[i] += eps*1j
                g[i] = self.objective(z, use_ad=use_ad).imag/eps
                z[i] -= eps*1j

            return g

        # unpack variable
        beta = x[self.idx_beta]
        gamma = x[self.idx_gamma]

        # residual and variance
        R = self.Y - self.F(beta)
        D = VarMat(self.V, self.Z, gamma, self.n)

        # gradient for beta
        g_beta = -self.JF(beta).T.dot(D.invDot(R))

        # gradient for gamma
        DZ = D.invDot(self.Z)
        g_gamma = 0.5*np.sum(self.Z*DZ, axis=0) - 0.5*np.sum(
            np.add.reduceat(DZ.T*R, self.idx_split, axis=1)**2,
            axis=1)

        g = np.hstack((g_beta, g_gamma))

        return g


    def optimize(self, x0=None, print_level=0, max_iter=100):
        if x0 is None:
            x0 = np.zeros(self.k)

        assert x0.size == self.k

        if self.use_lin_uprior:
            opt_problem = ipopt.problem(
                n=self.k,
                m=self.num_lin_uprior,
                problem_obj=self,
                lb=self.dir_uprior_val[0],
                ub=self.dir_uprior_val[1],
                cl=self.lin_uprior_val[0],
                cu=self.lin_uprior_val[1]
                )
        else:
            opt_problem = ipopt.problem(
                n=self.k,
                m=0,
                problem_obj=self,
                lb=self.dir_uprior_val[0],
                ub=self.dir_uprior_val[1]
                )


        opt_problem.addOption('print_level', print_level)
        opt_problem.addOption('max_iter', max_iter)

        soln, info = opt_problem.solve(x0)
        self.soln = soln
        self.info = info
        self.beta_soln = soln[self.idx_beta]
        self.gamma_soln = soln[self.idx_gamma]

    @classmethod
    def testProblem(cls):
        m = 10
        n = [5]*m
        N = sum(n)
        k_beta = 3
        k_gamma = 2

        beta_t = np.random.randn(k_beta)
        gamma_t = np.random.rand(k_gamma)*0.09 + 0.01

        X = np.random.randn(N, k_beta)
        Z = np.random.randn(N, k_gamma)

        S = np.random.rand(N)*0.09 + 0.01
        V = S**2
        D = np.diag(V) + (Z*gamma_t).dot(Z.T)

        U = np.random.multivariate_normal(np.zeros(N), D)
        E = np.random.randn(N)*S

        Y = X.dot(beta_t) + U + E

        def F(beta, X=X):
            return X.dot(beta)

        def JF(beta, X=X):
            return X

        return cls(n, k_beta, k_gamma, Y, F, JF, Z, S)
