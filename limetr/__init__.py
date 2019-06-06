# nonlinear mixed effects model
import numpy as np
import ipopt
from limetr.utils import VarMat


class LimeTr:
    def __init__(self, n, k_beta, k_gamma, Y, F, JF, Z, S,
                 C=None, JC=None, c=None, H=None, JH=None, h=None,
                 uprior=None, gprior=None):
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

        # if z for each study is rank 1
        self.z_rank_1 = (Z.shape[0] == self.m)
        if self.z_rank_1:
            self.Z_full = np.repeat(Z, n, axis=0)
        else:
            self.Z_full = Z

        # pass in the priors
        self.use_constraints = (C is not None)
        self.use_regularizer = (H is not None)
        self.use_uprior = (uprior is not None)
        self.use_gprior = (gprior is not None)

        if self.use_constraints:
            self.constraints = C
            self.jacobian = JC
            self.num_constraints = C(np.zeros(self.k)).size
            self.cl = c[0]
            self.cu = c[1]
        else:
            self.num_constraints = 0
            self.cl = None
            self.cu = None

        if self.use_regularizer:
            self.h = h
            self.H = H
            self.JH = JH
            self.num_regularizer = H(np.zeros(self.k)).size

        if self.use_uprior:
            self.uprior = uprior
        else:
            self.uprior = np.array([
                [-np.inf]*self.k_beta + [0.0]*self.k_gamma,
                [np.inf]*self.k_beta + [np.inf]*self.k_gamma
                ])
            self.use_uprior = True

        self.lb = self.uprior[0]
        self.ub = self.uprior[1]

        if self.use_gprior:
            self.gprior = gprior

        # check the input
        self.check()

    def check(self):
        assert self.Y.shape == (self.N,)
        assert self.S.shape == (self.N,)
        if self.z_rank_1:
            assert self.Z.shape == (self.m, self.k_gamma)
        else:
            assert self.Z.shape == (self.N, self.k_gamma)
        assert np.all(self.S > 0.0)

        if self.use_constraints:
            assert np.all(self.cl[0] <= self.cu[1])

        if self.use_regularizer:
            assert self.h.shape == (2, self.num_regularizer)
            assert np.all(self.h[1] > 0.0)

        if self.use_uprior:
            assert np.all(self.lb <= self.ub)

        if self.use_gprior:
            assert self.gprior.shape == (2, self.k)
            assert np.all(self.gprior[1] > 0.0)

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
        if self.use_regularizer:
            val += 0.5*np.sum(((self.H(x) - self.h[0])/self.h[1])**2)

        if self.use_gprior:
            val += 0.5*np.sum(((x - self.gprior[0])/self.gprior[1])**2)

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
        DZ = D.invDot(self.Z_full)

        g_gamma = 0.5*np.sum(self.Z_full*DZ, axis=0) - 0.5*np.sum(
            np.add.reduceat(DZ.T*R, self.idx_split, axis=1)**2,
            axis=1)

        g = np.hstack((g_beta, g_gamma))

        # add gradient from the regularizer
        if self.use_regularizer:
            g += self.JH(x).T.dot((self.H(x) - self.h[0])/self.h[1]**2)

        # add the gradient from the gprior
        if self.use_gprior:
            g += (x - self.gprior[0])/self.gprior[1]**2

        return g

    def optimize(self, x0=None, print_level=0, max_iter=100):
        if x0 is None:
            x0 = np.zeros(self.k)

        assert x0.size == self.k

        opt_problem = ipopt.problem(
            n=self.k,
            m=self.num_constraints,
            problem_obj=self,
            lb=self.uprior[0],
            ub=self.uprior[1],
            cl=self.cl,
            cu=self.cu
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
