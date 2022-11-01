# nonlinear mixed effects model
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import block_diag
from scipy.optimize import LinearConstraint, minimize
from spmat.dlmat import BDLMat, DLMat

from limetr import utils


class LimeTr:
    def __init__(self, n, k_beta, k_gamma, Y, F, JF, Z, S,
                 C=None, JC=None, c=None,
                 H=None, JH=None, h=None,
                 uprior=None, gprior=None, lprior=None,
                 certain_inlier_id=None,
                 inlier_percentage=1.0):
        """
        Create LimeTr object, for general mixed effects model

        Parameters
        ----------
        n : ndarray
            study sizes, n[i] is the number of observation for ith study.
        k_beta : int
            dimension of beta
        k_gamma : int
            dimension of gamma
        Y : ndarray
            study observations
        F : function
            return the predict observations given beta
        JF : function
            return the jacobian function of F
        Z : ndarray
            covariates matrix for the random effect
        S : optional, ndarray
            observation standard deviation
        """
        # pass in the dimension
        self.n = np.array(n)
        self.m = len(n)
        self.N = sum(n)
        self.k_beta = k_beta
        self.k_gamma = k_gamma

        self.k = self.k_beta + self.k_gamma
        self.k_total = self.k

        self.idx_beta = slice(0, self.k_beta)
        self.idx_gamma = slice(self.k_beta, self.k_beta + self.k_gamma)
        self.idx_split = np.cumsum(np.insert(n, 0, 0))[:-1]

        # pass in the data
        self.Y = Y
        self.F = F
        self.JF = JF
        self.Z = Z
        self.S = S
        self.V = S**2

        # pass in the priors
        self.use_constraints = (C is not None)
        self.use_regularizer = (H is not None)
        self.use_uprior = (uprior is not None)
        self.use_gprior = (gprior is not None)
        self.use_lprior = (lprior is not None)

        self.C = C
        self.JC = JC
        self.c = c
        if self.use_constraints:
            self.constraints = C
            self.jacobian = JC
            self.num_constraints = C(np.zeros(self.k)).size
            self.cl = c[0]
            self.cu = c[1]
        else:
            self.num_constraints = 0
            self.cl = []
            self.cu = []

        self.H = H
        self.JH = JH
        self.h = h
        if self.use_regularizer:
            self.num_regularizer = H(np.zeros(self.k)).size
            self.hm = self.h[0]
            self.hw = 1.0/self.h[1]**2
        else:
            self.num_regularizer = 0

        if self.use_uprior:
            self.uprior = uprior
        else:
            self.uprior = np.array([
                [-np.inf]*self.k_beta + [0.0]*self.k_gamma,
                [np.inf]*self.k
            ])
            self.use_uprior = True

        self.lb = self.uprior[0]
        self.ub = self.uprior[1]

        if self.use_gprior:
            self.gprior = gprior
            self.gm = gprior[0]
            self.gw = 1.0/gprior[1]**2

        if self.use_lprior:
            self.lprior = lprior
            self.lm = lprior[0]
            self.lw = np.sqrt(2.0)/lprior[1]

            # double dimension pass into ipopt
            self.k_total += self.k

            # extend the constraints matrix
            if self.use_constraints:
                def constraints(x):
                    v = x[:self.k]
                    v_abs = x[self.k:]

                    vec1 = C(v)
                    vec2 = np.hstack((v_abs - (v - self.lm),
                                      v_abs + (v - self.lm)))

                    return np.hstack((vec1, vec2))

                def jacobian(x):
                    v = x[:self.k]
                    v_abs = x[self.k:]
                    Id = np.eye(self.k)

                    mat1 = JC(v)
                    mat2 = np.block([[-Id, Id], [Id, Id]])

                    return np.vstack((mat1, mat2))
            else:
                def constraints(x):
                    v = x[:self.k]
                    v_abs = x[self.k:]

                    vec = np.hstack((v_abs - v, v_abs + v))

                    return vec

                def jacobian(x):
                    Id = np.eye(self.k)
                    mat = np.block([[-Id, Id], [Id, Id]])

                    return mat

            self.num_constraints += 2*self.k
            self.constraints = constraints
            self.jacobian = jacobian
            self.cl = np.hstack((self.cl, np.zeros(2*self.k)))
            self.cu = np.hstack((self.cu, np.repeat(np.inf, 2*self.k)))

            # extend the regularizer matrix
            if self.use_regularizer:
                def H_new(x):
                    v = x[:self.k]

                    return H(v)

                def JH_new(x):
                    v = x[:self.k]

                    return np.hstack((JH(v),
                                      np.zeros((self.num_regularizer,
                                                self.k))))

                self.H = H_new
                self.JH = JH_new

            # extend Uniform priors
            if self.use_uprior:
                uprior_abs = np.array([[0.0]*self.k, [np.inf]*self.k])
                self.uprior = np.hstack((self.uprior, uprior_abs))
                self.lb = self.uprior[0]
                self.ub = self.uprior[1]

        # trimming option
        self.use_trimming = (0.0 < inlier_percentage < 1.0)
        self.certain_inlier_id = certain_inlier_id
        self.inlier_percentage = inlier_percentage
        self.num_inliers = np.floor(inlier_percentage*self.N)
        self.num_outliers = self.N - self.num_inliers
        self.w = np.repeat(self.num_inliers/self.N, self.N)

        if self.certain_inlier_id is not None:
            self.certain_inlier_id = np.unique(self.certain_inlier_id)
            self.active_trimming_id = np.array(
                [i
                 for i in range(self.N)
                 if i not in self.certain_inlier_id])
        else:
            self.active_trimming_id = None

        # specify solution to be None
        self.soln = None
        self.info = None
        self.beta = np.zeros(self.k_beta)
        self.gamma = np.repeat(0.01, self.k_gamma)

        # check the input
        self.check()

    def check(self):
        assert self.Y.shape == (self.N,)
        assert self.Z.shape == (self.N, self.k_gamma)
        if self.S is not None:
            assert self.S.shape == (self.N,)
            assert np.all(self.S > 0.0)

        if self.use_constraints:
            assert self.c.shape == (2, self.num_constraints)
            assert np.all(self.cl <= self.cu)

        if self.use_regularizer:
            assert self.h.shape == (2, self.num_regularizer)
            assert np.all(self.h[1] > 0.0)

        if self.use_uprior:
            assert np.all(self.lb <= self.ub)

        if self.use_gprior:
            assert np.all(self.gprior[1] > 0.0)

        assert 0.0 < self.inlier_percentage <= 1.0
        if self.use_trimming and self.certain_inlier_id is not None:
            assert isinstance(self.certain_inlier_id, np.ndarray)
            assert self.certain_inlier_id.dtype == int
            assert self.certain_inlier_id.ndim == 1
            assert self.certain_inlier_id.size <= self.num_inliers
            assert np.min(self.certain_inlier_id) >= 0
            assert np.max(self.certain_inlier_id) < self.N

        if self.k > self.N:
            print('Warning: information insufficient!')

    def _get_vars(self, x: NDArray) -> tuple[NDArray, NDArray]:
        beta, gamma = x[self.idx_beta], x[self.idx_gamma]
        gamma = np.maximum(0.0, gamma)
        return beta, gamma

    def _get_nll_components(self, beta: NDArray) -> tuple[NDArray, ...]:
        # trimming option
        if self.use_trimming:
            sqrt_w = np.sqrt(self.w)
            sqrt_W = sqrt_w.reshape(self.N, 1)
            F_beta = self.F(beta)*sqrt_w
            JF_beta = self.JF(beta)*sqrt_W
            Y = self.Y*sqrt_w
            Z = self.Z*sqrt_W
        else:
            F_beta = self.F(beta)
            JF_beta = self.JF(beta)
            Y = self.Y
            Z = self.Z

        return F_beta, JF_beta, Y, Z

    def objective(self, x, use_ad=False):
        # unpack variable
        beta, gamma = self._get_vars(x)

        # trimming option
        F_beta, _, Y, Z = self._get_nll_components(beta)

        # residual and variance
        R = Y - F_beta

        val = 0.5*self.N*np.log(2.0*np.pi)

        if use_ad:
            # should only use when testing
            split_idx = np.cumsum(self.n)[:-1]
            v_study = np.split(self.V, split_idx)
            z_study = np.split(Z, split_idx, axis=0)
            D = block_diag(*[np.diag(v) + (z*gamma).dot(z.T)
                             for v, z in zip(v_study, z_study)])
            inv_D = np.linalg.inv(D)

            val += 0.5*np.log(np.linalg.det(D))
            val += 0.5*R.dot(inv_D.dot(R))
        else:
            D = BDLMat(diags=self.V, lmats=Z*np.sqrt(gamma), dsizes=self.n)
            val += 0.5*D.logdet()
            val += 0.5*R.dot(D.invdot(R))

        # add gpriors
        if self.use_regularizer:
            val += 0.5*self.hw.dot((self.H(x) - self.hm)**2)

        if self.use_gprior:
            val += 0.5*self.gw.dot((x[:self.k] - self.gm)**2)

        if self.use_lprior:
            val += self.lw.dot(x[self.k:])

        return val

    def gradient(self, x, use_ad=False, eps=1e-12):
        if use_ad:
            # should only use when testing
            g = np.zeros(self.k_total)
            z = x + 0j
            for i in range(self.k_total):
                z[i] += eps*1j
                g[i] = self.objective(z, use_ad=use_ad).imag/eps
                z[i] -= eps*1j

            return g

        # unpack variable
        beta, gamma = self._get_vars(x)

        # trimming option
        F_beta, JF_beta, Y, Z = self._get_nll_components(beta)

        # residual and variance
        R = Y - F_beta
        D = BDLMat(diags=self.V, lmats=Z*np.sqrt(gamma), dsizes=self.n)

        # gradient for beta
        DR = D.invdot(R)
        g_beta = -JF_beta.T.dot(DR)

        # gradient for gamma
        DZ = D.invdot(Z)
        g_gamma = 0.5*np.sum(Z*DZ, axis=0) - \
            0.5*np.sum(
                np.add.reduceat(DZ.T*R, self.idx_split, axis=1)**2,
                axis=1)

        g = np.hstack((g_beta, g_gamma))

        # add gradient from the regularizer
        if self.use_regularizer:
            g += self.JH(x).T.dot((self.H(x) - self.hm)*self.hw)

        # add gradient from the gprior
        if self.use_gprior:
            g += (x[:self.k] - self.gm)*self.gw

        # add gradient from the lprior
        if self.use_lprior:
            g = np.hstack((g, self.lw))

        return g

    def hessian(self, x: NDArray) -> NDArray:
        beta, gamma = self._get_vars(x)
        _, JF_beta, _, Z = self._get_nll_components(beta)

        sqrt_gamma = np.sqrt(gamma)
        d = BDLMat(diags=self.V, lmats=Z*np.sqrt(gamma), dsizes=self.n)

        split_idx = np.cumsum(self.n)[:-1]
        v_study = np.split(self.V, split_idx)
        z_study = np.split(Z, split_idx, axis=0)
        dlmats = [DLMat(v, z*sqrt_gamma) for v, z in zip(v_study, z_study)]

        beta_fisher = JF_beta.T.dot(d.invdot(JF_beta))
        gamma_fisher = np.zeros((self.k_gamma, self.k_gamma))
        for i, dlmat in enumerate(dlmats):
            gamma_fisher += 0.5*(z_study[i].T.dot(dlmat.invdot(z_study[i])))**2

        hessian = block_diag(beta_fisher, gamma_fisher)
        if self.use_regularizer:
            JH = self.JH(x)
            hessian += (JH.T*self.hw).dot(JH)

        if self.use_gprior:
            idx = np.arange(self.k)
            hessian[idx, idx] += self.gw

        return hessian

    def objectiveTrimming(self, w):
        t = (self.Z**2).dot(self.gamma)
        r = self.Y - self.F(self.beta)
        d = self.V + t

        val = 0.5*np.sum(r**2*w/d)
        val += 0.5*self.N*np.log(2.0*np.pi) + 0.5*w.dot(np.log(d))

        return val

    def gradientTrimming(self, w, use_ad=False, eps=1e-10):
        if use_ad:
            # only use when testing
            g = np.zeros(self.N)
            z = w + 0j
            for i in range(self.N):
                z[i] += eps*1j
                g[i] = self.objectiveTrimming(z).imag/eps
                z[i] -= eps*1j

            return g

        t = (self.Z**2).dot(self.gamma)
        r = (self.Y - self.F(self.beta))**2
        d = self.V + t

        g = 0.5*r/d
        g += 0.5*np.log(d)

        return g

    def optimize(self, x0=None, options=None):
        if x0 is None:
            x0 = np.hstack((self.beta, self.gamma))
            if self.use_lprior:
                x0 = np.hstack((x0, np.zeros(self.k)))

        assert x0.size == self.k_total

        constraints = [LinearConstraint(
            self.JC(),
            self.cl,
            self.cu
        )] if self.JC is not None else []
        self.info = minimize(
            self.objective,
            x0,
            method="trust-constr",
            jac=self.gradient,
            hess=self.hessian,
            constraints=constraints,
            bounds=self.uprior.T,
            options=options
        )

        self.soln = self.info.x
        self.beta, self.gamma = self._get_vars(self.soln)

    def fitModel(self, x0=None,
                 inner_options=None,
                 outer_verbose=False,
                 outer_max_iter=100,
                 outer_step_size=1.0,
                 outer_tol=1e-6,
                 normalize_trimming_grad=False):

        if not self.use_trimming:
            self.optimize(x0=x0, options=inner_options)

            return self.beta, self.gamma, self.w

        self.soln = x0

        num_iter = 0
        err = outer_tol + 1.0

        while err >= outer_tol:
            self.optimize(x0=self.soln, options=inner_options)

            w_grad = self.gradientTrimming(self.w)
            if normalize_trimming_grad:
                w_grad /= np.linalg.norm(w_grad)
            w_new = utils.proj_capped_simplex(
                self.w - outer_step_size*w_grad,
                self.num_inliers,
                active_id=self.active_trimming_id)

            err = np.linalg.norm(w_new - self.w)/outer_step_size
            np.copyto(self.w, w_new)

            num_iter += 1

            if outer_verbose:
                obj = self.objectiveTrimming(self.w)
                print('iter %4d, obj %8.2e, err %8.2e' % (num_iter, obj, err))

            if num_iter >= outer_max_iter:
                print('reach max outer iter')
                break

        return self.beta, self.gamma, self.w

    def estimateRE(self):
        """
        estimate random effect after fitModel
        """
        if self.soln is None:
            print('Please fit the model first.')
            return None

        if self.use_trimming:
            R = (self.Y - self.F(self.beta))*np.sqrt(self.w)
            Z = self.Z*(np.sqrt(self.w).reshape(self.N, 1))
        else:
            R = self.Y - self.F(self.beta)
            Z = self.Z

        iV = 1.0/self.V
        iVZ = Z*iV.reshape(iV.size, 1)

        r = np.split(R, np.cumsum(self.n)[:-1])
        v = np.split(self.V, np.cumsum(self.n)[:-1])
        z = np.split(Z, np.cumsum(self.n)[:-1], axis=0)
        ivz = np.split(iVZ, np.cumsum(self.n)[:-1], axis=0)

        u = []
        for i in range(self.m):
            rhs = ivz[i].T.dot(r[i])
            tmp = z[i]*self.gamma
            mat = np.diag(v[i]) + tmp.dot(z[i].T)
            vec = self.gamma*rhs - tmp.T.dot(np.linalg.solve(mat, tmp.dot(rhs)))
            u.append(vec)

        self.u = np.vstack(u)

        return self.u

    def get_re_vcov(self):
        if self.soln is None:
            print('Please fit the model first.')
            return None

        Z = self.Z
        if self.use_trimming:
            Z = self.Z*(np.sqrt(self.w).reshape(self.N, 1))

        v = np.split(self.V, np.cumsum(self.n)[:-1])
        z = np.split(Z, np.cumsum(self.n)[:-1], axis=0)

        vcov = []
        for i in range(self.m):
            hessian = (z[i].T/v[i]).dot(z[i]) + np.diag(1.0/self.gamma)
            vcov.append(np.linalg.pinv(hessian))

        return vcov

    def estimate_re(self,
                    beta: np.ndarray = None,
                    gamma: np.ndarray = None,
                    use_gamma: bool = True) -> np.ndarray:
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma
        r = np.split(self.Y - self.F(beta), np.cumsum(self.n)[:-1])
        z = np.split(self.Z, np.cumsum(self.n)[:-1], axis=0)
        v = np.split(self.S**2, np.cumsum(self.n)[:-1])

        u = []
        for i in range(self.m):
            rhs = (z[i].T/v[i]).dot(r[i])
            if use_gamma:
                q = (z[i].T/v[i]).dot(z[i])*gamma + np.identity(self.k_gamma)
                u.append(gamma[:, None]*np.linalg.inv(q).dot(rhs))
            else:
                q = (z[i].T/v[i]).dot(z[i])
                u.append(np.linalg.inv(q).dot(rhs))

        return np.vstack(u)

    def get_gamma_fisher(self, gamma: np.ndarray) -> np.ndarray:
        z = np.split(self.Z, np.cumsum(self.n)[:-1], axis=0)
        v = np.split(self.S**2, np.cumsum(self.n)[:-1])
        H = np.zeros((self.k_gamma, self.k_gamma))
        for i in range(self.m):
            q = np.diag(v[i]) + (z[i]*gamma).dot(z[i].T)
            q = z[i].T.dot(np.linalg.inv(q).dot(z[i]))
            H += 0.5*(q**2)
        return H

    def sample_beta(self, size: int = 1) -> NDArray:
        hessian = self.hessian(self.soln)
        beta_hessian = hessian[:self.k_beta, :self.k_beta]
        beta_vcov = np.linalg.inv(beta_hessian)
        return np.random.multivariate_normal(self.beta, beta_vcov, size=size)

    @classmethod
    def testProblem(cls,
                    use_trimming=False,
                    use_constraints=False,
                    use_regularizer=False,
                    use_uprior=False,
                    use_gprior=False):
        m = 10
        n = [5]*m
        N = sum(n)
        k_beta = 3
        k_gamma = 2
        k = k_beta + k_gamma

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

        # constraints, regularizer and priors
        if use_constraints:
            M = np.ones((1, k))

            def C(x, M=M):
                return M.dot(x)

            def JC(x, M=M):
                return M

            c = np.array([[0.0], [1.0]])
        else:
            C, JC, c = None, None, None

        if use_regularizer:
            M = np.ones((1, k))

            def H(x, M=M):
                return M.dot(x)

            def JH(x, M=M):
                return M

            h = np.array([[0.0], [2.0]])
        else:
            H, JH, h = None, None, None

        if use_uprior:
            uprior = np.array([[0.0]*k, [np.inf]*k])
        else:
            uprior = None

        if use_gprior:
            gprior = np.array([[0.0]*k, [2.0]*k])
        else:
            gprior = None

        if use_trimming:
            inlier_percentage = 0.9
        else:
            inlier_percentage = 1.0

        return cls(n, k_beta, k_gamma, Y, F, JF, Z, S=S,
                   C=C, JC=JC, c=c,
                   H=H, JH=JH, h=h,
                   uprior=uprior, gprior=gprior,
                   inlier_percentage=inlier_percentage)

    @classmethod
    def testProblemLasso(cls):
        m = 100
        n = [1]*m
        N = sum(n)
        k_beta = 150
        k_gamma = 1
        k = k_beta + k_gamma

        beta_t = np.zeros(k_beta)
        beta_t[np.random.choice(k_beta, 5)] = np.sign(np.random.randn(5))
        gamma_t = np.zeros(k_gamma)

        X = np.random.randn(N, k_beta)
        Z = np.ones((N, k_gamma))
        Y = X.dot(beta_t)
        S = np.repeat(1.0, N)

        weight = 0.1*np.linalg.norm(X.T.dot(Y), np.inf)

        def F(beta):
            return X.dot(beta)

        def JF(beta):
            return X

        uprior = np.array([[-np.inf]*k_beta + [0.0], [np.inf]*k_beta + [0.0]])
        lprior = np.array([[0.0]*k, [np.sqrt(2.0)/weight]*k])

        return cls(n, k_beta, k_gamma, Y, F, JF, Z, S=S,
                   uprior=uprior, lprior=lprior)


def get_baseline_model(model: LimeTr):
    n = np.copy(model.n)
    k_beta = 1
    k_gamma = 1
    Y = model.Y.copy()
    intercept = np.ones((Y.size, 1))
    def F(beta): return intercept.dot(beta)
    def JF(beta): return intercept
    Z = intercept
    S = model.S.copy()
    w = model.w.copy()

    baseline_model = LimeTr(n, k_beta, k_gamma, Y, F, JF, Z, S=S)
    baseline_model.optimize()
    return baseline_model


def get_fe_pred(model: LimeTr):
    return model.F(model.beta)


def get_re_pred(model: LimeTr):
    re = model.estimateRE()
    return np.sum(model.Z*np.repeat(re, model.n, axis=0), axis=1)


def get_varmat(model: LimeTr):
    S = model.S**model.w
    Z = model.Z*np.sqrt(model.w)[:, None]
    n = model.n
    gamma = model.gamma
    return BDLMat(diags=S**2, lmats=Z*np.sqrt(gamma), dsizes=n)


def get_marginal_rvar(model: LimeTr):
    residual = (model.Y - get_fe_pred(model))*np.sqrt(model.w)
    varmat = get_varmat(model)
    return residual.dot(varmat.invdot(residual))/model.w.sum()


def get_conditional_rvar(model: LimeTr):
    residual = (model.Y - get_fe_pred(model) - get_re_pred(model))*np.sqrt(model.w)
    varvec = (model.S**model.w)**2
    return (residual/varvec).dot(residual)/model.w.sum()


def get_marginal_R2(model: LimeTr,
                    baseline_model: LimeTr = None) -> float:
    if baseline_model is None:
        baseline_model = get_baseline_model(model)

    return 1 - get_marginal_rvar(model)/get_marginal_rvar(baseline_model)


def get_conditional_R2(model: LimeTr,
                       baseline_model: LimeTr = None):
    if baseline_model is None:
        baseline_model = get_baseline_model(model)

    return 1 - get_conditional_rvar(model)/get_conditional_rvar(baseline_model)


def get_R2(model: LimeTr):
    baseline_model = get_baseline_model(model)
    return {
        "conditional_R2": get_conditional_R2(model, baseline_model),
        "marginal_R2": get_marginal_R2(model, baseline_model)
    }


def get_rmse(model: LimeTr):
    return np.sqrt(get_marginal_rvar(model))
