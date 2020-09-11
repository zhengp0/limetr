"""
    model
    ~~~~~

    Main model module.
"""
# nonlinear mixed effects model
import numpy as np
import ipopt
from copy import deepcopy
from limetr.linalg import SquareBlockDiagMat, SmoothMapping, LinearMapping
from limetr.optim import project_to_capped_simplex


class LimeTr:
    def __init__(self,
                 n: np.ndarray,
                 Y: np.ndarray,
                 F: SmoothMapping,
                 Z: np.ndarray,
                 S: np.ndarray,
                 C: SmoothMapping = None, c=None,
                 H: SmoothMapping = None, h=None,
                 uprior=None, gprior=None, lprior=None,
                 certain_inlier_id=None,
                 inlier_percentage=1.0):
        # pass in the dimension
        self.n = np.array(n)
        self.m = len(n)
        self.N = sum(n)
        self.k_beta = F.shape[1]
        self.k_gamma = Z.shape[1]

        self.k = self.k_beta + self.k_gamma
        self.k_total = self.k

        self.idx_beta = slice(0, self.k_beta)
        self.idx_gamma = slice(self.k_beta, self.k_beta + self.k_gamma)
        self.idx_split = np.cumsum(np.insert(n, 0, 0))[:-1]

        # pass in the data
        self.Y = Y
        self.F = F
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
        self.c = c
        if self.use_constraints:
            self.constraints = self.C.fun
            self.jacobian = self.C.jac_fun
            self.num_constraints = self.C.shape[0]
            self.cl = c[0]
            self.cu = c[1]
        else:
            self.num_constraints = 0
            self.cl = []
            self.cu = []

        self.H = H
        self.h = h
        if self.use_regularizer:
            self.num_regularizer = self.H.shape[0]
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

                    vec1 = self.C.fun(v)
                    vec2 = np.hstack((v_abs - (v - self.lm),
                                      v_abs + (v - self.lm)))

                    return np.hstack((vec1, vec2))

                def jacobian(x):
                    v = x[:self.k]
                    v_abs = x[self.k:]
                    Id = np.eye(self.k)

                    mat1 = self.C.jac_fun(v)
                    mat2 = np.block([[-Id, Id], [Id, Id]])

                    return np.vstack((mat1, mat2))
            else:
                def constraints(x):
                    v = x[:self.k]
                    v_abs = x[self.k:]

                    vec = np.hstack((v_abs - v, v_abs + v))

                    return vec

                def jacobian(x):
                    v = x[:self.k]
                    v_abs = x[self.k:]
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

                    return H.fun(v)

                def JH_new(x):
                    v = x[:self.k]

                    return np.hstack((H.jac_fun(v),
                                      np.zeros((self.num_regularizer,
                                                self.k))))

                self.H = SmoothMapping((self.num_regularizer, self.k_total),
                                       H_new, JH_new)

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

    def objective(self, x, use_ad=False):
        # unpack variable
        beta = x[self.idx_beta]
        gamma = x[self.idx_gamma]

        gamma[gamma <= 0.0] = 0.0

        # trimming option
        if self.use_trimming:
            sqrt_w = np.sqrt(self.w)
            sqrt_W = sqrt_w.reshape(self.N, 1)
            F_beta = self.F.fun(beta)*sqrt_w
            Y = self.Y*sqrt_w
            Z = self.Z*sqrt_W
            V = self.V**self.w
        else:
            F_beta = self.F.fun(beta)
            Y = self.Y
            Z = self.Z
            V = self.V

        # residual and variance
        R = Y - F_beta
        v = np.split(V, self.idx_split[1:])
        z = np.split(Z, self.idx_split[1:], axis=0)
        D = SquareBlockDiagMat([(z[i]*gamma).dot(z[i].T) + np.diag(v[i])
                                for i in range(self.m)])

        val = 0.5*self.N*np.log(2.0*np.pi)

        if use_ad:
            # should only use when testing
            varmat = D
            D = varmat.full()
            inv_D = np.linalg.inv(D)
            val += 0.5*np.log(np.linalg.det(D))
            val += 0.5*R.dot(inv_D.dot(R))
        else:
            val += 0.5*D.logdet()
            val += 0.5*R.dot(D.invdot(R))

        # add gpriors
        if self.use_regularizer:
            val += 0.5*self.hw.dot((self.H.fun(x) - self.hm)**2)

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
        beta = x[self.idx_beta]
        gamma = x[self.idx_gamma]

        gamma[gamma <= 0.0] = 0.0

        # trimming option
        if self.use_trimming:
            sqrt_w = np.sqrt(self.w)
            sqrt_W = sqrt_w.reshape(self.N, 1)
            F_beta = self.F.fun(beta)*sqrt_w
            JF_beta = self.F.jac_fun(beta)*sqrt_W
            Y = self.Y*sqrt_w
            Z = self.Z*sqrt_W
            V = self.V**self.w
        else:
            F_beta = self.F.fun(beta)
            JF_beta = self.F.jac_fun(beta)
            Y = self.Y
            Z = self.Z
            V = self.V

        # residual and variance
        R = Y - F_beta
        v = np.split(V, self.idx_split[1:])
        z = np.split(Z, self.idx_split[1:], axis=0)
        D = SquareBlockDiagMat([(z[i]*gamma).dot(z[i].T) + np.diag(v[i])
                                for i in range(self.m)])

        # gradient for beta
        DR = D.invdot(R)
        g_beta = -JF_beta.T.dot(DR)

        # gradient for gamma
        DZ = D.invdot(Z)
        g_gamma = 0.5*np.sum(Z*DZ, axis=0) -\
            0.5*np.sum(
                np.add.reduceat(DZ.T*R, self.idx_split, axis=1)**2,
                axis=1)

        g = np.hstack((g_beta, g_gamma))

        # add gradient from the regularizer
        if self.use_regularizer:
            g += self.H.jac_fun(x).T.dot((self.H.fun(x) - self.hm)*self.hw)

        # add gradient from the gprior
        if self.use_gprior:
            g += (x[:self.k] - self.gm)*self.gw

        # add gradient from the lprior
        if self.use_lprior:
            g = np.hstack((g, self.lw))

        return g

    def objectiveTrimming(self, w):
        t = (self.Z**2).dot(self.gamma)
        r = self.Y - self.F.fun(self.beta)
        v = self.V
        d = v + t

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
        r = (self.Y - self.F.fun(self.beta))**2
        v = self.V
        d = v + t

        g = 0.5*r/d
        g += 0.5*np.log(d)

        return g

    def optimize(self, x0=None, print_level=0, max_iter=100, tol=1e-8,
                 acceptable_tol=1e-6,
                 nlp_scaling_method=None,
                 nlp_scaling_min_value=None):
        if x0 is None:
            x0 = np.hstack((self.beta, self.gamma))
            if self.use_lprior:
                x0 = np.hstack((x0, np.zeros(self.k)))

        assert x0.size == self.k_total

        opt_problem = ipopt.problem(
            n=int(self.k_total),
            m=int(self.num_constraints),
            problem_obj=self,
            lb=self.uprior[0],
            ub=self.uprior[1],
            cl=self.cl,
            cu=self.cu
            )

        opt_problem.addOption('print_level', print_level)
        opt_problem.addOption('max_iter', max_iter)
        opt_problem.addOption('tol', tol)
        opt_problem.addOption('acceptable_tol', acceptable_tol)
        if nlp_scaling_method is not None:
            opt_problem.addOption('nlp_scaling_method', nlp_scaling_method)
        if nlp_scaling_min_value is not None:
            opt_problem.addOption('nlp_scaling_min_value', nlp_scaling_min_value)

        soln, info = opt_problem.solve(x0)

        self.soln = soln
        self.info = info
        self.beta = soln[self.idx_beta]
        self.gamma = soln[self.idx_gamma]

    def fitModel(self, x0=None,
                 inner_print_level=0,
                 inner_max_iter=20,
                 inner_tol=1e-8,
                 inner_acceptable_tol=1e-6,
                 inner_nlp_scaling_method=None,
                 inner_nlp_scaling_min_value=None,
                 outer_verbose=False,
                 outer_max_iter=100,
                 outer_step_size=1.0,
                 outer_tol=1e-6,
                 normalize_trimming_grad=False):

        if not self.use_trimming:
            self.optimize(x0=x0,
                          print_level=inner_print_level,
                          max_iter=inner_max_iter,
                          acceptable_tol=inner_acceptable_tol,
                          nlp_scaling_method=inner_nlp_scaling_method,
                          nlp_scaling_min_value=inner_nlp_scaling_min_value)

            return self.beta, self.gamma, self.w

        self.soln = x0

        num_iter = 0
        err = outer_tol + 1.0

        while err >= outer_tol:
            self.optimize(x0=self.soln,
                          print_level=inner_print_level,
                          max_iter=inner_max_iter,
                          tol=inner_tol,
                          acceptable_tol=inner_acceptable_tol,
                          nlp_scaling_method=inner_nlp_scaling_method,
                          nlp_scaling_min_value=inner_nlp_scaling_min_value)

            w_grad = self.gradientTrimming(self.w)
            if normalize_trimming_grad:
                w_grad /= np.linalg.norm(w_grad)
            w_new = project_to_capped_simplex(
                        self.w - outer_step_size*w_grad,
                        self.num_inliers,
                        active_index=self.active_trimming_id)

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

        S = self.S

        if self.use_trimming:
            R = (self.Y - self.F.fun(self.beta))*np.sqrt(self.w)
            Z = self.Z*(np.sqrt(self.w).reshape(self.N, 1))
        else:
            R = self.Y - self.F.fun(self.beta)
            Z = self.Z

        iV = 1.0/S**2
        iVZ = Z*iV.reshape(iV.size, 1)

        r = np.split(R, np.cumsum(self.n)[:-1])
        v = np.split(S**2, np.cumsum(self.n)[:-1])
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

    def simulateData(self, beta_t, gamma_t, sim_prior=True, sim_re=True):
        # sample random effects and measurement error
        if sim_re:
            u = np.random.randn(self.m, self.k_gamma)*np.sqrt(gamma_t)
        else:
            if not hasattr(self, 'u'):
                self.estimateRE()
            u = self.u

        U = np.repeat(u, self.n, axis=0)
        ZU = np.sum(self.Z*U, axis=1)

        S = self.S

        E = np.random.randn(self.N)*S

        self.Y = self.F.fun(beta_t) + ZU + E

        if sim_prior:
            if self.use_gprior:
                valid_id = ~np.isinf(self.gprior[1])
                valid_num = np.sum(valid_id)
                self.gm = np.zeros(self.k)
                self.gm[valid_id] = self.gprior[0][valid_id] +\
                    np.random.randn(valid_num)*self.gprior[1][valid_id]

            if self.use_regularizer:
                valid_id = ~np.isinf(self.h[1])
                valid_num = np.sum(valid_id)
                self.hm = np.zeros(self.num_regularizer)
                self.hm[valid_id] = self.h[0][valid_id] +\
                    np.random.randn(valid_num)*self.h[1][valid_id]


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

        F = LinearMapping(X)

        # constraints, regularizer and priors
        if use_constraints:
            M = np.ones((1, k))
            C = LinearMapping(M)
            c = np.array([[0.0], [1.0]])
        else:
            C, c = None, None

        if use_regularizer:
            M = np.ones((1, k))
            H = LinearMapping(M)
            h = np.array([[0.0], [2.0]])
        else:
            H, h = None, None

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

        model = cls(n, Y, F, Z, S,
                   C=C, c=c,
                   H=H, h=h,
                   uprior=uprior, gprior=gprior,
                   inlier_percentage=inlier_percentage)
        return model

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

        F = LinearMapping(X)

        uprior = np.array([[-np.inf]*k_beta + [0.0], [np.inf]*k_beta + [0.0]])
        lprior = np.array([[0.0]*k, [np.sqrt(2.0)/weight]*k])

        return cls(n, Y, F, Z, S,
                   uprior=uprior, lprior=lprior)

    @staticmethod
    def sampleSoln(lt, sample_size=1, print_level=0, max_iter=100,
                   sim_prior=True, sim_re=True):
        beta_samples = np.zeros((sample_size, lt.k_beta))
        gamma_samples = np.zeros((sample_size, lt.k_gamma))

        beta_t = lt.beta.copy()
        gamma_t = lt.gamma.copy()

        lt_copy = deepcopy(lt)
        lt_copy.uprior[:, lt.k_beta:] = np.vstack((gamma_t, gamma_t))

        for i in range(sample_size):
            lt_copy.simulateData(beta_t, gamma_t,
                                 sim_prior=sim_prior, sim_re=sim_re)
            lt_copy.optimize(x0=np.hstack((beta_t, gamma_t)),
                             print_level=print_level,
                             max_iter=max_iter)

            u_samples = lt_copy.estimateRE()

            beta_samples[i] = lt_copy.beta.copy()
            gamma_samples[i] = np.maximum(
                lt.uprior[0, lt.k_beta:],
                np.minimum(lt.uprior[1, lt.k_beta:], np.var(u_samples, axis=0))
            )

            print('sampling solution progress %0.2f' % ((i + 1)/sample_size),
                  end='\r')

        return beta_samples, gamma_samples
