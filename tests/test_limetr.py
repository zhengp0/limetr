import numpy as np
import pytest
from limetr import LimeTr
from scipy.linalg import block_diag


def lmtr_test_problem(
    use_trimming=False,
    use_constraints=False,
    use_regularizer=False,
    use_uprior=False,
    use_gprior=False
):
    np.random.seed(123)
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

    return LimeTr(n, k_beta, k_gamma, Y, F, JF, Z, S=S,
                  C=C, JC=JC, c=c,
                  H=H, JH=JH, h=h,
                  uprior=uprior, gprior=gprior,
                  inlier_percentage=inlier_percentage)


def lmtr_test_problem_lasso():
    np.random.seed(123)
    m = 100
    n = [1]*m
    N = sum(n)
    k_beta = 10
    k_gamma = 1
    k = k_beta + k_gamma

    beta_t = np.zeros(k_beta)
    beta_t[np.random.choice(k_beta, 5)] = np.sign(np.random.randn(5))

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

    return LimeTr(n, k_beta, k_gamma, Y, F, JF, Z, S=S,
                  uprior=uprior, lprior=lprior)


def lmtr_objective(lmtr, x):
    # unpack variable
    beta, gamma = lmtr._get_vars(x)

    # trimming option
    F_beta, _, Y, Z = lmtr._get_nll_components(beta)

    # residual and variance
    R = Y - F_beta

    val = 0.5*lmtr.N*np.log(2.0*np.pi)

    # should only use when testing
    split_idx = np.cumsum(lmtr.n)[:-1]
    v_study = np.split(lmtr.V, split_idx)
    z_study = np.split(Z, split_idx, axis=0)
    D = block_diag(*[np.diag(v) + (z*gamma).dot(z.T)
                     for v, z in zip(v_study, z_study)])
    inv_D = np.linalg.inv(D)

    eigvals = np.linalg.eigvals(D)
    val += 0.5*np.sum(np.log(eigvals))
    val += 0.5*R.dot(inv_D.dot(R))

    # add gpriors
    if lmtr.use_regularizer:
        val += 0.5*lmtr.hw.dot((lmtr.H(x) - lmtr.hm)**2)

    if lmtr.use_gprior:
        val += 0.5*lmtr.gw.dot((x[:lmtr.k] - lmtr.gm)**2)

    if lmtr.use_lprior:
        val += lmtr.lw.dot(x[lmtr.k:])

    return val


def lmtr_gradient(lmtr, x, eps=1e-10):
    # should only use when testing
    g = np.zeros(lmtr.k_total)
    z = x + 0j
    for i in range(lmtr.k_total):
        z[i] += eps*1j
        g[i] = lmtr_objective(lmtr, z).imag/eps
        z[i] -= eps*1j

    return g


def lmtr_gradient_trimming(lmtr, w, eps=1e-10):
    g = np.zeros(lmtr.N)
    z = w + 0j
    for i in range(lmtr.N):
        z[i] += eps*1j
        g[i] = lmtr.objective_trimming(z).imag/eps
        z[i] -= eps*1j

    return g


def test_gradient():
    # setup test problem
    # -------------------------------------------------------------------------
    model = lmtr_test_problem(use_trimming=True,
                              use_constraints=True,
                              use_regularizer=True,
                              use_uprior=True,
                              use_gprior=True)

    # test the gradient
    # -------------------------------------------------------------------------
    x = np.random.randn(model.k)
    x[model.idx_gamma] = 0.1

    tr_grad = lmtr_gradient(model, x)
    my_grad = model.gradient(x)

    assert np.allclose(tr_grad, my_grad)


def test_hessian():
    model = lmtr_test_problem(use_trimming=True,
                              use_constraints=True,
                              use_regularizer=True,
                              use_uprior=True,
                              use_gprior=True)
    x = np.random.randn(model.k)
    x[model.idx_gamma] = 0.1
    hess = model.hessian(x)
    eigvals = np.linalg.eigvals(hess)
    assert (eigvals >= 0).all()


def test_limetr_gradient_trimming():
    # setup test problem
    # -------------------------------------------------------------------------
    model = lmtr_test_problem(use_trimming=True)

    # decouple all the studies
    model.n = np.array([1]*model.N)

    # test gradient_trimming
    # -------------------------------------------------------------------------
    w = model.w

    tr_grad = lmtr_gradient_trimming(model, w)
    my_grad = model.gradient_trimming(w)

    assert np.allclose(tr_grad, my_grad)


@pytest.mark.filterwarnings("ignore")
def test_limetr_lasso():
    # setup test problem
    # -------------------------------------------------------------------------
    model = lmtr_test_problem_lasso()

    tol = 1e-5

    # test lasso
    # -------------------------------------------------------------------------
    model.optimize()
    beta = model.beta
    zero_idx = np.abs(beta) <= 1e-6
    beta[zero_idx] = 0.0

    # calculate the gradient
    g_beta = -model.JF(beta).T.dot(model.Y - model.F(beta))
    for i in range(model.k_beta):
        if beta[i] == 0.0 and np.abs(g_beta[i]) < model.lw[i]:
            g_beta[i] = 0.0
        else:
            g_beta[i] += np.sign(beta[i])*model.lw[i]

    assert np.abs(g_beta).max() < tol


def test_limetr_objective():
    # setup test problem
    # -------------------------------------------------------------------------
    model = lmtr_test_problem(use_constraints=True,
                              use_regularizer=True,
                              use_uprior=True,
                              use_gprior=True)

    # test objective
    # -------------------------------------------------------------------------
    x = np.random.randn(model.k)
    x[model.idx_gamma] = 0.1

    tr_obj = lmtr_objective(model, x)
    my_obj = model.objective(x)

    assert np.isclose(tr_obj, my_obj)


def test_limetr_objective_trimming():
    # setup test problem
    # -------------------------------------------------------------------------
    model = lmtr_test_problem(use_trimming=True)

    # test objective_trimming
    # -------------------------------------------------------------------------
    w = model.w

    r = model.Y - model.F(model.beta)
    t = (model.Z**2).dot(model.gamma)
    d = model.V + t

    tr_obj = 0.5*np.sum(r**2*w/d) + 0.5*model.N*np.log(2.0*np.pi)\
        + 0.5*w.dot(np.log(d))
    my_obj = model.objective_trimming(w)

    assert np.isclose(tr_obj, my_obj)


def test_estimate_re():
    model = lmtr_test_problem()
    model.optimize()

    re = model.estimate_re()
    F_beta, _, Y, Z = model._get_nll_components(model.beta)

    r = F_beta + np.sum(Z*np.repeat(re, model.n, axis=0), axis=1) - Y
    r = np.split(r, np.cumsum(model.n)[:-1])
    z = np.split(Z, np.cumsum(model.n)[:-1], axis=0)
    v = np.split(model.V, np.cumsum(model.n)[:-1])

    g = np.vstack([
        (zi.T/vi).dot(ri) + ui / model.gamma
        for zi, vi, ri, ui in zip(z, v, r, re)
    ])

    assert np.allclose(g, 0.0)
