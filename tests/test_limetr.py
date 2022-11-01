import numpy as np
from limetr import LimeTr
from scipy.linalg import block_diag


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

    val += 0.5*np.log(np.linalg.det(D))
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
        g[i] = lmtr.objectiveTrimming(z).imag/eps
        z[i] -= eps*1j

    return g


def test_gradient():
    # setup test problem
    # -------------------------------------------------------------------------
    model = LimeTr.testProblem(use_trimming=True,
                               use_constraints=True,
                               use_regularizer=True,
                               use_uprior=True,
                               use_gprior=True)

    tol = 1e-6

    # test the gradient
    # -------------------------------------------------------------------------
    x = np.random.randn(model.k)
    x[model.idx_gamma] = 0.1

    tr_grad = lmtr_gradient(model, x)
    my_grad = model.gradient(x)

    err = np.linalg.norm(tr_grad - my_grad)
    assert err < tol


def test_limetr_gradientTrimming():
    # setup test problem
    # -------------------------------------------------------------------------
    model = LimeTr.testProblem(use_trimming=True)

    # decouple all the studies
    model.n = np.array([1]*model.N)

    tol = 1e-8

    # test gradientTrimming
    # -------------------------------------------------------------------------
    w = model.w

    tr_grad = lmtr_gradient_trimming(model, w)
    my_grad = model.gradientTrimming(w)

    err = np.linalg.norm(tr_grad - my_grad)
    assert err < tol


# def test_limetr_lasso():
#     # setup test problem
#     # -------------------------------------------------------------------------
#     model = LimeTr.testProblemLasso()

#     tol = 1e-6

#     # test lasso
#     # -------------------------------------------------------------------------
#     model.optimize()
#     beta = model.beta
#     zero_idx = np.abs(beta) <= 1e-8
#     beta[zero_idx] = 0.0

#     # calculate the gradient
#     g_beta = -model.JF(beta).T.dot(model.Y - model.F(beta))
#     for i in range(model.k_beta):
#         if beta[i] == 0.0 and np.abs(g_beta[i]) < model.lw[i]:
#             g_beta[i] = 0.0
#         else:
#             g_beta[i] += np.sign(beta[i])*model.lw[i]

#     err = np.linalg.norm(g_beta)
#     assert err < tol


def test_limetr_objective():
    # setup test problem
    # -------------------------------------------------------------------------
    model = LimeTr.testProblem(use_constraints=True,
                               use_regularizer=True,
                               use_uprior=True,
                               use_gprior=True)

    tol = 1e-8

    # test objective
    # -------------------------------------------------------------------------
    x = np.random.randn(model.k)
    x[model.idx_gamma] = 0.1

    tr_obj = lmtr_objective(model, x)
    my_obj = model.objective(x)

    err = np.abs(tr_obj - my_obj)
    assert err < tol


def test_limetr_objectiveTrimming():
    # setup test problem
    # -------------------------------------------------------------------------
    model = LimeTr.testProblem(use_trimming=True)

    tol = 1e-8

    # test objectiveTrimming
    # -------------------------------------------------------------------------
    w = model.w

    r = model.Y - model.F(model.beta)
    t = (model.Z**2).dot(model.gamma)
    d = model.V + t

    tr_obj = 0.5*np.sum(r**2*w/d) + 0.5*model.N*np.log(2.0*np.pi)\
        + 0.5*w.dot(np.log(d))
    my_obj = model.objectiveTrimming(w)

    err = np.abs(tr_obj - my_obj)
    assert err < tol
