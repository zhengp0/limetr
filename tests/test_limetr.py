import numpy as np
from limetr import LimeTr


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

    tr_grad = model.gradient(x, use_ad=True)
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

    tr_grad = model.gradientTrimming(w, use_ad=True)
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

    tr_obj = model.objective(x, use_ad=True)
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
