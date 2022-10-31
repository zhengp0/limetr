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
