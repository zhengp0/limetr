# test function objectiveTrimming

def limetr_objectiveTrimming():
    import numpy as np
    from limetr.__init__ import LimeTr

    ok = True
    # setup test problem
    # -------------------------------------------------------------------------
    model = LimeTr.testProblem(use_trimming=True)

    # decouple all the studies
    model.n = np.array([1]*model.N)

    tol = 1e-8

    # test objectiveTrimming
    # -------------------------------------------------------------------------
    x = np.hstack((model.beta, model.gamma))
    w = model.w

    tr_obj = model.objective(x, use_ad=True)
    my_obj = model.objectiveTrimming(w)

    err = np.abs(tr_obj - my_obj)
    ok = ok and err < tol

    if not ok:
        print('err', err)
        print('tr_obj', tr_obj)
        print('my_obj', my_obj)

    return ok
