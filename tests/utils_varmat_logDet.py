# test function logDet

def utils_varmat_logDet():
    import numpy as np
    from limetr.utils import VarMat

    ok = True
    # setup test problem
    # -------------------------------------------------------------------------
    var_mat = VarMat.testProblem()

    tol = 1e-8

    # test the logDet
    # -------------------------------------------------------------------------
    tr_y = np.log(np.linalg.det(var_mat.varMat()))
    my_y = var_mat.logDet()

    err = np.linalg.norm(tr_y - my_y)
    ok = ok and err < tol

    if not ok:
        print('err', err)
        print('tr_y', tr_y)
        print('my_y', my_y)

    return ok
