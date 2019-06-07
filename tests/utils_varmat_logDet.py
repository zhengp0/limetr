# test function logDet


def utils_varmat_logDet():
    import numpy as np
    from limetr.utils import VarMat

    ok = True
    ok_rank_1 = True
    # setup test problem
    # -------------------------------------------------------------------------
    var_mat = VarMat.testProblem()
    var_mat_rank_1 = VarMat.testProblemRank1()

    tol = 1e-8

    # test the logDet
    # -------------------------------------------------------------------------
    tr_y = np.log(np.linalg.det(var_mat.varMat()))
    my_y = var_mat.logDet()

    tr_y_rank_1 = np.log(np.linalg.det(var_mat_rank_1.varMat()))
    my_y_rank_1 = var_mat_rank_1.logDet()

    err = np.linalg.norm(tr_y - my_y)
    err_rank_1 = np.linalg.norm(tr_y_rank_1 - my_y_rank_1)
    ok = ok and err < tol
    ok_rank_1 = ok_rank_1 and err_rank_1 < tol

    if not ok:
        print('err', err)
        print('tr_y', tr_y)
        print('my_y', my_y)

    if not ok_rank_1:
        print('err_rank_1', err_rank_1)
        print('tr_y_rank_1', tr_y_rank_1)
        print('my_y_rank_1', my_y_rank_1)

    ok = ok and ok_rank_1

    return ok
