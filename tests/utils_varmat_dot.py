# test function dot


def utils_varmat_dot():
    import numpy as np
    from limetr.utils import VarMat

    ok = True
    # setup test problem
    # -------------------------------------------------------------------------
    var_mat = VarMat.testProblem()

    tol = 1e-10

    # test the dot product with vector
    # -------------------------------------------------------------------------
    x = np.random.randn(var_mat.num_data)

    tr_y = var_mat.varMat().dot(x)
    my_y = var_mat.dot(x)

    err = np.linalg.norm(tr_y - my_y)
    ok_vec = err < tol

    if not ok_vec:
        print('err', err)
        print('tr_y', tr_y)
        print('my_y', my_y)

    # test the dot product with matrix
    # -------------------------------------------------------------------------
    x = np.random.randn(var_mat.num_data, 2)

    tr_y = var_mat.varMat().dot(x)
    my_y = var_mat.dot(x)

    err = np.linalg.norm(tr_y - my_y)
    ok_mat = err < tol

    if not ok_mat:
        print('err', err)
        print('tr_y', tr_y)
        print('my_y', my_y)

    ok = ok and ok_vec and ok_mat

    return ok
