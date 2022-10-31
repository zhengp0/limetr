import numpy as np
from limetr.utils import proj_capped_simplex


def test_proj_capped_simplex():
    # setup test problem
    # -------------------------------------------------------------------------
    w = np.ones(10)
    sum_w = 9.0

    tr_w = np.repeat(0.9, 10)
    my_w = proj_capped_simplex(w, sum_w)

    tol = 1e-10
    err = np.linalg.norm(tr_w - my_w)

    assert err < tol
