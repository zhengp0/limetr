# utility classes and functions
import numpy as np
from scipy.optimize import bisect


def proj_capped_simplex(w, w_sum, active_id=None):
    N = w.size
    if active_id is None:
        active_id = np.arange(N)

    w_all = np.ones(N)
    w = w[active_id]
    w_sum = w_sum - (N - active_id.size)

    a = np.min(w) - 1.0
    b = np.max(w) - 0.0

    def f(x):
        return np.sum(np.maximum(np.minimum(w - x, 1.0), 0.0)) - w_sum

    x = bisect(f, a, b)

    w = np.maximum(np.minimum(w - x, 1.0), 0.0)
    w_all[active_id] = w
    return w_all
