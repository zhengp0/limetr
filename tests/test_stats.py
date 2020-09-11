# Test stats module
import numpy as np
import pytest
import limetr.stats as stats


@pytest.mark.parametrize(('mean', 'sd', 'size'),
                         [(np.zeros(5), 1.0, None),
                          (0.0, np.ones(5), None),
                          (0.0, 1.0, 5),
                          (None, None, 5)])
def test_gaussian(mean, sd, size):
    gaussian = stats.Gaussian(mean=mean, sd=sd, size=size)
    assert gaussian.size == 5


@pytest.mark.parametrize(('lb', 'ub', 'size'),
                         [(np.zeros(5), 1.0, None),
                          (0.0, np.ones(5), None),
                          (0.0, 1.0, 5),
                          (None, None, 5)])
def test_uniform(lb, ub, size):
    uniform = stats.Uniform(lb=lb, ub=ub, size=size)
    assert uniform.size == 5


@pytest.mark.parametrize(('mean', 'sd', 'size'),
                         [(np.zeros(5), 1.0, None),
                          (0.0, np.ones(5), None),
                          (0.0, 1.0, 5),
                          (None, None, 5)])
def test_laplace(mean, sd, size):
    laplace = stats.Laplace(mean=mean, sd=sd, size=size)
    assert laplace.size == 5
