"""
Test Fixed Effects Module
"""
import numpy as np
import pytest
from limetr.data import Data
from limetr.linalg import LinearMapping
from limetr.models import FeModel
from limetr.variable import FeVariable


def ad_jacobian(fun, x, shape, eps=1e-10):
    n = len(x)
    c = x + 0j
    g = np.zeros(shape)
    for i in range(n):
        c[i] += eps*1j
        g[i] = fun(c).imag/eps
        c[i] -= eps*1j
    return g


@pytest.fixture
def data():
    np.random.seed(123)
    obs = np.random.randn(5)
    obs_se = 0.1 + 0.1*np.random.rand(5)
    group_sizes = np.array([2, 2, 1])
    return Data(obs=obs,
                obs_se=obs_se,
                group_sizes=group_sizes)


@pytest.fixture
def fevar():
    np.random.seed(123)
    mat = np.random.randn(5, 2)
    return FeVariable(LinearMapping(mat))


@pytest.fixture
def inlier_pct():
    return 0.95


@pytest.fixture
def model(data, fevar, inlier_pct):
    return FeModel(data, fevar, inlier_pct=inlier_pct)


@pytest.mark.parametrize("my_fevar",
                         [FeVariable(LinearMapping(np.ones((4, 3))))])
def test_validate_fevar(data, my_fevar, inlier_pct):
    with pytest.raises(ValueError):
        FeModel(data, my_fevar, inlier_pct=inlier_pct)


@pytest.mark.parametrize("my_inlier_pct", [-0.1, 1.1])
def test_validate_inlier_pct(data, fevar, my_inlier_pct):
    with pytest.raises(ValueError):
        FeModel(data, fevar, inlier_pct=my_inlier_pct)


@pytest.mark.parametrize("beta", [np.ones(2)])
def test_get_residual(model, beta):
    my_residual = (model.data.obs - model.fevar.mapping(beta))
    assert np.allclose(model.get_residual(beta), my_residual)


@pytest.mark.parametrize("beta", [np.ones(2)])
def test_get_femat(model, beta):
    my_femat = model.fevar.mapping.jac(beta)
    assert np.allclose(model.get_femat(beta), my_femat)


def test_get_obs_varmat(model):
    my_obs_varmat = model.data.obs_varmat
    assert np.allclose(model.get_obs_varmat(), my_obs_varmat)


@pytest.mark.parametrize("beta", [np.ones(2)])
def test_objective(model, beta):
    my_obj = model.objective(beta)
    tr_obj = 0.5*np.sum(model.get_residual(beta)**2 /
                        np.diag(model.get_obs_varmat()))

    assert np.isclose(my_obj, tr_obj)


@pytest.mark.parametrize("beta", [np.ones(2)])
def test_gradient(model, beta):
    my_grad = model.gradient(beta)
    tr_grad = ad_jacobian(model.objective, beta, (beta.size,))

    assert np.allclose(my_grad, tr_grad)


@pytest.mark.parametrize("beta", [np.zeros(2)])
def test_hessian(model, beta):
    my_hess = model.hessian(beta)
    eigvals = np.linalg.eigvals(my_hess)

    assert all(eigvals >= 0.0)


def test_get_model_init(model):
    beta = model.get_model_init()
    assert beta.size == model.fevar.size
