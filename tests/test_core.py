"""
Test core module
"""
import pytest
import numpy as np

from limetr.data import Data
from limetr.linalg import LinearMapping
from limetr.variable import FeVariable, ReVariable
from limetr.core import LimeTr
from limetr.utils import split_by_sizes


# pylint:disable=redefined-outer-name


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
def revar():
    np.random.seed(123)
    mat = np.random.randn(5, 2)
    return ReVariable(LinearMapping(mat))


@pytest.fixture
def inlier_pct():
    return 0.95


@pytest.fixture
def model(data, fevar, revar, inlier_pct):
    return LimeTr(data, fevar, revar, inlier_pct=inlier_pct)


@pytest.mark.parametrize("my_fevar",
                         [FeVariable(LinearMapping(np.ones((4, 3))))])
def test_validate_fevar(data, my_fevar, revar, inlier_pct):
    with pytest.raises(ValueError):
        LimeTr(data, my_fevar, revar, inlier_pct=inlier_pct)


@pytest.mark.parametrize("my_revar",
                         [ReVariable(LinearMapping(np.ones((4, 3))))])
def test_validate_revar(data, fevar, my_revar, inlier_pct):
    with pytest.raises(ValueError):
        LimeTr(data, fevar, my_revar, inlier_pct=inlier_pct)


@pytest.mark.parametrize("my_inlier_pct", [-0.1, 1.1])
def test_validate_inlier_pct(data, fevar, revar, my_inlier_pct):
    with pytest.raises(ValueError):
        LimeTr(data, fevar, revar, inlier_pct=my_inlier_pct)


@pytest.mark.parametrize("var",
                         [np.hstack([np.ones(2), np.zeros(2)]),
                          np.hstack([np.ones(2), np.full(2, -0.1)])])
def test_get_vars(model, var):
    beta, gamma = model.get_vars(var)
    assert beta.size == model.fevar.size
    assert gamma.size == model.revar.size
    assert np.allclose(beta, 1.0)
    assert np.allclose(gamma, 0.0)


@pytest.mark.parametrize("beta", [np.ones(2)])
def test_get_residual(model, beta):
    my_residual = (model.data.obs - model.fevar.mapping(beta))
    assert np.allclose(model.get_residual(beta), my_residual)


@pytest.mark.parametrize("beta", [np.ones(2)])
def test_get_femat(model, beta):
    my_femat = model.fevar.mapping.jac(beta)
    assert np.allclose(model.get_femat(beta), my_femat)


def test_get_remat(model):
    my_remat = model.revar.mapping.mat
    assert np.allclose(model.get_remat(), my_remat)


def test_get_obsvar(model):
    my_obsvar = model.data.obs_se**2
    assert np.allclose(model.get_obsvar(), my_obsvar)


@pytest.mark.parametrize("gamma", [np.zeros(2)])
def test_get_varmat(model, gamma):
    my_varmat = np.diag(model.get_varmat(gamma).mat)
    assert np.allclose(my_varmat, model.get_obsvar())


@pytest.mark.parametrize("var", [np.hstack([np.ones(2), np.zeros(2)])])
def test_objective(model, var):
    beta, _ = model.get_vars(var)
    my_obj = model.objective(var)
    tr_obj = 0.5*(np.sum(model.get_residual(beta)**2/model.get_obsvar()) +
                  np.sum(np.log(model.get_obsvar())))

    assert np.isclose(my_obj, tr_obj)


@pytest.mark.parametrize("var", [np.hstack([np.ones(2), np.zeros(2)])])
def test_gradient(model, var):
    my_grad = model.gradient(var)
    tr_grad = ad_jacobian(model.objective, var, (var.size,))

    assert np.allclose(my_grad, tr_grad)


@pytest.mark.parametrize("var", [np.hstack([np.zeros(2), np.zeros(2)])])
def test_hessian(model, var):
    my_hess = model.hessian(var)
    eigvals = np.linalg.eigvals(my_hess)

    assert all(eigvals >= 0.0)


def test_get_model_init(model):
    var = model.get_model_init()
    beta, gamma = model.get_vars(var)
    assert beta.size == model.fevar.size
    assert gamma.size == model.revar.size
    assert all(gamma >= 0.0)


def test_fit_model(model):
    model.fit_model()
    assert model.result is not None


@pytest.mark.parametrize("var", [np.hstack([np.zeros(2), np.zeros(2)])])
def test_get_random_effects_zero_gamma(model, var):
    random_effects = model.get_random_effects(var)
    assert np.allclose(random_effects, 0.0)


@pytest.mark.parametrize("var", [np.hstack([np.zeros(2), np.ones(2)])])
def test_get_random_effects_nonzero_gamma(model, var):
    beta, gamma = model.get_vars(var)
    u = model.get_random_effects(var)
    r = split_by_sizes(model.get_residual(beta), model.data.group_sizes)
    v = split_by_sizes(model.get_obsvar(), model.data.group_sizes)
    z = split_by_sizes(model.get_remat(), model.data.group_sizes)

    def grad_fun(z, v, r, gamma, u):
        return (z.T/v).dot(z.dot(u) - r) + u/gamma

    assert np.allclose(np.hstack([
        grad_fun(z[i], v[i], r[i], gamma, u[i])
        for i in range(model.data.num_groups)
    ]), 0.0)
