"""
Test Variable Module
"""
import pytest
import numpy as np
from limetr.linalg import LinearMapping
from limetr.stats import GaussianPrior, UniformPrior, LinearGaussianPrior, LinearUniformPrior
from limetr.variable import Variable, FeVariable, ReVariable


# pylint: disable=redefined-outer-name


@pytest.fixture
def mapping():
    return LinearMapping(np.ones((3, 2)))


@pytest.fixture
def var(mapping):
    return Variable(mapping)


@pytest.fixture
def gprior():
    return GaussianPrior(mean=-1.0, size=2)


@pytest.fixture
def uprior():
    return UniformPrior(lb=-1.0, size=2)


@pytest.fixture
def linear_gprior():
    return LinearGaussianPrior(mat=np.ones((5, 2)), mean=-1.0)


@pytest.fixture
def linear_uprior():
    return LinearUniformPrior(mat=np.ones((5, 2)), lb=-1.0)


@pytest.mark.parametrize("mapping", [1, "a"])
def test_var_validate_mapping(mapping):
    with pytest.raises(TypeError):
        Variable(mapping)


def test_var_default(var):
    assert var.size == 2
    assert var.gprior.size == 2
    assert var.uprior.size == 2
    assert len(var.linear_gpriors) == 0
    assert len(var.linear_upriors) == 0


def test_var_update_gprior(var, gprior):
    var.update_gprior(gprior)
    assert np.allclose(var.gprior.mean, -1.0)


def test_var_update_uprior(var, uprior):
    var.update_uprior(uprior)
    assert np.allclose(var.uprior.lb, -1.0)


def test_var_update_linear_gpriors(var, linear_gprior):
    var.update_linear_gpriors(linear_gprior)
    assert len(var.linear_gpriors) == 1
    assert np.allclose(var.linear_gpriors[0].mat, 1.0)
    assert np.allclose(var.linear_gpriors[0].mean, -1.0)


def test_var_update_linear_upriors(var, linear_uprior):
    var.update_linear_upriors(linear_uprior)
    assert len(var.linear_upriors) == 1
    assert np.allclose(var.linear_upriors[0].mat, 1.0)
    assert np.allclose(var.linear_upriors[0].lb, -1.0)


def test_var_update_priors(var, gprior, uprior, linear_gprior, linear_uprior):
    var.update_priors([gprior, uprior, linear_gprior, linear_uprior])
    assert np.allclose(var.gprior.mean, -1.0)
    assert np.allclose(var.uprior.lb, -1.0)
    assert len(var.linear_gpriors) == 1
    assert len(var.linear_upriors) == 1


def test_var_reset_priors(var, gprior, uprior, linear_gprior, linear_uprior):
    var.update_priors([gprior, uprior, linear_gprior, linear_uprior])
    var.reset_priors()
    assert np.allclose(var.gprior.mean, 0.0)
    assert all(np.isneginf(var.uprior.lb))
    assert len(var.linear_gpriors) == 0
    assert len(var.linear_upriors) == 0


@pytest.mark.parametrize("x", [np.array([1.0, 2.0])])
def test_var_prior_objective(var, gprior, linear_gprior, x):
    var.update_priors([gprior, linear_gprior])
    my_val = var.prior_objective(x)
    tr_val = gprior.objective(x) + linear_gprior.objective(x)
    assert np.isclose(my_val, tr_val)


@pytest.mark.parametrize("x", [np.array([1.0, 2.0])])
def test_var_prior_gradient(var, gprior, linear_gprior, x):
    var.update_priors([gprior, linear_gprior])
    my_val = var.prior_gradient(x)
    tr_val = gprior.gradient(x) + linear_gprior.gradient(x)
    assert np.allclose(my_val, tr_val)


@pytest.mark.parametrize("x", [np.array([1.0, 2.0])])
def test_var_prior_hessian(var, gprior, linear_gprior, x):
    var.update_priors([gprior, linear_gprior])
    my_val = var.prior_hessian(x)
    tr_val = gprior.hessian(x) + linear_gprior.hessian(x)
    assert np.allclose(my_val, tr_val)


def test_var_get_uprior_info(var, uprior):
    my_val = var.get_uprior_info()
    assert np.allclose(my_val, np.array([[-np.inf]*2, [np.inf]*2]))
    var.update_priors([uprior])
    my_val = var.get_uprior_info()
    assert np.allclose(my_val, uprior.info)


def test_var_get_linear_upriors_mat(var, linear_uprior):
    my_val = var.get_linear_upriors_mat()
    assert my_val.shape == (0, 2)
    var.update_priors([linear_uprior])
    my_val = var.get_linear_upriors_mat()
    assert np.allclose(my_val, linear_uprior.mat)


def test_var_get_linear_upriors_info(var, linear_uprior):
    my_val = var.get_linear_upriors_info()
    assert my_val.shape == (2, 0)
    var.update_priors([linear_uprior])
    my_val = var.get_linear_upriors_info()
    assert np.allclose(my_val, linear_uprior.info)


def test_fevar(mapping):
    var = FeVariable(mapping)
    assert var.name == "fixed effects"


def test_revar(mapping):
    var = ReVariable(mapping)
    assert np.allclose(var.uprior.lb, 0.0)


def test_revar_update_uprior_validate(mapping, uprior):
    var = ReVariable(mapping)
    with pytest.raises(ValueError):
        var.update_priors([uprior])


def test_revar_reset_priors(mapping):
    var = ReVariable(mapping)
    uprior = UniformPrior(lb=1.0, size=2)
    var.update_priors([uprior])
    assert all(var.uprior.lb == 1.0)
    var.reset_priors()
    assert all(var.uprior.lb == 0.0)
