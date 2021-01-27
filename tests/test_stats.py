"""
Test Statistic Module
"""
import pytest
import numpy as np
from limetr.stats.prior import (GaussianPrior,
                                UniformPrior,
                                LinearPrior,
                                LinearGaussianPrior,
                                LinearUniformPrior)


def test_gprior_validate():
    with pytest.raises(ValueError):
        GaussianPrior(mean=1.0, sd=-1.0)


@pytest.mark.parametrize("mean", [np.array([1.0, 2.0, 3.0])])
@pytest.mark.parametrize("sd", [np.array([1.0, 1.0, 1.0])])
@pytest.mark.parametrize("var", [np.array([1.0, 1.0, 1.0])])
def test_gprior_objective(mean, sd, var):
    prior = GaussianPrior(mean=mean, sd=sd)
    my_obj = prior.objective(var)
    tr_obj = 0.5*np.sum((mean - var)**2/sd**2)
    assert my_obj == tr_obj


@pytest.mark.parametrize("mean", [np.array([1.0, 2.0, 3.0])])
@pytest.mark.parametrize("sd", [np.array([1.0, 1.0, 1.0])])
@pytest.mark.parametrize("var", [np.array([1.0, 1.0, 1.0])])
def test_gprior_gradient(mean, sd, var):
    prior = GaussianPrior(mean=mean, sd=sd)
    my_grad = prior.gradient(var)
    tr_grad = (var - mean)/sd**2
    assert np.allclose(my_grad, tr_grad)


@pytest.mark.parametrize("mean", [np.array([1.0, 2.0, 3.0])])
@pytest.mark.parametrize("sd", [np.array([1.0, 1.0, 1.0])])
@pytest.mark.parametrize("var", [np.array([1.0, 1.0, 1.0])])
def test_gprior_hessian(mean, sd, var):
    prior = GaussianPrior(mean=mean, sd=sd)
    my_hess = prior.hessian(var)
    tr_hess = np.diag(1.0/sd**2)
    assert np.allclose(my_hess, tr_hess)


def test_uprior_validate():
    with pytest.raises(ValueError):
        UniformPrior(lb=1.0, ub=-1.0)


@pytest.mark.parametrize("lb", [np.array([1.0, 1.0, 1.0])])
@pytest.mark.parametrize("ub", [np.array([1.0, 2.0, 3.0])])
@pytest.mark.parametrize("var", [np.array([1.0, 1.0, 1.0])])
def test_uprior_objective(lb, ub, var):
    prior = UniformPrior(lb=lb, ub=ub)
    my_obj = prior.objective(var)
    tr_obj = 0.0
    assert my_obj == tr_obj


@pytest.mark.parametrize("lb", [np.array([1.0, 1.0, 1.0])])
@pytest.mark.parametrize("ub", [np.array([1.0, 2.0, 3.0])])
@pytest.mark.parametrize("var", [np.array([1.0, 1.0, 1.0])])
def test_uprior_gradient(lb, ub, var):
    prior = UniformPrior(lb=lb, ub=ub)
    my_grad = prior.gradient(var)
    tr_grad = 0.0
    assert np.allclose(my_grad, tr_grad)


@pytest.mark.parametrize("lb", [np.array([1.0, 1.0, 1.0])])
@pytest.mark.parametrize("ub", [np.array([1.0, 2.0, 3.0])])
@pytest.mark.parametrize("var", [np.array([1.0, 1.0, 1.0])])
def test_uprior_hessian(lb, ub, var):
    prior = UniformPrior(lb=lb, ub=ub)
    my_hess = prior.hessian(var)
    tr_hess = 0.0
    assert np.allclose(my_hess, tr_hess)


@pytest.mark.parametrize(("mat, info"),
                         [(np.empty(shape=(0, 2)), np.empty(shape=(2, 0))),
                          (np.ones((3, 4)), np.ones((2, 3)))])
def test_linear_prior(mat, info):
    prior = LinearPrior(mat, info)
    assert prior.mat.shape[0] == prior.size


@pytest.mark.parametrize("mat", [np.ones((3, 4))])
@pytest.mark.parametrize("mean", [np.array([1.0, 2.0, 3.0])])
@pytest.mark.parametrize("sd", [np.array([1.0, 1.0, 1.0])])
@pytest.mark.parametrize("var", [np.array([1.0, 1.0, 1.0, 1.0])])
def test_linear_gprior_objective(mat, mean, sd, var):
    prior = LinearGaussianPrior(mat=mat, mean=mean, sd=sd)
    my_obj = prior.objective(var)
    tr_obj = 0.5*np.sum((mat.dot(var) - mean)**2/sd**2)
    assert my_obj == tr_obj


@pytest.mark.parametrize("mat", [np.ones((3, 4))])
@pytest.mark.parametrize("mean", [np.array([1.0, 2.0, 3.0])])
@pytest.mark.parametrize("sd", [np.array([1.0, 1.0, 1.0])])
@pytest.mark.parametrize("var", [np.array([1.0, 1.0, 1.0, 1.0])])
def test_linear_gprior_gradient(mat, mean, sd, var):
    prior = LinearGaussianPrior(mat=mat, mean=mean, sd=sd)
    my_grad = prior.gradient(var)
    tr_grad = mat.T.dot((mat.dot(var) - mean)/sd**2)
    assert np.allclose(my_grad, tr_grad)


@pytest.mark.parametrize("mat", [np.ones((3, 4))])
@pytest.mark.parametrize("mean", [np.array([1.0, 2.0, 3.0])])
@pytest.mark.parametrize("sd", [np.array([1.0, 1.0, 1.0])])
@pytest.mark.parametrize("var", [np.array([1.0, 1.0, 1.0, 1.0])])
def test_linear_gprior_hessian(mat, mean, sd, var):
    prior = LinearGaussianPrior(mat=mat, mean=mean, sd=sd)
    my_hess = prior.hessian(var)
    tr_hess = mat.T.dot(np.diag(1.0/sd**2).dot(mat))
    assert np.allclose(my_hess, tr_hess)


@pytest.mark.parametrize("mat", [np.ones((3, 4))])
@pytest.mark.parametrize("lb", [np.array([1.0, 1.0, 1.0])])
@pytest.mark.parametrize("ub", [np.array([1.0, 2.0, 3.0])])
def test_linear_uprior_hessian(mat, lb, ub):
    prior = LinearUniformPrior(mat=mat, lb=lb, ub=ub)
    assert mat.shape[0] == prior.size
