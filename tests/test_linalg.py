# Test linear algebra module
import numpy as np
import pytest
import limetr.linalg as linalg


#pylint: disable=redefined-outer-name


@pytest.fixture
def smooth_mapping():
    m = 5
    n = 2
    mat1 = np.random.randn(m, n)
    mat2 = np.random.randn(m, n)

    def fun(x):
        return np.exp(mat1.dot(x)) - np.exp(mat2.dot(x))

    def jac(x):
        return (mat1.T*np.exp(mat1.dot(x)) - mat2.T*np.exp(mat2.dot(x))).T

    return linalg.SmoothMapping((m, n), fun, jac)


@pytest.fixture
def linear_mapping():
    m = 5
    n = 2
    mat = np.random.randn(m, n)
    return linalg.LinearMapping(mat)


def test_smooth_mapping(smooth_mapping):
    x = np.zeros(smooth_mapping.shape[1])
    assert smooth_mapping(x).size == smooth_mapping.shape[0]
    assert smooth_mapping.jac(x).shape == smooth_mapping.shape


def test_linear_mapping(linear_mapping):
    x = np.zeros(linear_mapping.shape[1])
    assert linear_mapping.mat.shape == linear_mapping.shape
    assert linear_mapping(x).size == linear_mapping.shape[0]
    assert linear_mapping.jac(x).shape == linear_mapping.shape
