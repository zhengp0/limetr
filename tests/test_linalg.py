# Test linear algebra module
import numpy as np
import pytest
import limetr.linalg as linalg


#pylint: disable=redefined-outer-name


nblocks = 5
nrows = np.random.randint(1, 5, nblocks)
ncols = np.random.randint(1, 5, nblocks)


@pytest.fixture
def bdmat():
    mat_blocks = [np.random.randn(nrows[i], ncols[i]) for i in range(nblocks)]
    return linalg.BlockDiagMat(mat_blocks)


@pytest.fixture
def sbdmat():
    mat_blocks = [np.random.randn(ncols[i], ncols[i]) for i in range(nblocks)]
    mat_blocks = [mat.T.dot(mat) for mat in mat_blocks]
    return linalg.SquareBlockDiagMat(mat_blocks)


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


@pytest.mark.parametrize('vec', [np.ones(ncols.sum()),
                                 np.random.randn(ncols.sum())])
def test_bd_dot(bdmat, vec):
    fullmat = bdmat.full()
    assert np.allclose(fullmat.dot(vec), bdmat.dot(vec))


@pytest.mark.parametrize('array', [np.ones(ncols.sum()),
                                   np.random.randn(ncols.sum()),
                                   np.random.randn(ncols.sum(), 2)])
def test_sbd_dot(sbdmat, array):
    fullmat = sbdmat.full()
    assert np.allclose(fullmat.dot(array), sbdmat.dot(array))


def test_sbd_inv(sbdmat):
    fullmat = sbdmat.full()
    invmat = sbdmat.inv()
    fullinvmat = invmat.full()
    assert np.allclose(np.linalg.inv(fullmat), fullinvmat)


@pytest.mark.parametrize('array', [np.ones(ncols.sum()),
                                   np.random.randn(ncols.sum()),
                                   np.random.randn(ncols.sum(), 2)])
def test_sbd_invdot(sbdmat, array):
    fullinvmat = sbdmat.inv().full()
    assert np.allclose(fullinvmat.dot(array), sbdmat.invdot(array))


def test_sbd_diag(sbdmat):
    fullmat = sbdmat.full()
    assert np.allclose(np.diag(fullmat), sbdmat.diag())


def test_sbd_block_eigvals(sbdmat):
    assert np.allclose(np.hstack([np.linalg.eigvals(mat) for mat in sbdmat.mat_blocks]), sbdmat.block_eigvals())


def test_sbd_det(sbdmat):
    fullmat = sbdmat.full()
    assert np.isclose(np.linalg.det(fullmat), sbdmat.det())


def test_sbd_logdet(sbdmat):
    fullmat = sbdmat.full()
    assert np.isclose(np.linalg.slogdet(fullmat)[1], sbdmat.logdet())


def test_smooth_mapping(smooth_mapping):
    x = np.zeros(smooth_mapping.shape[1])
    assert smooth_mapping(x).size == smooth_mapping.shape[0]
    assert smooth_mapping.jac(x).shape == smooth_mapping.shape


def test_linear_mapping(linear_mapping):
    x = np.zeros(linear_mapping.shape[1])
    assert linear_mapping.mat.shape == linear_mapping.shape
    assert linear_mapping(x).size == linear_mapping.shape[0]
    assert linear_mapping.jac(x).shape == linear_mapping.shape
