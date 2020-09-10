# test block diagonal matrix in utils module
import numpy as np
import pytest
import limetr.utils as utils

nblocks = 5
nrows = np.random.randint(1, 5, nblocks)
ncols = np.random.randint(1, 5, nblocks)

@pytest.fixture
def bdmat():
    mat_blocks = [np.random.randn(nrows[i], ncols[i]) for i in range(nblocks)]
    return utils.BlockDiagMat(mat_blocks)


@pytest.fixture
def sbdmat():
    mat_blocks = [np.random.randn(ncols[i], ncols[i]) for i in range(nblocks)]
    mat_blocks = [mat.T.dot(mat) for mat in mat_blocks]
    return utils.SquareBlockDiagMat(mat_blocks)


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
