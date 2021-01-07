# Test helper function in utils module
import numpy as np
import pytest
import limetr.utils as utils


@pytest.mark.parametrize('vec', [np.arange(6)])
@pytest.mark.parametrize('sizes', [[1, 2, 3], [3, 2, 1]])
def test_split_by_sizes(vec, sizes):
    vecs = utils.split_by_sizes(vec, sizes)
    assert all([vecs[i].size == size for i, size in enumerate(sizes)])


def test_empty_array():
    array = utils.empty_array()
    assert array.size == 0
    assert np.issubdtype(array.dtype, float)


@pytest.mark.parametrize('vec', [[0, 1, 2]])
@pytest.mark.parametrize('size', [2])
def test_check_size_validate(vec, size):
    with pytest.raises(ValueError):
        utils.check_size(vec, size)


def test_check_size():
    utils.check_size([1, 2, 3], 3)


@pytest.mark.parametrize(('obj', 'result'),
                         [(3, False),
                          ([3], True)])
def test_iterable(obj, result):
    assert utils.iterable(obj) == result


@pytest.mark.parametrize(('array', 'result'),
                         [(np.array([1, 1, 2]), False),
                          (np.array([1, 2, 3]), True)])
def test_has_no_repeat(array, result):
    assert utils.has_no_repeat(array) == result


def test_sizes_to_slices():
    sizes = [1, 2, 3]
    slices = [slice(0, 1), slice(1, 3), slice(3, 6)]
    result = utils.sizes_to_slices(sizes)
    assert all([result[i] == slices[i] for i in range(len(sizes))])


@pytest.mark.parametrize(('vec', 'size', 'default_value'),
                         [([], 5, 1),
                          (1, 5, None),
                          ([1]*5, 5, None)])
@pytest.mark.parametrize('result', [np.ones(5)])
def test_default_vec_factory(vec, size, default_value, result):
    my_result = utils.default_vec_factory(vec, size, default_value)
    assert np.allclose(my_result, result)


def test_get_varmat():
    np.random.seed(123)
    gamma = np.full(5, 0.1)
    obsvar = [np.full(6, (i + 1)/10) for i in range(3)]
    remat = [np.random.randn(6, 5) for i in range(3)]

    varmat = utils.get_varmat(gamma, obsvar, remat)
    assert np.isclose(varmat.logdet(), np.log(np.linalg.det(varmat.mat)))


@pytest.mark.parametrize("objs", [[[1.0, 2.0, 3.0], 2.0, (2.0, 1.0)]])
def test_get_maxlen(objs):
    assert utils.get_maxlen(objs) == 3


@pytest.mark.parametrize("objs", [[[1.0, 2.0, 3.0], 2.0, (1.0,)]])
def test_broadcast(objs):
    my_result = utils.broadcast(objs, utils.get_maxlen(objs))
    assert np.allclose(my_result,
                       np.array([[1.0, 2.0, 3.0],
                                 [2.0, 2.0, 2.0],
                                 [1.0, 1.0, 1.0]]))
