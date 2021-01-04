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
def test_check_size_assert(vec, size):
    with pytest.raises(AssertionError):
        utils.check_size(vec, size)


def test_check_size():
    utils.check_size([1, 2, 3], 3)


@pytest.mark.parametrize(('obj', 'result'),
                         [(3, False),
                          ([3], True)])
def test_iterable(obj, result):
    assert utils.iterable(obj) == result
