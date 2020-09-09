# Test helper function in utils module
import numpy as np
import pytest
import limetr.utils as utils


@pytest.mark.parametrize('vec', [np.arange(6)])
@pytest.mark.parametrize('sizes', [[1, 2, 3], [3, 2, 1]])
def test_split_by_sizes(vec, sizes):
    vecs = utils.split_by_sizes(vec, sizes)
    assert all([vecs[i].size == size for i, size in enumerate(sizes)])
