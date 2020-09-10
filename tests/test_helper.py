# Test helper function in utils module
import numpy as np
import pytest
import limetr.utils as utils


@pytest.mark.parametrize('vec', [np.arange(6)])
@pytest.mark.parametrize('sizes', [[1, 2, 3], [3, 2, 1]])
def test_split_by_sizes(vec, sizes):
    vecs = utils.split_by_sizes(vec, sizes)
    assert all([vecs[i].size == size for i, size in enumerate(sizes)])


@pytest.mark.parametrize('weights', [np.ones(10)])
@pytest.mark.parametrize('cap', [9, 8])
def test_proj_to_capped_simplex(weights, cap):
    weights = utils.project_to_capped_simplex(weights, cap)
    assert np.allclose(weights, cap/weights.size)
