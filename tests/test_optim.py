# Test optimization module
import numpy as np
import pytest
from limetr.optim import project_to_capped_simplex


@pytest.mark.parametrize('weights', [np.ones(10)])
@pytest.mark.parametrize('cap', [9, 8])
def test_proj_to_capped_simplex(weights, cap):
    weights = project_to_capped_simplex(weights, cap)
    assert np.allclose(weights, cap/weights.size)
