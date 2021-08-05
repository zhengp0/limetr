"""
Test Data Module
"""
import pytest
import numpy as np
from limetr.data import Data


# pylint:disable=redefined-outer-name


@pytest.fixture
def obs():
    return np.random.randn(5)


@pytest.fixture
def obs_se():
    return np.ones(5)


@pytest.fixture
def obs_varmat():
    np.random.seed(123)
    mat = np.random.randn(10, 5)
    return mat.T.dot(mat)


@pytest.fixture
def group_sizes():
    return np.array([2, 2, 1])


@pytest.fixture
def weight():
    return np.ones(5)


@pytest.mark.parametrize("obs_bad", [[1.0, 1.0, 1.0, 1.0, np.nan]])
def test_obs_validate(obs_bad, obs_se, group_sizes, weight):
    with pytest.raises(ValueError):
        Data(obs_bad, obs_se=obs_se, group_sizes=group_sizes, weight=weight)


@pytest.mark.parametrize("obs_se_good", [1.0, 2.0, [1.0, 2.0, 3.0, 4.0, 5.0]])
def test_obs_se(obs, obs_se_good, group_sizes, weight):
    d = Data(obs,
             obs_se=obs_se_good,
             group_sizes=group_sizes,
             weight=weight)
    assert d.obs_se.size == d.num_obs
    assert all(d.obs_se > 0.0)


@pytest.mark.parametrize("obs_se_bad", [0.0,
                                        [1.0, 2.0, 3.0, 4.0, -5.0],
                                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
def test_obs_se_validate(obs, obs_se_bad, group_sizes, weight):
    with pytest.raises(ValueError):
        Data(obs, obs_se_bad, group_sizes, weight)


@pytest.mark.parametrize("obs_varmat_good", [pytest.lazy_fixture("obs_varmat"), None])
def test_obs_varmat(obs, obs_varmat_good, group_sizes, weight):
    d = Data(obs,
             obs_varmat=obs_varmat_good,
             group_sizes=group_sizes,
             weight=weight)
    assert d.obs_se.size == d.num_obs
    assert all(d.obs_se > 0.0)


@pytest.mark.parametrize("obs_varmat_bad", [0.0, np.zeros((5, 5))])
def test_obs_varmat_validate(obs, obs_varmat_bad, group_sizes, weight):
    with pytest.raises((ValueError, TypeError)):
        Data(obs, obs_varmat=obs_varmat_bad, group_sizes=group_sizes, weight=weight)


@pytest.mark.parametrize("weight_good", [1.0, 0.0, [0.5, 0.5, 0.5, 0.5, 0.5]])
def test_weight(obs, obs_se, group_sizes, weight_good):
    d = Data(obs, obs_se=obs_se, group_sizes=group_sizes, weight=weight_good)
    assert d.weight.size == d.num_obs
    assert all(d.weight >= 0.0) and all(d.weight <= 1.0)


@pytest.mark.parametrize("weight_bad", [-1.0,
                                        [1.0, 2.0, 1.0, 1.0, 1.0],
                                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
def test_weight_validate(obs, obs_se, group_sizes, weight_bad):
    with pytest.raises(ValueError):
        Data(obs, obs_se=obs_se, group_sizes=group_sizes, weight=weight_bad)


@pytest.mark.parametrize("group_sizes_good", [None,
                                              [2, 2, 1],
                                              [2.0, 2.0, 1.0]])
def test_group_sizes(obs, obs_se, group_sizes_good, weight):
    d = Data(obs, obs_se=obs_se, group_sizes=group_sizes_good, weight=weight)
    assert d.group_sizes.size == d.num_groups
    assert sum(d.group_sizes) == d.num_obs


@pytest.mark.parametrize("group_sizes_bad", [[2, 2, 1, 1],
                                             [1, -1, 2, 2, 1]])
def test_group_sizes_validate(obs, obs_se, group_sizes_bad, weight):
    with pytest.raises(ValueError):
        Data(obs, obs_se=obs_se, group_sizes=group_sizes_bad, weight=weight)


@pytest.mark.parametrize("group_sizes_bad", [1.0])
def test_group_sizes_validate_type(obs, obs_se, group_sizes_bad, weight):
    with pytest.raises(TypeError):
        Data(obs, obs_se=obs_se, group_sizes=group_sizes_bad, weight=weight)
