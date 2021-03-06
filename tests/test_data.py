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
def group_sizes():
    return np.array([2, 2, 1])


@pytest.fixture
def weight():
    return np.ones(5)


@pytest.mark.parametrize("obs_bad", [[1.0, 1.0, 1.0, 1.0, np.nan]])
def test_obs_validate(obs_bad, obs_se, group_sizes, weight):
    with pytest.raises(ValueError):
        Data(obs_bad, obs_se, group_sizes, weight)


@pytest.mark.parametrize("obs_se_good", [1.0, 2.0, [1.0, 2.0, 3.0, 4.0, 5.0]])
def test_obs_se(obs, obs_se_good, group_sizes, weight):
    d = Data(obs, obs_se_good, group_sizes, weight)
    assert d.obs_se.size == d.num_obs
    assert all(d.obs_se > 0.0)


@pytest.mark.parametrize("obs_se_bad", [0.0,
                                        [1.0, 2.0, 3.0, 4.0, -5.0],
                                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
def test_obs_se_validate(obs, obs_se_bad, group_sizes, weight):
    with pytest.raises(ValueError):
        Data(obs, obs_se_bad, group_sizes, weight)


@pytest.mark.parametrize("weight_good", [1.0, 0.0, [0.5, 0.5, 0.5, 0.5, 0.5]])
def test_weight(obs, obs_se, group_sizes, weight_good):
    d = Data(obs, obs_se, group_sizes, weight_good)
    assert d.weight.size == d.num_obs
    assert all(d.weight >= 0.0) and all(d.weight <= 1.0)


@pytest.mark.parametrize("weight_bad", [-1.0,
                                        [1.0, 2.0, 1.0, 1.0, 1.0],
                                        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
def test_weight_validate(obs, obs_se, group_sizes, weight_bad):
    with pytest.raises(ValueError):
        Data(obs, obs_se, group_sizes, weight_bad)


@pytest.mark.parametrize("group_sizes_good", [None,
                                              [2, 2, 1],
                                              [2.0, 2.0, 1.0]])
def test_group_sizes(obs, obs_se, group_sizes_good, weight):
    d = Data(obs, obs_se, group_sizes_good, weight)
    assert d.group_sizes.size == d.num_groups
    assert sum(d.group_sizes) == d.num_obs


@pytest.mark.parametrize("group_sizes_bad", [[2, 2, 1, 1],
                                             [1, -1, 2, 2, 1]])
def test_group_sizes_validate(obs, obs_se, group_sizes_bad, weight):
    with pytest.raises(ValueError):
        Data(obs, obs_se, group_sizes_bad, weight)


@pytest.mark.parametrize("group_sizes_bad", [1.0])
def test_group_sizes_validate_type(obs, obs_se, group_sizes_bad, weight):
    with pytest.raises(TypeError):
        Data(obs, obs_se, group_sizes_bad, weight)
