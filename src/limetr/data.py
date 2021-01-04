"""
    data
    ~~~~

    Data module.
"""
from dataclasses import dataclass, field
import numpy as np
from limetr.utils import check_size


@dataclass
class Data:
    obs: np.ndarray = field(repr=False)
    obs_se: np.ndarray = field(default=None, repr=False)
    group_sizes: np.ndarray = field(default=None, repr=False)
    weight: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self.obs = np.asarray(self.obs)
        self.num_obs = self.obs.size

        self.obs_se = np.ones(self.num_obs) if self.obs_se is None else np.asarray(self.obs_se)
        self.group_sizes = np.array([1]*self.num_obs) if self.group_sizes is None else np.asarray(self.group_sizes)
        self.weight = np.ones(self.num_obs) if self.weight is None else self.weight

        self.num_groups = self.group_sizes.size

    def check_attr(self):
        check_size(self.obs, self.num_obs, vec_name='obs')
        check_size(self.obs_se, self.num_obs, vec_name='obs_se')
        check_size(self.group_sizes, self.num_groups, vec_name='group_sizes')
        check_size(self.weight, self.num_obs, vec_name="weight")

        assert all(self.obs_se > 0.0), "Numbers in obs_se must be positive."
        assert all(self.group_sizes > 0.0), "Numbers in group_sizes must be positive."
        assert np.issubdtype(self.group_sizes.dtype, int), "Numbers in group_sizes must be integer."
        assert all(self.weight >= 0) and all(self.weight <= 1), "weight must be between 0 and 1."

    def __repr__(self) -> str:
        return f"Data(num_obs={self.num_obs}, num_groups={self.num_groups})"
