"""
    data
    ~~~~

    Data module.
"""
from typing import List
from dataclasses import dataclass, field
import numpy as np
from limetr.utils import check_size, has_no_repeat


@dataclass
class Data:
    obs: np.ndarray = field(repr=False)
    obs_se: np.ndarray = field(default=None, repr=False)
    group_sizes: np.ndarray = field(default=None, repr=False)
    index: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self.obs = np.asarray(self.obs)
        self.size = self.obs.size

        self.obs_se = np.ones(self.size) if self.obs_se is None else np.asarray(self.obs_se)
        self.group_sizes = np.array([1]*self.size) if self.group_sizes is None else np.asarray(self.group_sizes)
        self.index = np.arange(self.size) if self.index is None else np.asarray(self.index)

        self.num_groups = self.group_sizes.size

    def check_attr(self):
        check_size(self.obs, self.size, attr_name='obs')
        check_size(self.obs_se, self.size, attr_name='obs_se')
        check_size(self.group_sizes, self.num_groups, attr_name='group_sizes')
        check_size(self.index, self.size, attr_name='index')

        assert all(self.obs_se > 0.0), "Numbers in obs_se must be positive."
        assert all(self.group_sizes > 0.0), "Numbers in group_sizes must be positive."
        assert np.issubdtype(self.group_sizes.dtype, int), "Numbers in group_sizes must be integer."
        assert has_no_repeat(self.index), "Numbers in index must be unique."
