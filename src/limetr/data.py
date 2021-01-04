"""
    data
    ~~~~

    Data module.
"""
from numbers import Number
from typing import Iterable, Union

import numpy as np

from limetr.utils import default_vec_factory


class Data:
    """
    Data containers observations and group information.

    Attributes
    ----------
    obs : ndarray
        Array that contains observations. Assumed to be sorted by the group id.
    obs_se : ndarray
        Array that contains standard deviations of observation. Default is 1.
    group_sizes : ndarray
        Array that contains number of observations for each group.
        Default is every observation is one group.
    weight : ndarray
        Array that contains weight for each observation. Default is 1.

    Methods
    -------
    check_attr()
        Check instance attributes and raise ``ValueError`` or ``TypeError``.
    """

    def __init__(self,
                 obs: Iterable,
                 obs_se: Union[Number, Iterable] = 1.0,
                 group_sizes: Iterable[int] = None,
                 weight: Union[Number, Iterable] = 1.0):
        """
        Parameters
        ----------
        obs : Iterable
            Array that contains observations. Assumed to be sorted by the group id.
        obs_se : Union[Number, Iterable], optional
            Array or scalar that contains standard deviations of observation.
            Default is 1.
        group_sizes : Iterable[int], optional
            Array that contains number of observations for each group.
            Default is ``None``.
        weight : Union[Number, Iterable], optional
            Array or scalar that contains weight for each observation.
            Default is 1.
        """
        self.obs = np.asarray(obs)
        self.obs_se = default_vec_factory(obs_se, self.num_obs,
                                          default_value=1.0, vec_name="obs_se")
        self.weight = default_vec_factory(weight, self.num_obs,
                                          default_value=1.0, vec_name="weight")
        if group_sizes is None:
            self.group_sizes = np.ones(self.num_obs, dtype=int)
        else:
            self.group_sizes = np.asarray(group_sizes)

        self.check_attr()

    @property
    def num_obs(self) -> int:
        """Number of observations"""
        return self.obs.size

    @property
    def num_groups(self) -> int:
        """Number of groups"""
        return self.group_sizes.size

    def check_attr(self):
        """
        Raises
        ------
        ValueError
            If ``self.obs`` contains nan(s).
        ValueError
            If ``self.obs_se`` has non-positive values.
        TypeError
            If ``self.group_sizes`` is not integer type.
        ValueError
            If ``self.group_sizes`` has non-positive values.
        ValueError
            If ``self.weight`` has values less than 0 or greater than 1.
        """
        if any(np.isnan(self.obs)):
            raise ValueError("`obs` contains nan(s).")
        if not all(self.obs_se > 0.0):
            raise ValueError("`obs_se` must be all positive.")
        if not np.issubdtype(self.group_sizes.dtype, int):
            raise TypeError("`group_size` must be all integers.")
        if not all(self.group_sizes > 0.0):
            raise ValueError("`group_sizes` must be all positive.")
        if not (sum(self.group_sizes) == self.num_obs):
            raise ValueError("`group_size` not consistent with `num_obs`.")
        if not (all(self.weight >= 0) and all(self.weight <= 1)):
            raise ValueError("`weight` must be all between 0 and 1.")

    def __repr__(self) -> str:
        return f"Data(num_obs={self.num_obs}, num_groups={self.num_groups})"
