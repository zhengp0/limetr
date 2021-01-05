"""
    data
    ~~~~

    Data module.
"""
import operator
from numbers import Number
from typing import Iterable, Union

import numpy as np

from limetr.utils import default_vec_factory, iterable


# pylint: disable=too-many-instance-attributes
class Data:
    """
    Data containers observations and group information.

    Attributes
    ----------
    obs : Vector
        Observations. Assumed to be sorted by the group id.
    obs_se : Vector
        Standard deviations of observation. Default is one.
    group_sizes : Vector
        Number of observations for each group. Default is every observation is
        one group.
    weight : Vector
        Weights for each observation. Default is one.

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
            Observations. Assumed to be sorted by the group id.
        obs_se : Union[Number, Iterable], optional
            Standard deviations of observation. Default is one.
        group_sizes : Iterable[int], optional
            Number of observations for each group. Default is ``None``.
        weight : Union[Number, Iterable], optional
            Weights for each observation. Default is one.
        """
        self.obs = obs
        self.obs_se = obs_se
        self.weight = weight
        self.group_sizes = group_sizes

    @property
    def num_obs(self) -> int:
        """Number of observations"""
        return self.obs.size

    @property
    def num_groups(self) -> int:
        """Number of groups"""
        return self.group_sizes.size

    obs = property(operator.attrgetter("_obs"))

    @obs.setter
    def obs(self, vec: Iterable):
        vec = np.asarray(vec)
        if any(np.isnan(vec)):
            raise ValueError("`obs` must not containing nan(s).")
        self._obs = vec

    obs_se = property(operator.attrgetter("_obs_se"))

    @obs_se.setter
    def obs_se(self, vec: Union[Number, Iterable]):
        vec = default_vec_factory(vec, self.num_obs, vec_name="obs_se")
        if any(vec <= 0.0):
            raise ValueError("`obs_se` must be all positive.")
        self._obs_se = vec

    weight = property(operator.attrgetter("_weight"))

    @weight.setter
    def weight(self, vec: Union[Number, Iterable]):
        vec = default_vec_factory(vec, self.num_obs, vec_name="weight")
        if any(vec < 0) or any(vec > 1):
            raise ValueError("`weight` must be all between 0 and 1.")
        self._weight = vec

    group_sizes = property(operator.attrgetter("_group_sizes"))

    @group_sizes.setter
    def group_sizes(self, vec: Union[None, Iterable]):
        if vec is None:
            vec = np.ones(self.num_obs, dtype=int)
        elif iterable(vec):
            vec = np.asarray(vec).astype(int)
        else:
            raise TypeError("Use `None` or a vector to set `group_sizes`.")
        if any(vec <= 0.0):
            raise ValueError("`group_sizes` must be all positive.")
        if sum(vec) != self.num_obs:
            raise ValueError("Sum of `group_size` must equal to `num_obs`.")
        self._group_sizes = vec

    def __repr__(self) -> str:
        return f"Data(num_obs={self.num_obs}, num_groups={self.num_groups})"
