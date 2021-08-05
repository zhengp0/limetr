"""
    data
    ~~~~

    Data module.
"""
import operator
from numbers import Number
from typing import Iterable, Optional, Union

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
        Standard deviations of observation. Default is one for each observation.
    group_sizes : Vector
        Number of observations for each group. Default treat every observation
        as a group by itself.
    weight : Vector
        Weights for each observation. Default is one for each observation.
    """

    obs = property(operator.attrgetter("_obs"))
    obs_se = property(operator.attrgetter("_obs_se"))
    obs_varmat = property(operator.attrgetter("_obs_varmat"))
    weight = property(operator.attrgetter("_weight"))
    group_sizes = property(operator.attrgetter("_group_sizes"))

    def __init__(self,
                 obs: Iterable,
                 obs_se: Union[Number, Iterable] = 1.0,
                 obs_varmat: Optional[np.ndarray] = None,
                 group_sizes: Iterable[int] = None,
                 weight: Union[Number, Iterable] = 1.0):
        """
        Parameters
        ----------
        obs : Iterable
            Observations. Assumed to be sorted by the group id.
        obs_se : Union[Number, Iterable], optional
            Standard deviations of observation. Default is one.
        obs_varmat : Optional[np.ndarray], optional
            (Co)variance matrix of observation. When set to be ``None``, returns
            diagonal matrix with ``obs_se**2`` as the diagonal.
        group_sizes : Iterable[int], optional
            Number of observations for each group. Default is ``None``.
        weight : Union[Number, Iterable], optional
            Weights for each observation. Default is one.
        """
        self.obs = obs
        self.obs_se = obs_se
        self.obs_varmat = obs_varmat
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

    @obs.setter
    def obs(self, vec: Iterable):
        vec = np.asarray(vec)
        if any(np.isnan(vec)):
            raise ValueError("`obs` must not containing nan(s).")
        self._obs = vec

    @obs_se.setter
    def obs_se(self, vec: Union[Number, Iterable]):
        vec = default_vec_factory(vec, self.num_obs, vec_name="obs_se")
        if any(vec <= 0.0):
            raise ValueError("`obs_se` must be all positive.")
        self._obs_se = vec

    @obs_varmat.getter
    def obs_varmat(self) -> np.ndarray:
        if self._obs_varmat is None:
            return np.diag(self.obs_se**2)
        return self._obs_varmat

    @obs_varmat.setter
    def obs_varmat(self, mat: Optional[np.ndarray]):
        if mat is not None:
            if not isinstance(mat, np.ndarray):
                raise TypeError("obs_varmat must be None or a ndarray")
            if mat.ndim != 2 or mat.shape != (self.num_obs, self.num_obs):
                raise ValueError(f"obs_varmat must be a square matrix with "
                                 f"number of rows/cols to be {self.num_obs}.")
            if not (np.linalg.eigvals(mat) > 0.0).all():
                raise ValueError("obs_varmat must be a positive definite "
                                 "matrix.")
        self._obs_varmat = mat

    @weight.setter
    def weight(self, vec: Union[Number, Iterable]):
        vec = default_vec_factory(vec, self.num_obs, vec_name="weight")
        if any(vec < 0) or any(vec > 1):
            raise ValueError("`weight` must be all between 0 and 1.")
        self._weight = vec

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
