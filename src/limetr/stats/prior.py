"""
Prior Module
"""
from numbers import Number
from typing import Iterable, List, Union, Any

import numpy as np
from limetr.utils import broadcast, get_maxlen


class Prior:
    """
    Generic prior class, need to be inherited.

    Attributes
    ----------
    info : ndarray
        Information array of the prior.
    size : int
        Size of the prior.

    Methods
    -------
    objective(var)
        Objective function of optimization interface
    gradient(var)
        Gradient function of optimization interface
    hessian(var)
        Hessian function of the optimization interface
    """

    def __init__(self,
                 info: List[Any] = None,
                 size: int = 0):
        """
        Parameters
        ----------
        info : List[Any]
            Information of the prior. Length of the list is number of
            information components. Each component can be either scalar or a
            vector. If there are more than one vector with size more than one,
            the size of the vectors need to match. By default ``None``.
        size : int, optional
            Size of the prior, by default 0.
        """
        size = max(int(size), get_maxlen(info))
        info = broadcast(info, size)
        self.info = info
        self.size = size

    @property
    def is_empty(self) -> bool:
        """
        Returns
        -------
        bool
            If prior has zero size.
        """
        return self.size == 0

    # pylint:disable=unused-argument
    # pylint:disable=no-self-use
    def objective(self, var: np.ndarray) -> float:
        """
        Objective function for optimiation interface.

        Parameters
        ----------
        var : np.ndarray
            Variable that prior is acting on.

        Returns
        -------
        float
            Objective value regarding the log likelihood of the prior.
        """
        return 0.0

    def gradient(self, var: np.ndarray) -> np.ndarray:
        """
        Gradient function for optimiation interface.

        Parameters
        ----------
        var : ndarray
            Variable that prior is acting on.

        Returns
        -------
        ndarray
            Gradient value regarding the log likelihood of the prior.
        """
        return np.zeros(len(var))

    def hessian(self, var: np.ndarray) -> np.ndarray:
        """
        Hessian function for optimiation interface.

        Parameters
        ----------
        var : ndarray
            Variable that prior is acting on.

        Returns
        -------
        ndarray
            Hessian value regarding the log likelihood of the prior.
        """
        return np.zeros((len(var), len(var)))

    def __repr__(self) -> str:
        return f"Prior(size={self.size})"


class GaussianPrior(Prior):
    """
    Gaussian Prior

    Attributes
    ----------
        mean : ndarray
            Mean vector of the prior.
        sd : ndarray
            Standard deviation vector of the prior.
    """

    def __init__(self,
                 mean: Union[Number, Iterable] = 0.0,
                 sd: Union[Number, Iterable] = np.inf,
                 size: int = 0):
        """
        Parameters
        ----------
        mean : Union[Number, Iterable], optional
            Mean of the Gaussian prior, by default 0.
        sd : Union[Number, Iterable], optional
            Standard deviation of the Gaussian prior, by default inf.
        size : int, optional
            Size of the prior, by default 0.

        Raises
        ------
        ValueError
            If any standard deviations are less or equal to zero.
        """
        super().__init__([mean, sd], size=size)
        if not all(self.info[1] > 0):
            raise ValueError("Standard deviations have to be positive numbers.")
        self.mean = self.info[0]
        self.sd = self.info[1]

    def objective(self, var: np.ndarray) -> float:
        return 0.5*np.sum((var - self.mean)**2/self.sd**2)

    def gradient(self, var: np.ndarray) -> np.ndarray:
        return (var - self.mean)/self.sd**2

    # pylint: disable=unused-argument
    def hessian(self, var: np.ndarray) -> np.ndarray:
        return np.diag(1/self.sd**2)

    def __repr__(self) -> str:
        return f"GaussianPrior(mean={self.mean}, sd={self.sd})"


class UniformPrior(Prior):
    """
    Uniform Prior

    Attributes
    ----------
        lb : ndarray
            Lower bounds of the prior.
        ub : ndarray
            Upper bounds of the prior.
    """

    def __init__(self,
                 lb: Union[Number, Iterable] = -np.inf,
                 ub: Union[Number, Iterable] = np.inf,
                 size: int = 0):
        """
        Parameters
        ----------
        lb : Union[Number, Iterable], optional
            Lower bounds of the prior, by default -inf
        ub : Union[Number, Iterable], optional
            Upper bounds of the prior, by default inf
        size : int, optional
            Size of the prior, by default 0

        Raises
        ------
        ValueError
            If any lower bounds are greater than upper bounds.
        """
        super().__init__([lb, ub], size=size)
        if any(self.info[0] > self.info[1]):
            raise ValueError("Lower bounds must be less or equal than upper bounds.")
        self.lb = self.info[0]
        self.ub = self.info[1]

    def __repr__(self) -> str:
        return f"UniformPrior(lb={self.lb}, ub={self.ub})"


class LinearPrior(Prior):
    """
    Linear Prior

    Attributes
    ----------
    mat: ndarray
        Linear mapping (matrix).
    """

    def __init__(self,
                 mat: Iterable,
                 info: List[Any]):
        """
        Parameters
        ----------
        mat : Iterable
            Linear mapping (matrix).
        info : List[Any]
            Information array of the prior.
        """
        mat = np.asarray(mat)
        if mat.ndim != 2:
            raise ValueError("`mat` has to be a matrix.")
        Prior.__init__(self, info, mat.shape[0])
        self.mat = mat

    def __repr__(self) -> str:
        return f"LinearPrior(shape={self.mat.shape})"


class LinearGaussianPrior(LinearPrior, GaussianPrior):
    """
    Linear Gaussian Prior
    """

    def __init__(self,
                 mat: Iterable,
                 mean: Union[Number, Iterable] = 0.0,
                 sd: Union[Number, Iterable] = np.inf):
        """
        Parameters
        ----------
        mat: Iterable
            Linear mapping(matrix).
        mean: Union[Number, Iterable], optional
            Mean of the prior, by default tuple with zero length.
        sd: Union[Number, Iterable], optional
            Standard deviation of the prior, by default tuple with zero length.
        """
        LinearPrior.__init__(self, mat, [mean, sd])
        GaussianPrior.__init__(self, self.info[0], self.info[1], size=self.size)

    def objective(self, var: np.ndarray) -> float:
        trans_var = self.mat.dot(var)
        return super().objective(trans_var)

    def gradient(self, var: np.ndarray) -> np.ndarray:
        trans_var = self.mat.dot(var)
        return self.mat.T.dot(super().gradient(trans_var))

    def hessian(self, var: np.ndarray) -> np.ndarray:
        trans_var = self.mat.dot(var)
        return self.mat.T.dot(super().hessian(trans_var).dot(self.mat))

    def __repr__(self) -> str:
        return f"LinearGaussianPrior(mean={self.mean}, sd={self.sd}, shape={self.mat.shape})"


class LinearUniformPrior(LinearPrior, UniformPrior):
    """
    Linear Uniform Prior
    """

    def __init__(self,
                 mat: Iterable,
                 lb: Union[Number, Iterable] = -np.inf,
                 ub: Union[Number, Iterable] = np.inf):
        """
        Parameters
        ----------
        mat: Iterable
            Linear mapping(matrix).
        lb: Union[Number, Iterable], optional
            Lower bounds of the prior, by default tuple with zero length.
        ub: Union[Number, Iterable], optional
            Upper bounds of the prior, by default tuple with zero length.
        """
        LinearPrior.__init__(self, mat, [lb, ub])
        UniformPrior.__init__(self, self.info[0], self.info[1], size=self.size)

    def __repr__(self) -> str:
        return f"LinearUniformPrior(lb={self.lb}, ub={self.ub}, shape={self.mat.shape})"
