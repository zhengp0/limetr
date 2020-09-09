"""
    blk_diag_mat
    ~~~~~~~~~~~~

    Block diagonal matrices module.
"""
from __future__ import annotations
from typing import List, Any
import numpy as np
import scipy.linalg as spla
from .helper_funs import split_by_sizes


class BlockDiagMat:
    def __init__(self, mat_blocks: List[np.ndarray]):
        self.mat_blocks = mat_blocks
        self._check_blocks()
        self.block_shapes = [mat.shape for mat in self.mat_blocks]
        self.block_sizes = [mat.size for mat in self.mat_blocks]
        self.block_row_sizes = [shape[0] for shape in self.block_shapes]
        self.block_col_sizes = [shape[1] for shape in self.block_shapes]
        self.shape = tuple(sum(size) for size in zip(*self.block_shapes))
        self.size = np.prod(self.shape)
        self.active_size = np.sum(self.block_sizes)

    def _check_blocks(self):
        assert all([isinstance(mat, np.ndarray) for mat in self.mat_blocks])
        assert all([mat.ndim == 2 for mat in self.mat_blocks])

    def full(self) -> np.ndarray:
        return spla.block_diag(*self.mat_blocks)

    def dot(self, array: np.ndarray) -> np.ndarray:
        arrays = split_by_sizes(array, self.block_col_sizes)
        return np.concatenate([mat.dot(arrays[i])
                               for i, mat in enumerate(self.mat_blocks)], axis=0)


class SquareBlockDiagMat(BlockDiagMat):
    def __init__(self, mat_blocks: List[np.ndarray]):
        super().__init__(mat_blocks)
        self._check_square()
        self.block_side_sizes = [shape[0] for shape in self.block_shapes]
        self.side_size = np.sum(self.block_side_sizes)

    def _check_square(self):
        assert all([shape[0] == shape[1] for shape in self.block_shapes])

    def inv(self) -> SquareBlockDiagMat:
        inv_mat_blocks = [np.linalg.inv(mat) for mat in self.mat_blocks]
        return SquareBlockDiagMat(inv_mat_blocks)

    def invdot(self, array: np.ndarray) -> np.ndarray:
        arrays = split_by_sizes(array, self.block_col_sizes)
        return np.concatenate([np.linalg.solve(mat, arrays[i])
                               for i, mat in enumerate(self.mat_blocks)], axis=0)

    def diag(self) -> np.ndarray:
        return np.hstack([np.diag(mat) for mat in self.mat_blocks])

    def block_eigvals(self) -> np.ndarray:
        return np.hstack([np.linalg.eigvals(mat) for mat in self.mat_blocks])

    def det(self) -> Any:
        return np.prod(self.block_eigvals())

    def logdet(self) -> Any:
        return np.sum(np.log(self.block_eigvals()))
