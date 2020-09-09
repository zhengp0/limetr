"""
    blk_diag_mat
    ~~~~~~~~~~~~

    Block diagonal matrices module.
"""
from typing import List
import numpy as np


class BlockDiagMat:
    def __init__(self, mat_blocks: List[np.ndarray]):
        self.mat_blocks = mat_blocks
        self.check_blocks()
        self.block_shapes = [mat.shape for mat in self.mat_blocks]
        self.block_sizes = [mat.size for mat in self.mat_blocks]
        self.shape = tuple(sum(size) for size in zip(*self.block_shapes))
        self.size = np.prod(self.shape)
        self.active_size = np.sum(self.block_sizes)

    def check_blocks(self):
        assert all([isinstance(mat, np.ndarray) for mat in self.mat_blocks])
        assert all([mat.ndim == 2 for mat in self.mat_blocks])