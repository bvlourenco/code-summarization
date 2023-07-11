import numpy as np
from scipy.sparse import coo_matrix
import torch

def build_tensor_from_sparse_matrix(sparse_matrix, dim):
    '''
    Given a sparse matrix in the format of csr_matrix, it converts it to a tensor.

    Args:
        sparse_matrix: The sparse matrix in csr_matrix format.
        dim: The dimension of the sparse matrix.

    Returns:
        A tensor built from the sparse matrix.
    
    Source: https://stackoverflow.com/questions/50665141/converting-a-scipy-coo-matrix-to-pytorch-sparse-tensor
    '''
    coo = coo_matrix(sparse_matrix, shape=(dim, dim))

    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()