from divisi2.dense import DenseMatrix
from divisi2.sparse import SparseMatrix
from divisi2.reconstructed import ReconstructedMatrix
from divisi2._svdlib import svd_llmat, svd_ndarray
from divisi2 import operators
import numpy as np

def svd(matrix, k=50):
    """
    Calculate the truncated singular value decomposition
    :math:`A = U * Sigma * V^T` using SVDLIBC.

    Returns a triple of:
    
    - U as a dense labeled matrix
    - S, a dense vector representing the diagonal of Sigma
    - V as a dense labeled matrix
    
    This matrix must not contain any empty rows or columns. If it does,
    use the .squish() method first.
    """
    assert matrix.ndim == 2
    if isinstance(matrix, DenseMatrix):
        Ut, S, Vt = svd_ndarray(matrix, k)
    elif isinstance(matrix, SparseMatrix):
        if matrix.nnz == 0:
            # don't let svdlib touch a matrix of all zeros. It explodes and
            # corrupts its state. Just return a zero result instead.
            U = DenseMatrix((matrix.shape[0], k))
            S = np.zeros((k,))
            V = DenseMatrix((matrix.shape[1], k))
            return U, S, V
        if matrix.shape[1] >= matrix.shape[0] * 1.2:
            # transpose the matrix for speed
            V, S, U = matrix.T.svd(k)
            return U, S, V
        Ut, S, Vt = svd_llmat(matrix.llmatrix, k)
    else:
        raise TypeError("Don't know how to SVD a %r", type(matrix))

    U = DenseMatrix(Ut.T, matrix.row_labels, None)
    V = DenseMatrix(Vt.T, matrix.col_labels, None)

    return U, S, V


