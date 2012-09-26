import scipy.sparse
import numpy as np

from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_range_finder, safe_sparse_dot

from divisi2.dense import DenseMatrix
from divisi2._svdlib import svd_ndarray


def divisi_sparse_to_scipy_sparse(matrix):
    values, rows, cols = matrix.find()
    coo_mat = scipy.sparse.coo_matrix((values, (rows, cols)))
    return coo_mat.tocsc()

def randomized_svd(matrix, k, p=10, q=5, random_state=0):
    M = divisi_sparse_to_scipy_sparse(matrix)
    U, S, V = _randomized_svd(M, k, p, q, random_state)

    U = DenseMatrix(U, matrix.row_labels, None)
    V = DenseMatrix(V, matrix.col_labels, None)

    return U, S, V

def _randomized_svd(M, k, p, q, random_state):
    random_state = check_random_state(random_state)
    
    Q = randomized_range_finder(M, k+p, q, random_state)

    # project M to the (k + p) dimensional space using the basis vectors
    B = safe_sparse_dot(Q.T, M)

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat_t, S, Vt = svd_ndarray(B, k)
    del B
    U = np.dot(Q, Uhat_t.T)

    return U, S, Vt.T
