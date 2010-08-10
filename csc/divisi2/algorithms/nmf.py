from py_nnma.nnma import FNMAI
from csc.divisi2.dense import DenseMatrix
import numpy as np

def fnmai(matrix, k, U=None, V=None, **params):
    Y = matrix.to_scipy()
    init_A = U
    if V is not None: init_X = V.T
    else: init_X = None
    A, X, obj, count, converged = FNMAI(Y, k, A=init_A, X=init_X, **params)
    sums = np.sum(A, axis=0) + np.sum(X.T, axis=0)
    sum_order = np.argsort(sums)[::-1]
    U_out = A[:, sum_order]
    V_out = X.T[:, sum_order]

    return (DenseMatrix(U_out, matrix.row_labels, None),
            DenseMatrix(V_out, matrix.col_labels, None))
