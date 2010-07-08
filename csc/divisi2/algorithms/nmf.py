from py_nnma.nnma import FNMAI
from csc.divisi2.dense import DenseMatrix

def fnmai(matrix, k, U=None, V=None, **params):
    Y = matrix.to_scipy_csr()
    init_A = U
    if V is not None: init_X = V.T
    else: init_X = None
    A, X, obj, count, converged = FNMAI(Y, k, A=init_A, X=init_X, **params)
    return (DenseMatrix(A, matrix.row_labels, None),
            DenseMatrix(X.T, matrix.col_labels, None))
