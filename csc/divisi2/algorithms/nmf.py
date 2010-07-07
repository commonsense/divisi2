from py_nnma.nnma import FNMAI

def fnmai(matrix, k, U=None, V=None, **param):
    Y = matrix.to_scipy_csr()
    A, X, obj, count, converged = FNMAI(Y, k, A=U, X=V.T, **param)
    # A is our U; X is our V.T
    return A, X.T
