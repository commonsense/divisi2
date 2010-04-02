"""
Mathematical operators that support various kinds of Divisi matrices.
"""
import numpy as np
from csc.divisi2.sparse import SparseMatrix, SparseVector, AbstractSparseArray
from csc.divisi2.dense import DenseMatrix, DenseVector, AbstractDenseArray, from_ndarray

def multiply(arg1, arg2):
    """
    Elementwise or scalar multiplication.

    Delegate to specific methods if we know about them. Otherwise, rely on
    the np.multiply ufunc to do the right thing.
    """
    if isinstance(arg1, SparseMatrix):
        if isinstance(arg2, SparseMatrix):
            return arg1._multiply_sparse(arg2)
        elif isinstance(arg2, SparseVector):
            return arg1.col_scale(arg2)
        elif isinstance(arg2, (DenseVector, np.ndarray)):
            return arg1.col_scale(arg2)
        elif isinstance(arg2, (DenseMatrix, np.ndarray)):
            return multiply(arg1.to_dense(), arg2)
        elif np.isscalar(arg2):
            return arg1.cmul(arg2)
        else:
            raise NotImplementedError

    elif isinstance(arg1, SparseVector):
        if isinstance(arg2, SparseMatrix):
            return arg2.col_scale(arg1)
        elif isinstance(arg2, SparseVector):
            return arg1._multiply_sparse(arg2)
        elif isinstance(arg2, (AbstractDenseArray, np.ndarray)):
            return multiply(arg1.to_dense(), arg2)
        elif np.isscalar(arg2):
            return arg1.cmul(arg2)
        else:
            raise NotImplementedError

    elif isinstance(arg1, AbstractDenseArray):
        if isinstance(arg2, AbstractSparseArray):
            return multiply(arg1, arg2.to_dense())
        else:
            return np.multiply(arg1, arg2)

    elif isinstance(arg1, np.ndarray):
        if isinstance(arg2, AbstractSparseArray):
            return multiply(arg1, arg2.to_dense())
        else:
            return np.multiply(arg1, arg2)

    elif np.isscalar(arg1):
        if isinstance(arg2, AbstractSparseArray):
            return arg2.cmul(arg1)
        else:
            return np.multiply(arg1, arg2)

    else:
        return np.multiply(arg1, arg2)

def divide(arg1, arg2):
    """
    Elementwise or scalar division.

    Delegate to specific methods if we know about them. Otherwise, rely on
    the np.divide ufunc to do the right thing.
    """
    if isinstance(arg2, AbstractSparseArray):
        raise ZeroDivisionError("Dividing by a sparse array doesn't make sense")

    elif isinstance(arg1, AbstractSparseArray):
        if np.isscalar(arg2):
            return arg1.cmul(1.0/arg2)
        else:
            return divide(arg1.to_dense(), arg2)

    else:
        return np.divide(arg1, arg2)

def dot(arg1, arg2):
    """
    Matrix multiplication.

    Delegate to specific methods if we know about them. Otherwise, rely on
    np.dot to do the right thing.
    """
    if isinstance(arg1, AbstractSparseArray):
        if isinstance(arg2, AbstractSparseArray):
            return arg1._dot_sparse(arg2)
        elif isinstance(arg2, (AbstractDenseArray, np.ndarray)):
            return arg1._dot_dense(arg2)
        elif np.isscalar(arg2):
            return arg1.cmul(arg2)
        else:
            raise NotImplementedError

    elif isinstance(arg1, AbstractDenseArray):
        if isinstance(arg2, AbstractSparseArray):
            return (arg2.T._dot_dense(arg1.T)).T
        elif isinstance(arg2, AbstractDenseArray):
            return arg1._dot(arg2)
        elif isinstance(arg2, np.ndarray):
            # ignore labels so that operations like .matvec work
            return np.dot(arg1, arg2)
        elif isscalar(arg2):
            return np.multiply(arg1, arg2)
        else:
            raise NotImplementedError

    elif isinstance(arg1, np.ndarray):
        if isinstance(arg2, AbstractSparseArray):
            return dot(from_ndarray(arg1), arg2)
        else:
            return np.dot(arg1, arg2)

    elif np.isscalar(arg1):
        return multiply(arg1, arg2)

    else:
        return np.dot(arg1, arg2)

def transpose_dot(arg1, arg2):
    """
    Matrix multiplication.

    Delegate to specific methods if we know about them. Otherwise, rely on
    np.dot to do the right thing.
    """
    if isinstance(arg1, SparseMatrix):
        if isinstance(arg2, SparseMatrix):
            return arg1._transpose_dot_sparse(arg2)
        elif isinstance(arg2, SparseVector):
            return arg1._transpose_dot_sparse(arg2.to_column())[:,0]
        else:
            return dot(arg1.T, arg2)
    else:
        return dot(arg1.T, arg2)
