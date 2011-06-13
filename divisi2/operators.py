"""
Mathematical operators that support various kinds of Divisi matrices.
"""
import numpy as np
from divisi2.sparse import SparseMatrix, SparseVector, AbstractSparseArray
from divisi2.dense import DenseMatrix, DenseVector, AbstractDenseArray, from_ndarray
from divisi2.ordered_set import OrderedSet

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
        elif hasattr(arg2, 'matvec_transpose'):
            return arg2.matvec_transpose(arg1.T)
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
        elif hasattr(arg2, 'matvec_transpose'):
            return arg2.matvec_transpose(arg1.T)
        elif np.isscalar(arg2):
            return np.multiply(arg1, arg2)
        else:
            raise NotImplementedError

    elif isinstance(arg1, np.ndarray):
        if isinstance(arg2, AbstractSparseArray):
            return dot(from_ndarray(arg1), arg2)
        else:
            return np.dot(arg1, arg2)

    elif hasattr(arg1, 'matvec'):
        # allows multiplying ReconstructedMatrices by ad-hoc categories
        return arg1.matvec(arg2)

    elif np.isscalar(arg1):
        return multiply(arg1, arg2)

    else:
        return np.dot(arg1, arg2)

matrixmultiply = dot

def match_inner_labels(left, right):
    """
    Return (left2, right2), so that the inner labels of left2 and right2
    line up. This allows for matrix multiplication.

    See also SparseMatrix.match_labels.
    """
    left_inner = left.all_labels()[-1]
    right_inner = right.all_labels()[0]
    if left_inner is None or right_inner is None: return (left, right)
    if left_inner == right_inner: return (left, right)
    intersection = OrderedSet(set(left_inner) & set(right_inner))
    left_indices = [left_inner.index(x) for x in intersection]
    right_indices = [right_inner.index(x) for x in intersection]
    return left[..., left_indices], right[right_indices, ...]

def filter_labels(matrix, rows=None, cols=None):
    all_rows, all_cols = matrix.row_labels, matrix.col_labels
    if rows is not None:
        # Slice rows first
        row_indices = [all_rows.index(row) for row in rows]
        matrix = matrix[row_indices, ...]
    if cols is not None:
        col_indices = [all_cols.index(col) for col in cols]
        matrix = matrix[..., col_indices]
    return matrix

def aligned_matrix_multiply(arg1, arg2):
    arg1m, arg2m = match_inner_labels(arg1, arg2)
    return dot(arg1m, arg2m)

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

def projection(u, v):
    """
    The projection onto `u` of `v`.

    `u` should be a vector. Its magnitude will affect the scale of the output,
    so for a pure projection, `u` should be a unit vector.

    `v` is the vector to be projected. Following NumPy style, it can also be
    a matrix whose rows are multiple vectors to be projected.

    `u` comes first because you can think of `projection(u, ...)` as an
    operator.

    >>> u = DenseVector([0.6, 0.8])
    >>> v = DenseMatrix([[0, 1], [1, 0], [1, 1], [-3, -4]])
    >>> print projection(u, v)
    DenseMatrix (4 by 2)
     0.480000   0.640000
     0.360000   0.480000
     0.840000   1.120000
    -3.000000  -4.000000
    """
    if v.ndim == 1:
        return _vector_projection(u, v)
    elif v.ndim == 2:
        return _matrix_projection(u, v)
    else:
        raise ValueError("I don't know how to project something with %d dimensions" % v.ndim)

def _vector_projection(u, v):
    return u * dot(u.conj(), v)

def projection_residue(u, v):
    """
    Finds the projection onto `u` of the vector `v`, and returns a pair of:

    - The magnitude of the projection in the direction of `u`
    - The component of `v` that is orthogonal to `u` and therefore unprojected
    """
    un = u.normalize()
    dotprod = dot(un, v)
    return (dotprod, v - un*dotprod)

def _matrix_projection(u, v):
    dots = dot(v, u)
    return outer_product(dots, u)

def outer_product(u, v):
    """
    Take the outer product `u * v`, giving a matrix where each row represents
    an entry in `u` and each column represents an entry in `v`. The entries
    of the matrix are the products of the corresponding vector entries.
    
    >>> u = DenseVector([0.6, 0.8], 'ab')
    >>> print outer_product(u,u).to_sparse()
    SparseMatrix (2 by 2)
             a          b
    a        0.360000   0.480000
    b        0.480000   0.640000
    """
    rows = np.asarray(u)[:,np.newaxis]
    cols = np.asarray(v)[np.newaxis,:]
    return DenseMatrix(rows * cols, getattr(u, 'labels'), getattr(v, 'labels'))

