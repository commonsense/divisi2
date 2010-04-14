from csc.divisi2.ordered_set import OrderedSet, apply_indices
from csc.divisi2.labels import LabeledVectorMixin, LabeledMatrixMixin, format_label
from copy import copy
import numpy as np
import sys

SLICE_ALL = slice(None, None, None)

def from_ndarray(array):
    if array.ndim == 1:
        return DenseVector(array)
    elif array.ndim == 2:
        return DenseMatrix(array)
    else:
        raise ValueError("Don't know how to make a AbstractDenseArray with %d dimensions" % array.ndim)

class AbstractDenseArray(np.ndarray):
    """
    The abstract class that Divisi's dense representations inherit from.

    Inherits from np.ndarray to make magic happen. Magic that, hopefully, the
    numpy people have thought through a lot.
    """
    def __new__(cls, input_array, *args, **kwargs):
        raise NotImplementedError("AbstractDenseArray is an abstract class")

    def _dot(self, other):
        raise NotImplementedError
    
    def __getitem__(self, indices):
        labels = apply_indices(indices, self.all_labels())
        data = np.ndarray.__getitem__(self, indices)
        if len(labels) == 1:
            return DenseVector(data, labels[0])
        elif len(labels) == 2:
            return DenseMatrix(data, labels[0], labels[1])
        else:
            # notice that this works for >= 3 dimensions as well as 0;
            # it just gives up on labeling
            return data
    
    def multiply(self, other):
        """
        Elementwise multiplication. Ask divisi2.operators how to do it.
        """
        from csc.divisi2 import operators
        return operators.multiply(self, other)

    def dot(self, other):
        """
        Matrix or scalar multiplication. Delegate to divisi2.operators to
        sort it out.
        """
        from csc.divisi2 import operators
        return operators.dot(self, other)

    matrixmultiply = dot

    def transpose_dot(self, other):
        """
        Multiply this matrix *transposed* by another matrix. This can save
        a lot of computation when multiplying two sparse matrices.
        """
        from csc.divisi2 import operators
        return operators.transpose_dot(self, other)
    
    def equals(self, other):
        """
        Compare two matrices by value.

        This name is necessary because __eq__ is already defined as an
        elementwise operation in NumPy.
        """
        return (self.same_labels_as(other) and np.all(self == other))

class DenseVector(AbstractDenseArray, LabeledVectorMixin):
    def __new__(cls, input_array, labels=None):
        # add cases for compatibility with SparseVector's constructor
        if np.isscalar(input_array):
            input_array = np.zeros((input_array,))
        elif hasattr(input_array, 'is_sparse'):
            input_array = input_array.to_dense()
        ndarray = np.asarray(input_array)
        if ndarray.ndim != 1:
            raise ValueError("Input is not a 1-D vector")
        obj = ndarray.view(cls)
        obj.labels = OrderedSet(labels)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.labels = getattr(obj, 'labels', None)
    
    def to_sparse(self):
        """
        Get this as a SparseVector.
        """
        from csc.divisi2.sparse import SparseVector
        return SparseVector(self, self.labels)
    
    def _dot(self, other):
        result = np.dot(self, other)
        if isinstance(other, DenseVector):
            return result
        elif isinstance(other, DenseMatrix):
            return DenseVector(result, other.col_labels)
        else: raise TypeError

    def top_items(self, n=10):
        """
        Get the `n` highest-magnitude items from this vector.
        """
        if n > len(self): n = len(self)
        order = np.argsort(self)
        results = []
        for i in range(1, n+1):
            where = order[-i]
            results.append((self.label(where), self[where]))
        return results
    
    def normalize(self):
        return self / np.linalg.norm(self)
    hat = normalize

    def __reduce__(self):
        return DenseVector, (np.asarray(self), self.labels)

class DenseMatrix(AbstractDenseArray, LabeledMatrixMixin):
    def __new__(cls, input_array, row_labels=None, col_labels=None):
        # add cases for compatibility with SparseMatrix's constructor
        if isinstance(input_array, tuple):
            input_array = np.zeros(input_array)
        elif hasattr(input_array, 'is_sparse'):
            input_array = input_array.to_dense()
        ndarray = np.asarray(input_array)
        if ndarray.ndim != 2:
            raise ValueError("Input is not a 2-D matrix")
        obj = ndarray.view(cls)
        if row_labels is None:
            obj.row_labels = None
        else:
            obj.row_labels = OrderedSet(row_labels)
        if col_labels is None:
            obj.col_labels = None
        else:
            obj.col_labels = OrderedSet(col_labels)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.row_labels = getattr(obj, 'row_labels', None)
        self.col_labels = getattr(obj, 'col_labels', None)
    
    def to_sparse(self):
        """
        Get this as a SparseMatrix.
        """
        from csc.divisi2.sparse import SparseMatrix
        return SparseMatrix(self, self.row_labels, self.col_labels)

    def transpose(self):
        result = np.ndarray.transpose(self)
        result.col_labels = copy(self.row_labels)
        result.row_labels = copy(self.col_labels)
        return result
    
    def _dot(self, other):
        result = np.dot(self, other)
        if isinstance(other, DenseVector):
            return DenseVector(result, self.row_labels)
        elif isinstance(other, DenseMatrix):
            return DenseMatrix(result, self.row_labels, other.col_labels)
        else: raise TypeError
    
    def normalize_rows(self):
        norms = np.sqrt(np.sum(self*self, axis=1))[:, np.newaxis]
        return self / norms

    @property
    def T(self):
        return self.transpose()
    
    ### eigenproblems
    def svd(self, k):
        U, S, Vh = np.linalg.svd(self)
        U = DenseMatrix(U, self.row_labels, None)
        V = DenseMatrix(Vh.T, self.col_labels, None)
        return (U[:,:k], S[:k], V[:,:k])

    def spectral(self):
        raise NotImplementedError
    
    ### SVD summary
    def summarize_axis(self, axis, output=sys.stdout):
        if isinstance(axis, int):
            print >> output, "\nAxis %d" % (axis,)
            theslice = self[:, axis]

        else:
            print >> output, 'Ad-hoc axis'
            theslice = self.dot(axis)
        
        def utf8(s):
            if isinstance(s, unicode): return s.encode('utf-8')
            else: return str(s)

        for key, value in (-theslice).top_items():
            print >> output, "%+9.5f %s" % (-value, utf8(format_label(key)))
        print >> output
        for key, value in theslice.top_items()[::-1]:
            print >> output, "%+9.5f %s" % (value, utf8(format_label(key)))

    def summarize(self, k=None, output=sys.stdout):
        if k is None: k = self.shape[1]
        else: k = min(int(k), self.shape[1])
        for axis in range(k):
            self.summarize_axis(axis, output)

    def __reduce__(self):
        return DenseMatrix, (np.asarray(self), self.row_labels, self.col_labels)
