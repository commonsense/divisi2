import numpy as np
from divisi2.dense import DenseVector, DenseMatrix
from divisi2.ordered_set import OrderedSet, indexable_set, apply_indices
from divisi2.exceptions import LabelError, DimensionMismatch
from divisi2.labels import LabeledVectorMixin, LabeledMatrixMixin, format_label
from divisi2.algorithms import LearningMixin
from pysparse.sparse import spmatrix
from pysparse.sparse.pysparseMatrix import PysparseMatrix
from copy import copy
from collections import defaultdict
import warnings

SLICE_ALL = slice(None, None, None)

def _dense_row(psmatrix):
    """
    Convert a 1xN PysparseMatrix into a NumPy vector.
    """
    vec = np.zeros((psmatrix.shape[1],))
    vec.put(psmatrix.matrix.keys()[1], psmatrix.matrix.values())
    return vec

def _dense_col(psmatrix):
    """
    Convert a Nx1 PysparseMatrix into a NumPy vector.
    """
    vec = np.zeros((psmatrix.shape[0],))
    vec.put(psmatrix.matrix.keys()[0], psmatrix.matrix.values())
    return vec

def _inv_norm(vec):
    """
    Return the inverse of the Euclidean norm of a vector. Useful in
    normalizing.
    """
    return 1.0/np.linalg.norm(vec)

def _inv_root_norm(vec):
    """
    Return the inverse square root of the Euclidean norm of a vector.
    Useful in normalizing.
    """
    return 1.0/np.sqrt(np.linalg.norm(vec))

def _ndarray_to_sparse(ndarray):
    """
    Convert a dense numpy.ndarray to a PysparseMatrix.
    """
    if ndarray.ndim == 1:
        ndarray = ndarray[np.newaxis, :]
    assert ndarray.ndim == 2
    psmatrix = PysparseMatrix(nrow=ndarray.shape[0], ncol=ndarray.shape[1])
    xcoords, ycoords = np.indices(ndarray.shape)
    psmatrix.put(ndarray.flatten(), xcoords.flatten(), ycoords.flatten())
    return psmatrix

class AbstractSparseArray(object):
    """
    An abstract base class for operations that are common to both sparse
    representations -- mostly operators.
    """
    is_sparse = True

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("AbstractSparseArray is an abstract class")

    def __add__(self, other):
        return self._add_sparse(other)

    def __iadd__(self, other):
        return self._iadd_sparse(other)

    def __sub__(self, other):
        return self + (-other)

    def __isub__(self, other):
        self += (-other)
        return self

    def __neg__(self):
        return self * -1

    def __mul__(self, other):
        """
        Scalar multiplication.

        If `other` is not a scalar, it's ambiguous whether you want matrix
        or elementwise multiplication, so it raises an error.
        """
        if np.isscalar(other):
            return self.cmul(other)
        else:
            raise ValueError(
              "The * operator is ambiguous. "
              "Please use divisi2.multiply for elementwise multiplication, or "
              "divisi2.dot for matrix multiplication."
            )

    def __rmul__(self, other):
        """
        Handle the case where we're on the right side of a scalar
        multiplication, through divisi2.operators.
        """
        from divisi2 import operators
        return operators.multiply(other, self)

    def __div__(self, other):
        """
        Divide by something (probably a constant). Delegate to divisi2.operators
        to figure it out.
        """
        from divisi2 import operators
        return operators.divide(self, other)
    
    def __rdiv__(self, other):
        """
        Divide something else by self. On the off chance that this is possible,
        divisi2.operators should do the right thing.
        """
        from divisi2 import operators
        return operators.divide(other, self)

    def __imul__(self, scalefactor):
        """
        Multiply in-place by a scalar.
        """
        if not np.isscalar(scalefactor):
            raise TypeError('In-place multiplication requires a scalar.')
        self.psmatrix *= scalefactor
        return self

    def __idiv__(self, scalefactor):
        """
        Divide in-place by a scalar.
        """
        if not np.isscalar(scalefactor):
            raise TypeError('In-place division requires a scalar.')
        self.psmatrix *= 1.0/scalefactor
        return self

    def multiply(self, other):
        """
        Elementwise multiplication. Ask divisi2.operators how to do it.
        """
        from divisi2 import operators
        return operators.multiply(self, other)

    def dot(self, other):
        """
        Matrix or scalar multiplication. Delegate to divisi2.operators to
        sort it out.
        """
        from divisi2 import operators
        return operators.dot(self, other)

    matrixmultiply = dot

    def transpose_dot(self, other):
        """
        Multiply this matrix *transposed* by another matrix. This can save
        a lot of computation when multiplying two sparse matrices.
        """
        from divisi2 import operators
        return operators.transpose_dot(self, other)

    def cmul(self, scalefactor):
        """
        Multiplication by a scalar.
        """
        return self.replacedata(self.psmatrix * scalefactor)
    
    def __eq__(self, other):
        """
        Compare two matrices by value.
        """
        return (type(other) == type(self) and
                other.keys() == self.keys() and
                np.allclose(other.value_array(), self.value_array()) and
                self.same_labels_as(other))

    def __ne__(self, other):
        return not self.__eq__(other)
    
    # support the same interface as dense
    equals = __eq__

    def __str__(self):
        return unicode(self).encode('utf-8')

    @property
    def llmatrix(self):
        """
        Get the PySparse list-of-lists matrix, which is the lowest-level C
        representation of this matrix or vector.
        """
        return self.psmatrix.matrix
    
    def to_sparse(self):
        """
        :meth:`to_sparse` on a sparse array is the identity.
        """
        return self

class SparseMatrix(AbstractSparseArray, LabeledMatrixMixin, LearningMixin):
    """
    A SparseMatrix is a matrix (a 2-D array) in which only the non-zero values
    are represented.

    If you take a single row or column of a SparseMatrix, you get a
    :class:`SparseVector`. In this way, the behavior of a SparseMatrix is more
    like NumPy's ``ndarray`` than its ``matrix`` class.

    ``ndarray`` and ``matrix`` are infamously inconsistent about what the ``*``
    operator means. To an ``ndarray``, ``*`` means scalar or elementwise
    multiplication.  To a ``matrix``, ``*`` means scalar or matrix
    multiplication. The way that SparseMatrix resolves this is to take
    *neither* side: ``*`` is only used for scalar multiplication. To multiply
    elementwise, use :func:`divisi2.multiply`, and to multiply matrices,
    use :func:`divisi2.dot`.

    Divisi2 currently uses PysparseMatrix as its underlying sparse
    representation. This allows sparse matrices to be indexed and sliced just
    like NumPy's dense matrices.
    """

    def __init__(self, arg1, row_labels=None, col_labels=None):
        """
        Create a SparseMatrix given appropriate arguments.

        arg1 can be a PysparseMatrix object, something that can be turned
        into a PysparseMatrix object (such as an ndarray), or a tuple
        of (rows, columns) for creating a zero matrix.

        Creating a zero matrix:
        
        >>> z = SparseMatrix((2, 4))
        >>> print z
        SparseMatrix (2 by 4)
            ---        ---        ---        ---
            ---        ---        ---        ---


        Creating a sparse matrix from dense list data (not advisable):
        
        >>> print SparseMatrix([[1,2], [3, 0]])
        SparseMatrix (2 by 2)
         1.000000   2.000000
         3.000000      ---
        >>> foobar = SparseMatrix([[1,2], [3, 0]], ['foo', 'bar'], None)
        >>> print foobar
        SparseMatrix (2 by 2)
        foo      1.000000   2.000000
        bar      3.000000      ---
        >>> foobar.nnz
        3

        Passing an existing SparseMatrix copies its values (but not
        its labels, unless you ask):

        >>> foobar2 = SparseMatrix(foobar)
        >>> foobar2 == foobar
        False
        >>> foobar2 == foobar.unlabeled()
        True
        
        This is a copy, so the original matrix remains unchanged:

        >>> foobar2[0,0] = 5
        >>> foobar[0,0]
        1.0

        If you ask for a matrix, you'll get a matrix, even if the underlying
        data is a vector.

        >>> SparseMatrix(np.arange(10))
        <SparseMatrix (1 by 10)>
        """
        if isinstance(arg1, PysparseMatrix):
            psmatrix = arg1
        elif isinstance(arg1, tuple):
            assert len(arg1) == 2
            nrow, ncol = arg1
            psmatrix = PysparseMatrix(nrow=nrow, ncol=ncol)
        elif isinstance(arg1, (list, tuple)):
            # First turn a sequence into a NumPy array, which we know how
            # to handle (in the next step).
            psmatrix = _ndarray_to_sparse(np.array(arg1))
        elif isinstance(arg1, np.ndarray):
            # Given a NumPy array, turn it into sparse data.
            psmatrix = _ndarray_to_sparse(arg1)
        elif isinstance(arg1, AbstractSparseArray):
            psmatrix = arg1.psmatrix.copy()
        else:
            raise TypeError("I don't know how to construct a SparseMatrix from a %s" % type(arg1))
        # Now, assuming we were given anything reasonable, we should have
        # a PysparseMatrix as the input data.
        assert isinstance(psmatrix, PysparseMatrix)
        self.psmatrix = psmatrix
        if row_labels is None: self.row_labels = None
        else: self.row_labels = indexable_set(row_labels)
        if col_labels is None: self.col_labels = None
        else: self.col_labels = indexable_set(col_labels)
        self._setup_wrapped_methods()
    
    ### numpy-like properties

    @property
    def shape(self):
        """Returns the height and width of this matrix."""
        return self.psmatrix.shape

    @property
    def nnz(self):
        """Returns the number of non-zero entries in this matrix."""
        return self.psmatrix.nnz

    ndim = 2

    ### factory methods

    @staticmethod
    def from_lists(values, rows, cols, nrows=None, ncols=None):
        """
        Create a new SparseMatrix from sparse data.  The data should be
        expressed as three parallel lists, containing the values and the
        corresponding rows and columns that they go into.

        Optional arguments `nrows` and `ncols` specify the shape the resulting
        matrix should have, in case it is larger than the largest index.
        """
        if nrows is None: nrows = np.max(rows)+1
        if ncols is None: ncols = np.max(cols)+1
        sparse = PysparseMatrix(nrow=nrows, ncol=ncols)
        if isinstance(values, tuple): values = list(values)
        if isinstance(rows, tuple): rows = list(rows)
        if isinstance(cols, tuple): cols = list(cols)
        sparse.addAt(values, rows, cols)

        return SparseMatrix(sparse)
    
    @staticmethod
    def from_entries(tuples):
        """
        Create a new SparseMatrix from a list of tuples. Each tuple is
        of the form (value, row, col), expressing a value and where it goes
        in the matrix.

        If possible, use ``from_lists``, because it's faster.
        """
        return SparseMatrix.from_lists(*zip(*tuples))

    @staticmethod
    def from_named_lists(values, rownames, colnames, row_labels=None, col_labels=None):
        """
        Constructs a SparseMatrix similarly to :meth:`from_lists`,
        but you specify the *labels* of rows and columns, not their
        indices.

        Optionally, you can provide existing label lists or
        OrderedSets to use for the row and column label lists.
        """
        if row_labels is None: row_labels = OrderedSet()
        if col_labels is None: col_labels = OrderedSet()
        # Ensure that the labels are indeed OrderedSets.
        row_labels = indexable_set(row_labels)
        col_labels = indexable_set(col_labels)
        # Ensure all labels are present.
        row_labels.extend(rownames)
        col_labels.extend(colnames)
        # Look up indices.
        rows = [row_labels.index(name) for name in rownames]
        cols = [col_labels.index(name) for name in colnames]
        # Construct matrix.
        result = SparseMatrix.from_lists(values, rows, cols,
                                         nrows=len(row_labels),
                                         ncols=len(col_labels))
        result.row_labels = row_labels
        result.col_labels = col_labels
        return result
    
    @staticmethod
    def square_from_named_lists(values, rownames, colnames, labels=None):
        """
        Constructs a SparseMatrix similarly to :meth:`from_named_lists`,
        but ensures that the resulting matrix is square and has the same
        row and column labels.

        The ``row_labels`` and ``col_labels`` properties of the resulting
        matrix will actually end up aliased to each other.
        """
        if labels is None: labels = OrderedSet()
        # Ensure that the labels are indeed an OrderedSet.
        labels = indexable_set(labels)
        # Ensure all labels are present.
        labels.extend(rownames)
        labels.extend(colnames)
        # Look up indices.
        rows = [labels.index(name) for name in rownames]
        cols = [labels.index(name) for name in colnames]
        # Construct matrix.
        result = SparseMatrix.from_lists(values, rows, cols,
                                         nrows=len(labels),
                                         ncols=len(labels))
        result.row_labels = labels
        result.col_labels = labels
        return result

    @staticmethod
    def from_named_entries(tuples):
        """
        Create a new SparseMatrix from a list of tuples. Each tuple is
        of the form (value, rowname, colname), expressing a value and the
        labels for where it goes in the matrix.

        The ``row_labels`` and ``col_labels`` properties of the resulting
        matrix will actually end up aliased to each other.
        
        If possible, use ``from_named_lists``, because it's faster.
        """
        lists = zip(*tuples)
        if not lists: return SparseMatrix((0,0))
        return SparseMatrix.from_named_lists(*lists)
    
    @staticmethod
    def square_from_named_entries(tuples, labels=None):
        """
        Create a new SparseMatrix from a list of tuples. Each tuple is
        of the form (value, rowname, colname), expressing a value and the
        labels for where it goes in the matrix. Ensure that the matrix is
        square and has the same row and column labels.

        If possible, use ``square_from_named_lists``, because it's faster.
        
        Example:

        >>> mat1 = SparseMatrix.square_from_named_entries([
        ...     (2, "apple", "red"),
        ...     (2, "orange", "orange"),
        ...     (1, "apple", "green"),
        ...     (1, "celery", "green"),
        ... ])
        >>> print mat1
        SparseMatrix (5 by 5)
                 apple      orange     celery     red        green
        apple       ---        ---        ---     2.000000   1.000000
        orange      ---     2.000000      ---        ---        ---
        celery      ---        ---        ---        ---     1.000000
        red         ---        ---        ---        ---        ---
        green       ---        ---        ---        ---        ---
        """
        lists = zip(*tuples)
        if not lists: return SparseMatrix((0,0))
        lists = lists + [labels]
        return SparseMatrix.square_from_named_lists(*lists)
    
    ### basic operations

    def data(self):
        """
        Get the underlying data of this matrix as a PysparseMatrix object.
        """
        return self.psmatrix

    def copy(self):
        """
        Return a copy of this matrix.
        """
        # labels are automatically copied in the constructor
        return SparseMatrix(self.psmatrix.copy(), self.row_labels,
                                  self.col_labels)
    
    def replacedata(self, newsparse):
        """
        Make a sparse matrix with the same labels, but different data. Useful
        for applying an operation to the underlying PysparseMatrix.
        """
        assert isinstance(newsparse, PysparseMatrix)
        return SparseMatrix(newsparse, self.row_labels, self.col_labels)

    def transpose(self):
        vals, rows, cols = self.psmatrix.find()
        result = SparseMatrix.from_lists(vals, cols, rows, self.shape[1], self.shape[0])
        result.col_labels = copy(self.row_labels)
        result.row_labels = copy(self.col_labels)
        return result
    
    @property
    def T(self):
        return self.transpose()

    def sym(self):
        """
        Get the symmetric self-similarity matrix, A*A^T, of this matrix,
        as a ReconstructedMatrix.

        If this matrix (A) has shape m by n, the result will have shape
        m by m.
        """
        from divisi2.reconstructed import ReconstructedMatrix
        return ReconstructedMatrix(self, self.T)

    def density(self):
        """
        Calculate how dense the matrix is.

        Returns (num specified elements)/(num possible elements).
        """
        return float(self.nnz) / self.shape[0] / self.shape[1]
    
    ### operations that export the data in another format

    def entries(self):
        """
        Get a list of (value, row, col) tuples that describe the content of
        this matrix. `row` and `col` will be indices, not labels.

        This format can be used to construct a new matrix using
        :meth:`SparseMatrix.from_entries`.
        """
        return zip(*self.lists())

    def named_entries(self):
        """
        Get a list of (value, row_name, col_name) tuples that describe the
        content of this matrix.

        This format can be used to construct a new matrix using
        :meth:`SparseMatrix.from_named_entries`.
        """
        return [(val, self.row_label(i), self.col_label(j))
                for (val, i, j) in self.entries()]

    def lists(self):
        """
        Get the content of this matrix as three lists: a list of values,
        a list of row indices, and a list of column indices, so that the
        matrix could be reconstructed using
        :meth:`SparseMatrix.from_lists`.
        """
        values, rows, cols = self.find()
        return (list(float(v) for v in values),
                list(int(r) for r in rows),
                list(int(c) for c in cols))

    def named_lists(self):
        """
        Get the content of this matrix as three lists: a list of values,
        a list of row labels, and a list of column labels, so that the matrix
        could be reconstructed using
        :meth:`SparseMatrix.from_named_lists`.
        """
        values, rows, cols = self.find()
        named_rows = [self.row_label(row) for row in rows]
        named_cols = [self.col_label(col) for col in cols]
        return values, named_rows, named_cols

    def to_dense(self):
        """
        Convert this to a :class:`DenseMatrix`.
        """
        data = self.psmatrix.getNumpyArray()
        return DenseMatrix(data, self.row_labels, self.col_labels)
    
    ### indexing
    
    def __getitem__(self, indices):
        # workaround for psmatrix index glitch
        if not isinstance(indices, tuple):
            indices = (indices,)
        
        if Ellipsis in indices:
            ps_list = list(indices)
            index_deficit = 2 - (len(ps_list)-1)
            for index in xrange(len(ps_list)):
                if ps_list[index] is Ellipsis:
                    ps_list[index:index+1] = [SLICE_ALL] * index_deficit
                    break
            indices = tuple(ps_list)
        
        if len(indices) < 2:
            indices += (SLICE_ALL,) * (2 - len(indices))

        labels = apply_indices(indices, self.all_labels())
        data = self.psmatrix[indices]
        if len(labels) == 1:
            if data.shape[0] == 1:
                return SparseVector(data, labels[0])
            elif data.shape[1] == 1:
                return SparseVector(data.getNumpyArray()[:,0], labels[0])
            else:
                assert False
        elif len(labels) == 2:
            return SparseMatrix(data, labels[0], labels[1])
        else:
            return data

    def __setitem__(self, indices, targetdata):
        if not isinstance(indices, (list, tuple)):
            indices = (indices, SLICE_ALL)
        if isinstance(indices, (list, tuple)) and len(indices) == 1:
            indices = (indices[0], SLICE_ALL)

        if len(indices) != 2:
            raise IndexError("Need 1 or 2 indices, got %d" % len(indices))

        if isinstance(targetdata, SparseMatrix):
            if targetdata.row_labels is not None and targetdata.col_labels is not None:
                warnings.warn('Not yet checking that the labels make sense!')
            sparsedata = targetdata.psmatrix
        if isinstance(targetdata, SparseVector):
            if targetdata.labels is not None:
                warnings.warn('Not yet checking that the labels make sense!')
            sparsedata = targetdata.psmatrix
        elif isinstance(targetdata, np.ndarray):
            sparsedata = _ndarray_to_sparse(targetdata)
        elif isinstance(targetdata, PysparseMatrix):
            sparsedata = targetdata
        elif isinstance(targetdata, (list, tuple)):
            nd = np.array(targetdata)
            sparsedata = _ndarray_to_sparse(nd)
        elif isinstance(targetdata, (int, long, float)):
            sparsedata = targetdata
        else:
            raise TypeError("Don't know how to assign from %s" % type(targetdata))
        self.psmatrix[indices] = sparsedata
    
    def match_labels(self, other):
        """
        Returns two new SparseMatrices, containing all the labels of
        self and other.
        """
        # Align rows.
        if self.row_labels is None and other.row_labels is None:
            # Both matrices are unlabeled.
            merged_rows = None
            row_indices = xrange(other.shape[0])
            nrows = max(self.shape[0], other.shape[0])
        elif self.row_labels is None or other.row_labels is None:
            raise LabelError("I don't know how to merge labeled and unlabeled rows")
        else:
            # Merge the labels.
            merged_rows, row_indices = self.row_labels.merge(other.row_labels)
            nrows = len(merged_rows)

        # Align columns.
        if self.col_labels is None and other.col_labels is None:
            merged_cols = None
            col_indices = xrange(other.shape[1])
            ncols = max(self.shape[1], other.shape[1])
        elif self.col_labels is None or other.col_labels is None:
            raise LabelError("I don't know how to merge labeled and unlabeled columns")
        else:
            merged_cols, col_indices = self.col_labels.merge(other.col_labels)
            ncols = len(merged_cols)

        # Make aligned sparse matrices.
        mat1 = PysparseMatrix(nrow=nrows, ncol=ncols)
        mat1[:self.shape[0], :self.shape[1]] = self.psmatrix
        mat2 = PysparseMatrix(nrow=nrows, ncol=ncols)
        for mat2row, newrow in enumerate(row_indices):
            mat2[newrow, col_indices] = other.psmatrix[mat2row, :]
        return (SparseMatrix(mat1, merged_rows, merged_cols),
                SparseMatrix(mat2, merged_rows, merged_cols))
    
    ### vectorwise operations

    def row_scale(self, v):
        """
        Scale the rows of this matrix in place, multiplying each row by a
        value given in the vector *v*.
        """
        if isinstance(v, SparseVector):
            v = v.to_dense()
        if isinstance(v, DenseVector):
            if (v.labels != self.row_labels):
                raise LabelError("Row labels do not match")
            if len(v) != self.shape[0]:
                raise LabelError("Number of rows does not match")
            innervec = v
        else:
            innervec = v
        self.psmatrix.row_scale(innervec)
                
    def col_scale(self, v):
        """
        Scale the columns of this matrix in place, multiplying each column by a
        value given in the vector *v*.
        """
        if isinstance(v, SparseVector):
            v = v.to_dense()
        if isinstance(v, DenseVector):
            if (v.labels != self.col_labels):
                raise LabelError("Column labels do not match")
            if len(v) != self.shape[1]:
                raise LabelError("Number of columns does not match")
            innervec = v
        else:
            innervec = v
        self.psmatrix.col_scale(innervec)
    
    def row_op(self, op):
        """
        Perform a NumPy operation on every row, and return a
        DenseVector of the results.

        The operation will receive a NumPy vector of *only* the non-zero values.
        """
        vec = np.zeros((self.shape[0],))
        #for row in xrange(self.shape[0]):
        #    sparserow = self.psmatrix[row,:]
        #    vec[row] = op(sparserow.find()[0])
        values, rows, cols = self.find()
        startptr = 0
        endptr = 0
        N = len(values)
        while endptr < N:
            while endptr < N and rows[endptr] == rows[startptr]:
                endptr += 1
            vec[rows[startptr]] = op(values[startptr:endptr])
            startptr = endptr
        return DenseVector(vec, self.row_labels)

    def col_op(self, op):
        """
        Perform a NumPy operation on every column, and return a
        DenseVector of the results.
        
        The operation will receive a NumPy vector of *only* the non-zero values.
        """
        return self.T.row_op(op)
    
    def check_zero_rows(self):
        epsilon = 1e-16
        min_magnitude = np.min(self.row_op(np.linalg.norm))
        if min_magnitude < epsilon:
            min_index = np.argmin(self.row_op(np.linalg.norm))
            raise ValueError("Row %d of this matrix is all zeros. Use .squish() first." % min_index)
    
    # TODO: add offset like DenseMatrix has
    def normalize_rows(self):
        """
        Rescale the rows of this matrix so that they all have unit Euclidean
        magnitude. This can be used as the first step of a scaled PCA.
        """
        self.check_zero_rows()
        newmat = self.copy()
        newmat.row_scale(self.row_op(_inv_norm))
        return newmat

    def normalize_cols(self):
        """
        Rescale the columns of this matrix so that they all have unit Euclidean
        magnitude. This can be used as the first step of a scaled PCA.
        """
        newmat = self.copy()
        newmat.col_scale(self.col_op(_inv_norm))
        return newmat

    def normalize_all(self):
        """
        Rescale the rows and columns of this matrix, by dividing both the rows
        and the columns by the square root of their Euclidean norm. Thisc
        This can be used as the first step of a scaled PCA.
        """
        newmat = self.copy()
        newmat.row_scale(self.row_op(_inv_root_norm))
        newmat.col_scale(self.col_op(_inv_root_norm))
        return newmat

    def normalize_tfidf(self, cols_are_terms=False):
        '''
        Return a copy of the this matrix, normalized using tfidf.

        By default, treats the matrix as terms-by-documents; pass
        ``cols_are_terms=True`` if the matrix is instead
        documents-by-terms.

        Details:
        Normalizes by document length (the "term frequency"/tf part)
        and by term importance, measured by what fraction of the documents
        the term appears in (the "inverse document frequency"/idf part).
        
        In the resulting matrix, the entry ``(term, document)`` is given
        by: ``tf(term, document) * idf(term)``, where ``tf(term, document)
        = occurrances(term, document) / occurrances(*, document)`` and
        ``idf(term) = log(num_documents / num_docs_that_contain_term(term)``.
        '''
        from itertools import izip
        from math import log

        values, rows, cols = self.find()
        num_terms, num_documents = self.shape
        if cols_are_terms:
            rows, cols = cols, rows
            num_terms, num_documents = num_documents, num_terms

        # Compute aggregate counts
        counts_for_document = np.zeros((num_documents,))
        num_docs_that_contain_term = np.zeros((num_terms,), dtype=np.uint32)
        for term, document, value in izip(rows, cols, values):
            counts_for_document[document] += abs(value)
            num_docs_that_contain_term[term] += 1        

        normalized_values = [term_count / counts_for_document[document] # tf
                             * log(num_documents / num_docs_that_contain_term[term]) # idf
                             for term_count, term, document in izip(values, rows, cols)]

        if cols_are_terms: rows, cols = cols, rows
        result = SparseMatrix.from_lists(normalized_values,
                                         rows, cols,
                                         nrows=self.shape[0], ncols=self.shape[1])
        result.row_labels = copy(self.row_labels)
        result.col_labels = copy(self.col_labels)
        return result
    
    def row_mean_center(self):
        row_means = self.row_op(np.mean)

        shifted = self.copy()
        for row, col in shifted.keys():
            shifted[row, col] -= row_means[row]
        return (shifted, row_means)
    
    def col_mean_center(self):
        col_means = self.col_op(np.mean)

        shifted = self.copy()
        for row, col in shifted.keys():
            shifted[row, col] -= col_means[col]
        return (shifted, col_means)
    
    def mean_center(self):
        """
        Shift the rows and columns of the matrix so that their means are 0.

        Return the new matrix, plus the lists of row and column offsets,
        plus the global offset, that can be added to undo the shift.
        """
        total_mean = np.mean(self.values())
        row_means = self.row_op(np.mean) - total_mean
        col_means = self.col_op(np.mean) - total_mean
        row_lengths = self.row_op(len)
        col_lengths = self.col_op(len)

        shifted = self.copy()
        for row, col in shifted.keys():
            shifted[row, col] -= (
                (row_means[row]*row_lengths[row]
                 + col_means[col]*col_lengths[col]
                ) / (row_lengths[row] + col_lengths[col])
            ) + total_mean
            #shifted[row, col] -= (row_means[row] + col_means[col] + total_mean)
        return (shifted, row_means, col_means, total_mean)
    
    ### specific implementations of arithmetic operators

    def _add_sparse(self, other):
        """
        Add another SparseMatrix to this one.
        """
        assert isinstance(other, SparseMatrix)
        if self.same_labels_as(other):
            # the easy way
            newps = self.psmatrix.copy()
            newps.matrix.shift(1.0, other.llmatrix)
            return self.replacedata(newps)
        else:
            newself, newother = self.match_labels(other)
            newself += newother
            return newself

    def _iadd_sparse(self, other):
        """
        Add another SparseMatrix to this one in place.
        """
        assert isinstance(other, SparseMatrix)
        if not self.same_labels_as(other):
            raise LabelError("In-place operations require matching labels.")
        self.llmatrix.shift(1.0, other.llmatrix)
        return self
    
    def _multiply_sparse(self, other):
        """
        Elementwise multiplication by a sparse matrix.
        """
        assert isinstance(other, SparseMatrix)
        if self.same_labels_as(other):
            result = self.replacedata(PysparseMatrix(nrow=self.shape[0], ncol=self.shape[1]))
            for key, value in other.items():
                result[key] = value * self[key]
            return result
        else:
            newself, newother = self.match_labels(other)
            return newself._multiply_sparse(newother)
    
    def _dot_sparse(self, other):
        """
        Matrix multiplication with another sparse matrix or vector.
        """
        if isinstance(other, SparseVector):
            return self._dot_sparse(other.to_column())[:,0]

        assert isinstance(other, SparseMatrix)
        if self.shape[1] != other.shape[0]:
            raise DimensionMismatch("Inner dimensions do not match.")
        # Inner labels must match (or be unlabeled):
        if self.col_labels != other.row_labels:
            if self.col_labels is None:
                raise LabelError("Can't multiply labeled with unlabeled data.")
            else:
                raise LabelError("Inner labels do not match.")
        newmat = self.psmatrix * other.psmatrix
        return SparseMatrix(newmat, self.row_labels, other.col_labels)
    
    def _transpose_dot_sparse(self, other):
        """
        Matrix multiplication of the transpose of this matrix with another
        sparse matrix.
        """
        if isinstance(other, SparseVector):
            return self._transpose_dot_sparse(other.to_column())[:,0]

        assert isinstance(other, SparseMatrix)
        if self.shape[0] != other.shape[0]:
            raise DimensionMismatch("Row dimensions do not match.")
        # Inner labels must match (or be unlabeled):
        if self.row_labels != other.row_labels:
            if other.row_labels is None:
                raise LabelError("Can't multiply labeled with unlabeled data.")
            else:
                raise LabelError("Row labels do not match.")
        
        # The low-level 'dot' function actually multiplies rows.
        new_inner = spmatrix.dot(self.llmatrix, other.llmatrix)
        newmat = PysparseMatrix(matrix=new_inner)
        return SparseMatrix(newmat, self.col_labels, other.col_labels)
    
    def _dot_dense(self, other):
        """
        Multiply this SparseMatrix by a dense vector or matrix.
        """
        if other.ndim == 1:
            return self._dot_dense_vector(other)
        if not isinstance(other, DenseMatrix):
            other = DenseMatrix(other)
        
        if self.shape[1] != other.shape[0]:
            raise DimensionMismatch("Inner dimensions do not match.")
        # Inner labels must match (or be unlabeled):
        if self.col_labels != other.row_labels:
            if self.col_labels is None:
                raise LabelError("Can't multiply labeled with unlabeled data.")
            else:
                raise LabelError("Inner labels do not match.")
        out = np.zeros((self.shape[0], other.shape[1]))
        for col in xrange(other.shape[1]):
            out[:,col] = self.matvec(other[:,col])
        return DenseMatrix(out, self.row_labels, other.col_labels)

    def _dot_dense_vector(self, other):
        assert other.ndim == 1
        out = self.matvec(other)
        return DenseVector(out, self.row_labels)

    ### dictionary-like operations

    def keys(self):
        """
        Returns a list of tuples, giving the indices of non-zero entries.
        """
        # llmatrix.keys() doesn't do what you expect
        return zip(*self.llmatrix.keys())
    
    def keylists(self):
        """
        Returns the indices of non-zero entries as a list of rows and a list
        of columns.
        """
        return tuple(self.llmatrix.keys())

    def value_array(self):
        """
        Get the values in this matrix, in key order, as a NumPy array.
        """
        return self.find()[0]
    
    ### convenience methods for learning

    def squish(self, cutoff=1):
        """
        Discard all rows and columns that do not have at least
        `cutoff` entries. The default, `cutoff=1`, discards empty
        rows and columns.
        """
        row_entries = self.row_op(len)
        col_entries = self.col_op(len)
        entries = self.entries()
        rows = set(rowidx for (rowidx, val) in enumerate(row_entries)
                if val >= cutoff)
        cols = set(colidx for (colidx, val) in enumerate(col_entries)
                if val >= cutoff)
        if len(rows) == self.shape[0] and len(cols) == self.shape[1]:
            return self
        newentries = [(val, self.row_label(row), self.col_label(col))
                      for (val, row, col) in entries
                      if row in rows and col in cols]
        return SparseMatrix.from_named_entries(newentries).squish()
    
    # adds compatibility with SciPy-based methods
    def to_scipy_csr(self):
        """
        Convert this matrix to a SciPy csr_matrix.
        """
        from scipy.sparse import csr_matrix
        data, row, col = self.find()
        return csr_matrix( (data,(row,col)), shape=self.shape )
    to_scipy = to_scipy_csr

    # Pickling and unpickling
    def to_state(self):
        """
        Express this matrix as a dictionary of pure-Python objects that can be
        pickled.
        """
        return {
            'version': 1,
            'lists': self.lists(),
            'row_labels': self.row_labels,
            'col_labels': self.col_labels,
            'nrows': self.shape[0],
            'ncols': self.shape[1]
        }

    @staticmethod
    def from_state(d):
        """
        Load a SparseMatrix from a pure-Python dictionary, of the form output
        by :meth:`to_state`.
        """
        assert d['version'] == 1
        mat = SparseMatrix.from_lists(d['lists'][0], d['lists'][1], d['lists'][2],
                                      nrows=d['nrows'],
                                      ncols=d['ncols'])
        mat.row_labels = d['row_labels']
        mat.col_labels = d['col_labels']
        return mat
    
    def __reduce__(self):
        return (_matrix_from_state, (self.to_state(),))

    ### methods that fall through directly to PySparse

    def _setup_wrapped_methods(self):
        """
        Get simple methods from the psmatrix, or its internal ll_mat,
        and pass them through.
        """
        self.compress = self.llmatrix.compress
        self.norm = self.llmatrix.norm
        self.values = self.llmatrix.values
        self.items = self.llmatrix.items
        self.take = self.psmatrix.take
        self.put = self.psmatrix.put
        self.find = self.psmatrix.find
        self.addAt = self.psmatrix.addAt
        self.addAtDiagonal = self.psmatrix.addAtDiagonal

        # Watch out: psmatrix.matvec and llmatrix.matvec disagree on what
        # arguments they take!
        self.matvec = self.psmatrix.matvec
    
    ### string representations

    def __repr__(self):
        return "<SparseMatrix (%d by %d)>" % (self.shape[0], self.shape[1])
    
    def __unicode__(self):
        r"""
        Write out a representative picture of this matrix.

        The upper left corner of the matrix will be shown, with up to 20x5
        entries, and the rows and columns will be labeled with up to 8
        characters.
        
        >>> print SparseMatrix((2,2))
        SparseMatrix (2 by 2)
            ---        ---
            ---        ---
        >>> print SparseMatrix((1, 8))
        SparseMatrix (1 by 8)
            ---        ---        ---        ---        ---     ...
        >>> print SparseMatrix((1, 8), row_labels=['rowname'])
        SparseMatrix (1 by 8)
        rowname     ---        ---        ---        ---        ---     ...
        >>> str(SparseMatrix((0, 0)))
        'SparseMatrix (0 by 0)\n'
        >>> str(SparseMatrix((80, 1)))[-3:]
        '...'
        """
        inner_str = str(self.psmatrix[:20, :5])
        lines = inner_str.split('\n')
        headers = [repr(self)[1:-1]]
        if self.col_labels:
            col_headers = [('%-8s' % (format_label(item),))[:8] for item in self.col_labels[:5]]
            headers.append(' '+('   '.join(col_headers)))
        if self.row_labels:
            for (i, line) in enumerate(lines):
                lines[i] = ('%-8s' % (format_label(self.row_labels[i]),))[:8] + line
            for (i, line) in enumerate(headers):
                if i > 0:
                    headers[i] = ' '*8+line
        lines = headers+lines
        if self.shape[1] > 5 and self.shape[0] > 0:
            lines[1] += ' ...'
        if self.shape[0] > 20:
            lines.append('...')
        return '\n'.join(line.rstrip() for line in lines)

class SparseVector(AbstractSparseArray, LabeledVectorMixin):
    """
    A SparseVector is a vector (a 1-D array) in which only the non-zero values
    are represented. Sparse vectors can be indexed and sliced as if they were
    dense NumPy vectors.

    The underlying representation of a SparseVector is a one-row
    PysparseMatrix.
    """
    def __init__(self, arg1, labels=None):
        if isinstance(arg1, PysparseMatrix):
            psmatrix = arg1
        elif np.isscalar(arg1):
            psmatrix = PysparseMatrix(nrow=1, ncol=arg1)
        elif isinstance(arg1, (list, tuple)):
            # First turn a sequence into a NumPy array, which we know how
            # to handle (in the next step).
            psmatrix = _ndarray_to_sparse(np.array(arg1))
        elif isinstance(arg1, np.ndarray):
            # Given a NumPy array, turn it into sparse data.
            psmatrix = _ndarray_to_sparse(arg1)
        elif isinstance(arg1, AbstractSparseArray):
            psmatrix = arg1.psmatrix
        else:
            raise TypeError("I don't know how to construct a SparseVector from a %s" % type(arg1))
        
        assert isinstance(psmatrix, PysparseMatrix)
        assert psmatrix.shape[0] == 1
        self.psmatrix = psmatrix
        self.labels = indexable_set(labels or [])
        self._setup_wrapped_methods()
    
    ### numpy-like properties

    def __len__(self):
        return self.psmatrix.shape[1]

    @property
    def shape(self):
        return (len(self),)

    @property
    def nnz(self):
        return self.psmatrix.nnz

    ndim = 1

    ### factory methods

    @staticmethod
    def from_lists(values, cols, n=None):
        """
        Create a new SparseVector from sparse data.  The data should be
        expressed as two parallel lists, containing the values and the
        corresponding entries that they go into.

        The optional argument `n` specifies how many entries the vector
        should have, so that it is possible to construct vectors with
        zeros at the end this way.
        """
        if n is None: ncols = np.max(cols)+1
        else: ncols = n
        sparse = PysparseMatrix(nrow=1, ncol=ncols)
        if isinstance(values, tuple): values = list(values)
        if isinstance(cols, tuple): cols = list(cols)
        sparse.addAt(values, [0]*len(cols), cols)

        return SparseVector(sparse)
    
    @staticmethod
    def from_entries(tuples):
        """
        Create a new SparseVector from a list of tuples. Each tuple is
        of the form (value, index), expressing a value and where it goes
        in the vector.

        The reason the value comes first is that it is consistent with the
        typical sparse representation of matrices. You may want `from_items`
        for something that looks more Pythonic.
        """
        return SparseVector.from_lists(*zip(*tuples))
    
    @staticmethod
    def from_items(items):
        """
        Create a new SparseVector from a list of tuples. Each tuple is
        of the form (index, value), expressing a value and where it goes
        in the vector.
        """
        return SparseVector.from_entries([(value, key) for (key, value) in items])

    @staticmethod
    def from_named_lists(values, keys, labels=None):
        """
        Constructs a SparseVector similarly to :meth:`from_lists`,
        but you specify the *labels* of rows and columns, not their
        indices.

        Optionally, you can provide an existing label list or
        OrderedSet to use for the labels.
        """
        if labels is None: labels = OrderedSet()
        # Ensure that the labels are indeed OrderedSets.
        labels = indexable_set(labels)
        # Ensure all labels are present.
        labels.extend(keys)
        # Look up indices.
        cols = [labels.index(name) for name in keys]
        # Construct matrix.
        result = SparseVector.from_lists(values, cols)
        result.labels = labels
        return result

    @staticmethod
    def from_named_entries(tuples):
        """
        Create a new SparseVector from a list of tuples. Each tuple is
        of the form (value, key), expressing a value and the
        label for where it goes in the vector.

        You may want from_named_items, which takes input in the more
        Pythonic (key, value) order.
        """
        if tuples:
            return SparseVector.from_named_lists(*zip(*tuples))
        else:
            return SparseVector(0)
    
    @staticmethod
    def from_named_items(items):
        """
        Create a new SparseVector from a list of tuples. Each tuple is
        of the form (key, value), expressing a value and where it goes
        in the vector.
        """
        return SparseVector.from_named_entries([(value, key) for (key, value) in items])

    @staticmethod
    def from_dict(d):
        """
        Create a new SparseVector from a dictionary. The keys will become
        the labels.
        """
        return SparseVector.from_named_items(d.items())

    @staticmethod
    def from_counts(lst):
        """
        Create a new SparseVector that counts the number of times each item
        appears in a list.
        """
        d = defaultdict(int)
        for item in lst:
            d[item] += 1
        return SparseVector.from_dict(d)

    ### basic operations

    def copy(self):
        """
        Return a copy of this vector.
        """
        # labels are automatically copied
        return SparseVector(self.psmatrix.copy(), self.labels)
    
    def replacedata(self, newsparse):
        """
        Make a sparse vector with the same labels, but different data. Useful
        for applying an operation to the underlying PysparseMatrix.
        """
        assert isinstance(newsparse, PysparseMatrix)
        return SparseVector(newsparse, self.labels)

    def transpose(self):
        """
        A true vector has no "row" or "column" directionality, so it is its
        own transpose.
        """
        return self
    
    @property
    def T(self):
        return self.transpose()

    def density(self):
        """
        Calculate how dense the vector is.

        Returns (num specified elements)/(num possible elements).
        """
        return float(self.nnz) / len(self)
    
    ### operations that export the data in another format

    def entries(self):
        """
        Get a list of (value, index) tuples that describe the content of
        this vector.

        This format can be used to construct a new vector using
        :meth:`SparseVector.from_entries`.
        """
        return zip(*self.lists())

    def named_entries(self):
        """
        Get a list of (value, label) tuples that describe the
        content of this vector.

        This format can be used to construct a new vector using
        :meth:`SparseVector.from_named_entries`.
        """
        return [(val, self.label(i))
                for (val, i) in self.entries()]

    def lists(self):
        """
        Get the content of this vector as two lists: a list of values,
        and a list of indices, so that the vector could be reconstructed using
        :meth:`SparseVector.from_lists`.
        """
        values, rows, cols = self.find()
        return (list(float(v) for v in values),
                list(int(c) for c in cols))

    def named_lists(self):
        """
        Get the content of this vector as two lists: a list of values,
        and a list of labels, so that the vector
        could be reconstructed using
        :meth:`SparseVector.from_named_lists`.
        """
        values, rows, cols = self.find()
        named_cols = [self.label(col) for col in cols]
        return values, named_cols

    def items(self):
        """
        Get the content as this vector as a list of (index, value) items.
        """
        return [(key, value) for (value, key) in self.entries()]
    
    def named_items(self):
        """
        Get the content as this vector as a list of (label, value) items.
        """
        return [(key, value) for (value, key) in self.named_entries()]

    def keys(self):
        """
        The indices of non-zero entries.
        """
        return list(self.find()[2])
    nonzero_entries = keys

    def zero_entries(self):
        """
        The indices of zero (unspecified) entries.
        """
        return list(set(xrange(len(self))) - set(self.keys()))

    def to_dict(self):
        """
        Represent this vector as a dictionary from labels to values.
        """
        d = {}
        for value, key in self.named_entries():
            d[key] = value
        return d
    
    def to_dense(self):
        """
        Convert this to a :class:`DenseVector`.
        """
        data = self.psmatrix.getNumpyArray()[0]
        return DenseVector(data, self.labels)
    
    ### indexing
    
    def __getitem__(self, indices):
        # workaround for psmatrix index glitch
        if not isinstance(indices, tuple):
            indices = (indices,)
        ps_indices = (0,)+indices
        
        if Ellipsis in indices:
            ps_list = list(ps_indices)
            index_deficit = 2 - (len(ps_list)-1)
            for index in xrange(len(ps_list)):
                if ps_list[index] is Ellipsis:
                    ps_list[index:index+1] = [SLICE_ALL] * index_deficit
                    break
            ps_indices = tuple(ps_list)
        
        if len(indices) == 0:
            indices += (SLICE_ALL,)
        labels = apply_indices(indices, self.all_labels())

        data = self.psmatrix[ps_indices]
        if len(labels) == 1:
            return SparseVector(data, labels[0])
        elif len(labels) == 2:
            return SparseMatrix(data, labels[0], labels[1])
        else:
            return data
    
    def __setitem__(self, index, targetdata):
        if isinstance(index, tuple):
            assert len(index) == 1
            index = index[0]

        if isinstance(targetdata, SparseMatrix):
            if targetdata.labels is not None:
                warnings.warn('Not yet checking that the labels make sense!')
            sparsedata = targetdata.psmatrix
        elif isinstance(targetdata, SparseVector):
            if targetdata.labels is not None:
                warnings.warn('Not yet checking that the labels make sense!')
            sparsedata = targetdata.psmatrix
        elif isinstance(targetdata, np.ndarray):
            sparsedata = _ndarray_to_sparse(targetdata)
        elif isinstance(targetdata, PysparseMatrix):
            sparsedata = targetdata
        elif isinstance(targetdata, (list, tuple)):
            nd = np.array(targetdata)
            sparsedata = _ndarray_to_sparse(nd)
        elif isinstance(targetdata, (int, long, float)):
            sparsedata = targetdata
        else:
            raise TypeError("Don't know how to assign from %s" % type(targetdata))
        self.psmatrix[0, index] = sparsedata
    
    def match_labels(self, other):
        """
        Returns two new SparseVectors, containing all the labels of
        self and other.
        """
        if self.labels is None and other.labels is None:
            merged = None
            indices = xrange(other.shape[0])
            nentries = max(self.shape[0], other.shape[0])
        elif self.labels is None or other.labels is None:
            raise LabelError("I don't know how to merge labeled and unlabeled indices")
        else:
            merged, indices = self.labels.merge(other.labels)
            nentries = len(merged)
        
        # Make aligned sparse matrices.
        mat1 = PysparseMatrix(nrow=1, ncol=nentries)
        mat1[0, :self.shape[0]] = self.psmatrix
        mat2 = PysparseMatrix(nrow=1, ncol=nentries)
        mat2[0, indices] = other.psmatrix
        return (SparseVector(mat1, merged),
                SparseVector(mat2, merged))
    
    def to_row(self):
        """
        Represent this vector as a one-row SparseMatrix.
        """
        return SparseMatrix(self.psmatrix, None, self.labels)

    def to_column(self):
        """
        Represent this vector as a one-column SparseMatrix.
        """
        return self.to_row().T
    to_col = to_column

    ### vector operations

    def vec_op(self, op):
        """
        Perform a NumPy operation on the non-zero values of the vector.
        """
        return op(self.value_array())

    def normalize(self):
        return self.cmul(self.vec_op(_inv_norm))
    hat = normalize
    
    ### specific implementations of arithmetic operators

    def _add_sparse(self, other):
        """
        Add another SparseVector to this one.

        In the interest of avoiding black magic, this does not coerce
        other types of objects.
        """
        assert isinstance(other, SparseVector)
        if self.same_labels_as(other):
            # the easy way
            newps = self.psmatrix.copy()
            newps.matrix.shift(1.0, other.psmatrix.matrix)
            return self.replacedata(newps)
        else:
            newself, newother = self.match_labels(other)
            newself += newother
            return newself

    def _iadd_sparse(self, other):
        assert isinstance(other, SparseVector)
        assert self.same_labels_as(other)
        self.psmatrix.matrix.shift(1.0, other.psmatrix.matrix)
        return self
    
    def _multiply_sparse(self, other):
        """
        Elementwise multiplication by a sparse vector.
        """
        assert isinstance(other, SparseVector)
        if self.same_labels_as(other):
            result = self.replacedata(PysparseMatrix(nrow=1, ncol=self.shape[0]))
            for key, value in other.items():
                result[key] = value * self[key]
            return result
        else:
            newself, newother = self.match_labels(other)
            return newself._multiply_sparse(newother)
    
    def _dot_sparse(self, other):
        """
        Matrix multiplication with another sparse vector.

        >>> vec1 = SparseVector([1, 2, 0, -1])
        >>> vec2 = SparseVector([2, 0, 1, 3])
        >>> vec1.dot(vec2)
        -1.0
        """
        if isinstance(other, SparseMatrix):
            return self.to_row()._dot_sparse(other)[0,:]

        assert isinstance(other, SparseVector)
        if self.shape != other.shape:
            raise DimensionMismatch("Dimensions do not match.")
        # Labels must match (or be unlabeled):
        if self.labels != other.labels:
            if self.labels is None:
                raise LabelError("Can't multiply labeled with unlabeled data.")
            else:
                raise LabelError("Labels do not match.")
        return self._multiply_sparse(other).vec_op(sum)
    _transpose_dot_sparse = _dot_sparse
    
    def _dot_dense(self, other):
        from divisi2.operators import dot
        return dot(self.to_dense(), other)

    ### dictionary-like operations

    #def keys(self):
    #    """
    #    Returns a list of tuples, giving the indices of non-zero entries.
    #    """
    #    # psmatrix.matrix.keys() doesn't do what you expect
    #    return zip(*self.psmatrix.matrix.keys())
    
    def keylists(self):
        """
        Returns the indices of non-zero entries as a list of rows and a list
        of columns.
        """
        return tuple(self.psmatrix.matrix.keys())

    def value_array(self):
        """
        Get the values in this matrix, in key order, as a NumPy array.
        """
        return self.find()[0]

    def to_state(self):
        return {
            'version': 1,
            'lists': self.lists(),
            'labels': self.labels,
            'nentries': len(self),
        }

    @staticmethod
    def from_state(d):
        assert d['version'] == 1
        mat = SparseVector.from_lists(d['lists'][0], d['lists'][1],
                                      n=d['nentries'])
        mat.labels = d['labels']
        return mat
    
    def __reduce__(self):
        return (_vector_from_state, (self.to_state(),))

    ### methods that fall through directly to PySparse

    def _setup_wrapped_methods(self):
        """
        Get simple methods from the psmatrix, or its internal ll_mat,
        and pass them through.
        
        This includes fewer methods than SparseMatrix: some of the
        methods would be weirdified by the different indices, but there's
        probably a simpler way to do them anyway.
        """
        self.compress = self.psmatrix.matrix.compress
        self.norm = self.psmatrix.matrix.norm
        self.values = self.psmatrix.matrix.values
        self.find = self.psmatrix.find
        self.matvec = self.psmatrix.matvec
    
    ### string representations

    def __repr__(self):
        # TODO: show something about labels
        return "<SparseVector (%d of %d entries)>" % (self.nnz, len(self))

    def __unicode__(self):
        pairs = ["%s=%0.6g" % (key, value) for key, value in self.named_items()]
        if len(pairs) > 20: pairs[20:] = ['...']
        therepr = "%s: [%s]" % (repr(self)[1:-1], ', '.join(pairs))
        return therepr

# Put the factory methods in a form __reduce__ likes
def _matrix_from_state(state):
    return SparseMatrix.from_state(state)
_matrix_from_state.__safe_for_unpickling__ = True

# backward compatibility with a pickle file
def _matrix_from_named_lists(*lists):
    return SparseMatrix.from_named_lists(*lists)
_matrix_from_named_lists.__safe_for_unpickling__ = True

def _vector_from_state(state):
    return SparseVector.from_state(state)
_vector_from_state.__safe_for_unpickling__ = True

def category(*entries, **dict):
    """
    A helper for making ad-hoc categories. Positional arguments turn into
    entries in the category with value 1. Keyword arguments turn into
    entries in the category with the given values.
    """
    vec1 = SparseVector.from_counts(entries)
    vec2 = SparseVector.from_dict(dict)
    return vec1 + vec2
