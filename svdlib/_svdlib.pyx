import warnings
import numpy as np
cimport cython
cimport numpy as np

# The type of doubles in numpy
DTYPE=np.float64
ctypedef np.float64_t DTYPE_t

cdef extern from "svdlib.h":
    ###
    ### Structures
    ###
   
    # Harwell-Boeing sparse matrix.
    cdef struct smat:
        long rows
        long cols
        long vals      # /* Total non-zero entries. */
        long *pointr   # /* For each col (plus 1), index of first non-zero entry. */
        long *rowind   # /* For each nz entry, the row index. */
        double *value  # /* For each nz entry, the value. */
        double *offset_for_row
        double *offset_for_col
    
    # Row-major dense matrix.  Rows are consecutive vectors.
    cdef struct dmat:
        long rows
        long cols
        double **value # /* Accessed by [row][col]. Free value[0] and value to free.*/

    cdef struct svdrec:
        int d       #  /* Dimensionality (rank) */
        dmat *Ut    #  /* Transpose of left singular vectors. (d by m)
                    #     The vectors are the rows of Ut. */
        double *S   #  /* Array of singular values. (length d) */
        dmat *Vt    #  /* Transpose of right singular vectors. (d by n)
                    #     The vectors are the rows of Vt. */
    
    #/* Creates an empty sparse matrix. */
    cdef extern smat *svdNewSMat(int rows, int cols, int vals)
    
    #/* Frees a sparse matrix. */
    cdef extern void svdFreeSMat(smat *S)
    
    #/* Creates an empty SVD record. */
    cdef extern svdrec *svdNewSVDRec()
    
    #/* Frees an svd rec and all its contents. */
    cdef extern void svdFreeSVDRec(svdrec *R)
    
    #/* Performs the las2 SVD algorithm and returns the resulting Ut, S, and Vt. */
    #cdef extern svdrec *svdLAS2(smat *A, long dimensions, long iterations, double end[2], double kappa)
    
    #/* Chooses default parameter values.  Set dimensions to 0 for all dimensions: */
    cdef extern svdrec *svdLAS2A(smat *A, long dimensions)
    
    #cdef extern void freeVector(double *v)
    #cdef extern double *mulDMatSlice(DMat *D1, DMat *D2, int index, double *weight)
    #cdef extern double *dMatNorms(DMat *D)

cdef extern from "svdwrapper.h":
    cdef extern object wrapDMat(dmat *d)
    cdef extern object wrap_double_array(double* d, int len)
    cdef extern object wrapSVDrec(svdrec *rec, int transposed)

cdef extern from "math.h":
    cdef extern double sqrt(double n)

cdef extern from "svdutil.h":
    cdef extern double *svd_doubleArray(long size, char empty, char *name)

# Understand Pysparse's ll_mat format
cdef extern from "ll_mat.h":
    ctypedef struct LLMatObject:
        int *dim         # array dimension
        int issym          # non-zero, if obj represents a symmetric matrix
        int nnz            # number of stored items
        int nalloc         # allocated size of value and index arrays
        int free           # index to first element in free chain
        double *val        # pointer to array of values
        int *col           # pointer to array of indices
        int *link          # pointer to array of indices
        int *root          # pointer to array of indices

# val -> value
# col -> rowind
# ind -> pointr

cdef smat *llmat_to_smat(LLMatObject *llmat):
    """
    Transform a Pysparse ll_mat object into an svdlib SMat by packing 
    its rows into the compressed sparse columns. This has the effect of
    transposing the matrix at the same time.
    """
    cdef smat *output 
    cdef int i, j, k, r

    r = 0
    output = svdNewSMat(llmat.dim[1], llmat.dim[0], llmat.nnz)
    output.pointr[0] = 0
    for i from 0 <= i < llmat.dim[0]:
        k = llmat.root[i]
        while (k != -1):
            output.value[r] = llmat.val[k]
            output.rowind[r] = llmat.col[k]
            r += 1
            k = llmat.link[k]
        output.pointr[i+1] = r
    return output;

def svd_llmat(llmat, int k):
    cdef smat *packed
    cdef svdrec *svdrec
    llmat.compress()
    packed = llmat_to_smat(<LLMatObject *> llmat)
    svdrec = svdLAS2A(packed, k)
    svdFreeSMat(packed)
    return wrapSVDrec(svdrec, 1)

cdef class CSCMatrix:
    cdef object tensor #holds our Python tensor
    cdef object offset_for_row, offset_for_col
    cdef smat *cmatrix  #our C matrix to send to svdlib
    cdef int transposed #variable to pass to wrapper letting it know if we transposed for speed
    cdef long rows, cols 
    cdef long nonZero    #number of non zero entries in matrix
    def __init__(self, pyTensor, offset_for_row=None, offset_for_col=None):
        #init function takes 1 or 2 arguments, the Python tensor, and an array representing
        #the row_factors to divide by 
        self.tensor = pyTensor
        self.transposed = 0

        if self.tensor.ndim != 2:
            raise ValueError("You can only pack a 2 dimensional tensor")
        self.rows, self.cols = self.tensor.shape
        self.nonZero = len(self.tensor)
        assert self.nonZero != 0

        self.offset_for_row = offset_for_row
        self.offset_for_col = offset_for_col

    cpdef pack(self, row_norms=None):
        cdef np.ndarray[DTYPE_t, ndim=1] row_factors # holds row _multiplication_ factors, possibly just 1.
        cdef long row, col, index, cols, col_len, rowk
        cdef double value
        if row_norms is None:
            row_factors = np.ones(self.rows)
        else:
            row_factors = np.reciprocal(np.sqrt(np.array(row_norms, dtype=DTYPE)))

        self.cmatrix = svdNewSMat(self.rows, self.cols, self.nonZero)
        self.transposed = 0
        columnDict = {}
        cols = self.cols
        for (row, col), value in self.tensor.iteritems():
            columnDict.setdefault(col, {})[row] = value * row_factors[row]
        assert len(columnDict) <= cols
        index = 0
        for col in xrange(cols):
            col_len = len(columnDict.get(col, []))
            self.setColumnSize(col, col_len)
            if col_len == 0: continue
            for rowk in sorted(columnDict[col].keys()):
                self.setValue(index, rowk, columnDict[col][rowk])
                index += 1
        if self.offset_for_row is not None:
            self.cmatrix.offset_for_row = self.toVector(self.offset_for_row)
        if self.offset_for_col is not None:
            self.cmatrix.offset_for_col = self.toVector(self.offset_for_col)
                
    cdef double *toVector(self, vec):
         cdef long n = len(vec)
         cdef double *temp = svd_doubleArray(n, False, "toVector")
         for i in range(n):
             temp[i] = vec[i]
         return temp

    cpdef dictPack(self, row_norms=None):
        cdef np.ndarray[DTYPE_t, ndim=1] row_factors # holds row _multiplication_ factors, possibly just 1.
        if row_norms is None:
            row_factors = np.ones(self.rows)
        else:
            row_factors = np.reciprocal(np.sqrt(np.array(row_norms, dtype=DTYPE)))

        self.cmatrix = svdNewSMat(self.cols, self.rows, self.nonZero)
        self.transposed = 1
        cdef object rowDict = self.tensor._data
        cdef int rows = self.tensor.shape[0]
        cdef int colk
        cdef int index = 0
        cdef int rownum
        cdef int row_len
        cdef double val
        assert len(rowDict) <= rows 
        for rownum from 0 <= rownum < rows:
            row_len = len(rowDict.get(rownum, []))
            self.setColumnSize(rownum, row_len)
            if row_len == 0: continue
            for colk in sorted(rowDict[rownum].keys()):
                val = float(rowDict[rownum][colk]) * row_factors[rownum]
                self.setValue(index, colk, val)
                index += 1
        # Transposed offset vectors.
        if self.offset_for_row is not None:
            self.cmatrix.offset_for_col = self.toVector(self.offset_for_row)
        if self.offset_for_col is not None:
            self.cmatrix.offset_for_row = self.toVector(self.offset_for_col)


    def __repr__(self):
        return u'<CSCMatrix>'

    def __dealloc__(self):
        svdFreeSMat(self.cmatrix)

    cdef void setColumnSize(self, int col, int size):
        if size == 0:
            warnings.warn('Column %d is empty' % col)
        self.cmatrix[0].pointr[0] = 0
        self.cmatrix.pointr[col+1] = self.cmatrix.pointr[col] + size

    cdef void setValue(self, int index, int rowind, double value):
        if value == 0:
               warnings.warn('Matrix has zero value (row %d, index %d)' % (rowind, index))
        self.cmatrix.rowind[index] = rowind
        self.cmatrix.value[index] = value
    
    #cdef svd(self, iterations, end1, end2, kappa, k=50):
    #    svdrec = svdLAS2(self.cmatrix, k, iterations=iterations, end1=end1, end2=end2, kappa=kappa)
    #    ut, s, vt = wrapSVDrec(svdrec)
    #    return SVD2DResults(DenseTensor(ut.T), DenseTensor(vt.T), DenseTensor(s))


    cpdef object svdA(self, int k):
        cdef svdrec *svdrec
        svdrec = svdLAS2A(self.cmatrix, k)
        return wrapSVDrec(svdrec, self.transposed)
                
    @cython.boundscheck(False) 
    def isvd(self, int k=50, int niter=100, double lrate=.001):
        from csc.divisi.tensor import DenseTensor
        cdef smat* A = self.cmatrix
        print "COMPUTING INCREMENTAL SVD"
        print "ROWS: %d, COLUMNS: %d, VALS: %d" % (A.rows, A.cols, A.vals)
        print "K: %d, LEARNING_RATE: %r, ITERATIONS: %d" % (k, lrate, niter)

        cdef np.ndarray[DTYPE_t, ndim=2] u = np.add(np.zeros((A.rows, k), dtype=DTYPE), .001)
        cdef np.ndarray[DTYPE_t, ndim=2] v = np.add(np.zeros((A.cols, k), dtype=DTYPE), .001)

        # Maintain a cache of dot-products up to the current axis
        cdef smat* predicted = svdNewSMat(A.rows, A.cols, A.vals)

        # Type all loop vars
        cdef unsigned int axis, i, cur_row,cur_col, col_index, next_col_index, value_index
        cdef double err, u_value

        # Initialize dot-product cache
        # (This should be done with memcpy, but i'm not certain
        # how to do that here)
        for i in range(A.cols + 1):
            predicted.pointr[i] = A.pointr[i]
        
        for i in range(A.vals):
            predicted.rowind[i] = A.rowind[i]
            predicted.value[i] = 0

        for axis in range(k):
            for i in range(niter):
                # Iterate over all values of the sparse matrix
                for cur_col in range(A.cols):
                    col_index = A.pointr[cur_col]
                    next_col_index = A.pointr[cur_col + 1]
                    for value_index in range(col_index, next_col_index):
                        cur_row = A.rowind[value_index]
                        err = A.value[value_index] - (predicted.value[value_index] + 
                                                      u[cur_row, axis] * v[cur_col, axis])

                        u_value = u[cur_row, axis]
                        u[cur_row, axis] += lrate * err * v[cur_col, axis]
                        v[cur_col, axis] += lrate * err * u_value
    
            # Update cached dot-products
            for cur_col in range(predicted.cols):
                col_index = predicted.pointr[cur_col]
                next_col_index = predicted.pointr[cur_col + 1]
                for value_index in range(col_index, next_col_index):
                    cur_row = predicted.rowind[value_index]
                    predicted.value[value_index] += u[cur_row, axis] * v[cur_col, axis]

        # Factor out the svals from u and v
        u_sigma = np.sqrt(np.add.reduce(np.multiply(u, u)))
        v_sigma = np.sqrt(np.add.reduce(np.multiply(v, v)))
    
        u_tensor = DenseTensor(np.divide(u, u_sigma))
        v_tensor = DenseTensor(np.divide(v, v_sigma))
        sigma = DenseTensor(np.multiply(u_sigma, v_sigma))

        svdFreeSMat(predicted)

        if self.transposed:
            return v_tensor, u_tensor, sigma
        else:
            return u_tensor, v_tensor, sigma

def svd(tensor, k=50, row_factors=None, offset_for_row=None, offset_for_col=None):
    CSC = CSCMatrix(tensor, offset_for_row, offset_for_col)
    CSC.pack(row_factors)
    return CSC.svdA(k)

def dictSvd(tensor, k=50, row_factors=None, offset_for_row=None, offset_for_col=None):
    CSC = CSCMatrix(tensor, offset_for_row, offset_for_col)
    CSC.dictPack(row_factors)
    return CSC.svdA(k)

def isvd(tensor, k=50, niter=100, lrate=.001, row_factors=None):
    CSC = CSCMatrix(tensor)
    CSC.pack(row_factors)
    return CSC.isvd(k, niter, lrate)

def dictIsvd(tensor, k=50, niter=100, lrate=.001, row_factors=None):
    CSC = CSCMatrix(tensor)
    CSC.dictPack(row_factors)
    return CSC.isvd(k, niter, lrate)
