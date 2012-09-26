import warnings
import numpy as np
cimport cython
cimport numpy as np

# The type of doubles in numpy
DTYPE=np.float64
ctypedef np.float64_t DTYPE_t

np.import_array()

cdef extern from "svdlib.h":
    ###
    ### Structures
    ###
   
    # Abstract matrix class
    cdef struct matrix:
        # we don't have to tell Cython about the ops pointers.
        long rows
        long cols
        long vals # Total specified entries.
        matrix* (*transposed)(matrix *A)
        void (*mat_by_vec)(matrix *A, double *vec, double *out)
        void (*mat_transposed_by_vec)(matrix *A, double *vec, double *out)

    # Harwell-Boeing sparse matrix.
    cdef struct smat:
        matrix h
        long *pointr   # /* For each col (plus 1), index of first non-zero entry. */
        long *rowind   # /* For each nz entry, the row index. */
        double *value  # /* For each nz entry, the value. */
        double *offset_for_row
        double *offset_for_col
    
    # Row-major dense matrix.  Rows are consecutive vectors.
    cdef struct dmat:
        matrix h
        double **value # /* Accessed by [row][col]. Free value[0] and value to free.*/

    cdef struct summing_mat:
        matrix h
        # Cython doesn't need to know anything beyond that.

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

    # Creates an empty dense matrix. */
    cdef extern dmat *svdNewDMat(int rows, int cols)
    # Frees a dense matrix.
    cdef extern void svdFreeDMat(dmat *D)

    cdef extern summing_mat *summing_mat_new(int n)
    cdef extern void summing_mat_set(summing_mat *m, int i, matrix *m)
    cdef extern void summing_mat_free(summing_mat *m)

    
    #/* Creates an empty SVD record. */
    cdef extern svdrec *svdNewSVDRec()
    
    #/* Frees an svd rec and all its contents. */
    cdef extern void svdFreeSVDRec(svdrec *R)
    
    #/* Performs the las2 SVD algorithm and returns the resulting Ut, S, and Vt. */
    #cdef extern svdrec *svdLAS2(smat *A, long dimensions, long iterations, double end[2], double kappa)
    
    #/* Chooses default parameter values.  Set dimensions to 0 for all dimensions: */
    cdef extern svdrec *svdLAS2A(matrix *A, long dimensions)
    
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

cdef struct py_mat:
    matrix h
    void *py_object

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
    return output

cdef smat *llmat_to_smat_shifted(LLMatObject *llmat, row_mapping_, col_mapping_):
    """
    Transform a Pysparse ll_mat object into an svdlib SMat by packing 
    its rows into the compressed sparse columns. This has the effect of
    transposing the matrix at the same time.

    Also, set a shift on each row and each column of the SMat, to allow mean
    centering.
    """
    cdef smat *output 
    cdef np.ndarray[double, ndim=1] row_mapping = row_mapping_
    cdef np.ndarray[double, ndim=1] col_mapping = col_mapping_
    cdef double *row_array = <double *> row_mapping.data
    cdef double *col_array = <double *> col_mapping.data

    output = llmat_to_smat(llmat)
    output.offset_for_row = col_array    # remember, it's transposed
    output.offset_for_col = row_array
    return output

cdef smat *llmat_to_smat_remapped(LLMatObject *llmat, row_mapping_, col_mapping_, double weight) except? NULL:
    """
    Transform a Pysparse ll_mat object into an svdlib SMat by packing 
    its rows into the compressed sparse columns. This has the effect of
    transposing the matrix at the same time.

    row_mapping and col_mapping specify the output indices for each
    input row and column (before transposing).
    
    Implementation notes: we need to build the output in order of
    columns, so we iterate through source rows in sorted order, given
    by np.argsort(row_mapping). We then need to build each output row
    in sorted order, which requires a bit more care.
    """
    cdef smat *output
    cdef int prev_out_column, cur_input_row, cur_out_column, i, k, col_len, output_index, start_of_column, row_index
    cdef np.ndarray[unsigned long, ndim=1] row_mapping = row_mapping_
    cdef np.ndarray[unsigned long, ndim=1] col_mapping = col_mapping_
    cdef np.ndarray[long, ndim=1] row_order, col_order, column_indices

    # Create the (transposed) output matrix.
    output = svdNewSMat(np.max(col_mapping)+1, np.max(row_mapping)+1, llmat.nnz)
    output.pointr[0] = 0

    # Iterate through rows in the order they appear as output columns.
    # Note importantly that this may yield empty output columns.
    row_order = np.argsort(row_mapping)
    prev_out_column = 0
    for row_index from 0 <= row_index < len(row_order):
        cur_input_row = row_order[row_index]
        cur_out_column = row_mapping[cur_input_row]

        # Find where the column starts
        start_of_column = output.pointr[prev_out_column + 1]
        # Fill in start pointers for skipped columns.
        for i from prev_out_column + 1 <= i <= cur_out_column:
            output.pointr[i] = start_of_column
        prev_out_column = cur_out_column

        # Iterate through columns in the source row, placing them in
        # the proper places in the output column.

        # First, determine the length of the column
        col_len = 0
        k = llmat.root[cur_input_row]
        while k != -1: # signifies end of the source row
            col_len += 1
            k = llmat.link[k]

        # Now get the column of each entry as an array.
        column_indices = np.zeros(col_len, dtype=np.int64)
        i = 0
        k = llmat.root[cur_input_row]
        while k != -1:
            column_indices[i] = llmat.col[k]
            i += 1
            k = llmat.link[k]

        # Find the index each column goes in.
        col_order = np.argsort(column_indices)

        # Put each value in the appropriate column.
        i = 0
        k = llmat.root[cur_input_row]
        while k != -1:
            output_index = start_of_column + col_order[i]
            output.value[output_index] = llmat.val[k] * weight
            output.rowind[output_index] = col_mapping[llmat.col[k]]
            i += 1
            k = llmat.link[k]
        
        output.pointr[cur_out_column+1] = output.pointr[cur_out_column] + col_len
    return output


def svd_llmat(llmat, int k):
    cdef smat *packed
    cdef svdrec *svdrec
    llmat.compress()
    packed = llmat_to_smat(<LLMatObject *> llmat)
    svdrec = svdLAS2A(<matrix *>packed, k)
    svdFreeSMat(packed)
    ut, svals, vt = wrapSVDrec(svdrec, 1)
    
    # in cases where there aren't enough nonzero singular values, make sure
    # to output the zeros too
    s = np.zeros(ut.shape[0])
    s[:len(svals)] = svals
    return ut, s, vt

def svd_llmat_shifted(llmat, int k, row_shift, col_shift):
    cdef smat *packed
    cdef svdrec *svdrec
    llmat.compress()
    packed = llmat_to_smat_shifted(<LLMatObject *> llmat, row_shift, col_shift)
    svdrec = svdLAS2A(<matrix *>packed, k)
    svdFreeSMat(packed)
    return wrapSVDrec(svdrec, 1)

@cython.boundscheck(False)
cdef void ndarray_mat_by_vec(matrix *mat, double *vec, double *out):
    cdef py_mat *pymat = <py_mat*>mat
    cdef np.ndarray[DTYPE_t, ndim=2] ndarr = <object> pymat.py_object
    for row in range(mat.rows):
        out[row] = 0.0
        for col in range(mat.cols):
            out[row] += ndarr[row, col] * vec[col]

@cython.boundscheck(False)
cdef void ndarray_mat_transposed_by_vec(matrix *mat, double *vec, double *out):
    cdef py_mat *pymat = <py_mat*>mat
    cdef np.ndarray[DTYPE_t, ndim=2] ndarr = <object> pymat.py_object
    for col in range(mat.cols):
        out[col] = 0.0
    for row in range(mat.rows):
        for col in range(mat.cols):
            out[col] += ndarr[row, col] * vec[row]

def svd_ndarray(np.ndarray[DTYPE_t, ndim=2] arr, int k):
    cdef py_mat pmat
    cdef svdrec *svdrec
    pmat.h.rows = arr.shape[0]
    pmat.h.cols = arr.shape[1]
    pmat.h.vals = pmat.h.rows * pmat.h.cols
    pmat.h.transposed = NULL
    pmat.h.mat_by_vec = ndarray_mat_by_vec
    pmat.h.mat_transposed_by_vec = ndarray_mat_transposed_by_vec
    pmat.py_object = <void*> arr
    svdrec = svdLAS2A(<matrix*> &pmat, k)
    return wrapSVDrec(svdrec, 0)

def svd_sum(mats, int k, weights, row_mappings, col_mappings):
    cdef summing_mat *sum_mat = summing_mat_new(len(mats))
    cdef svdrec *svdrec
    cdef smat *tmp
    for i in range(len(mats)):
        # Go ahead and let it transpose.
        mat = mats[i].llmatrix
        mat.compress()
        summing_mat_set(sum_mat, i, <matrix *>llmat_to_smat_remapped(<LLMatObject *>mat, row_mappings[i], col_mappings[i], weights[i]))
    svdrec = svdLAS2A(<matrix *>sum_mat, k)
    summing_mat_free(sum_mat)
    return wrapSVDrec(svdrec, 1) # transposed.

# Incremental SVD    
@cython.boundscheck(False)
cdef isvd(smat* A, int k=50, int niter=100, double lrate=.001):
    print "COMPUTING INCREMENTAL SVD"
    print "ROWS: %d, COLUMNS: %d, VALS: %d" % (A.h.rows, A.h.cols, A.h.vals)
    print "K: %d, LEARNING_RATE: %r, ITERATIONS: %d" % (k, lrate, niter)

    cdef np.ndarray[DTYPE_t, ndim=2] u = np.add(np.zeros((A.h.rows, k), dtype=DTYPE), .001)
    cdef np.ndarray[DTYPE_t, ndim=2] v = np.add(np.zeros((A.h.cols, k), dtype=DTYPE), .001)

    # Maintain a cache of dot-products up to the current axis
    cdef smat* predicted = svdNewSMat(A.h.rows, A.h.cols, A.h.vals)

    # Type all loop vars
    cdef unsigned int axis, i, cur_row,cur_col, col_index, next_col_index, value_index
    cdef double err, u_value

    # Initialize dot-product cache
    # (This should be done with memcpy, but i'm not certain
    # how to do that here)
    for i in range(A.h.cols + 1):
        predicted.pointr[i] = A.pointr[i]
    
    for i in range(A.h.vals):
        predicted.rowind[i] = A.rowind[i]
        predicted.value[i] = 0

    for axis in range(k):
        for i in range(niter):
            # Iterate over all values of the sparse matrix
            for cur_col in range(A.h.cols):
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
        for cur_col in range(predicted.h.cols):
            col_index = predicted.pointr[cur_col]
            next_col_index = predicted.pointr[cur_col + 1]
            for value_index in range(col_index, next_col_index):
                cur_row = predicted.rowind[value_index]
                predicted.value[value_index] += u[cur_row, axis] * v[cur_col, axis]

    # Factor out the svals from u and v
    u_sigma = np.sqrt(np.add.reduce(np.multiply(u, u)))
    v_sigma = np.sqrt(np.add.reduce(np.multiply(v, v)))

    np.divide(u, u_sigma, u)
    np.divide(v, v_sigma, v)
    sigma = np.multiply(u_sigma, v_sigma)

    svdFreeSMat(predicted)

    return u, v, sigma

def isvd_llmat(llmat, int k, int niter=100, double lrate=.001):
    cdef smat *packed
    cdef svdrec *svdrec
    llmat.compress()
    packed = llmat_to_smat(<LLMatObject *> llmat) # transposes the matrix.
    v, s, u = isvd(packed, k, niter, lrate)
    svdFreeSMat(packed)
    return u, s, v
