from cython.view cimport array as cvarray

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

cdef struct py_mat:
    matrix h
    void *py_object

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

###
### Run SVDs on Python objects
###

cdef void pmat_matvec(matrix *mat, double *vec, double *out):
    cdef py_mat *pymat = <py_mat*>mat
    out_ = np.asarray(<np.float64_t[:mat.rows]> out)
    out_.fill(0)
    (<object>pymat.py_object).mat_by_vec(
        np.asarray(<np.float64_t[:mat.cols]> vec),
        out_)

cdef void pmat_transposed_matvec(matrix *mat, double *vec, double *out):
    cdef py_mat *pymat = <py_mat*>mat
    out_ = np.asarray(<np.float64_t[:mat.cols]> out)
    out_.fill(0)
    (<object>pymat.py_object).mat_transposed_by_vec(
        np.asarray(<np.float64_t[:mat.rows]> vec),
        out_)

def svd_pyobj(obj, int rows, int cols, int num_vals, int k):
    """Run an SVD on a Python object that provides mat_by_vec and mat_transposed_by_vec methods.

    The method signatures should be:
    - mat_by_vec(vec, out)
    - mat_transposed_by_vec(vec, out)
    where both vec and out will be NumPy arrays.
    """
    cdef py_mat pmat
    cdef svdrec *svdrec
    pmat.h.rows = rows
    pmat.h.cols = cols
    pmat.h.vals = num_vals
    pmat.h.transposed = NULL
    pmat.h.mat_by_vec = pmat_matvec
    pmat.h.mat_transposed_by_vec = pmat_transposed_matvec
    pmat.py_object = <void*> obj
    svdrec = svdLAS2A(<matrix*> &pmat, k)
    ut, svals, vt = wrapSVDrec(svdrec, 0)
    
    # in cases where there aren't enough nonzero singular values, make sure
    # to output the zeros too
    s = np.zeros(ut.shape[0])
    s[:len(svals)] = svals
    return ut, s, vt


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
