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

def svd_llmat(llmat, int k):
    cdef smat *packed
    cdef svdrec *svdrec
    llmat.compress()
    packed = llmat_to_smat(<LLMatObject *> llmat)
    svdrec = svdLAS2A(packed, k)
    svdFreeSMat(packed)
    return wrapSVDrec(svdrec, 1)

def svd_llmat_shifted(llmat, int k, row_shift, col_shift):
    cdef smat *packed
    cdef svdrec *svdrec
    llmat.compress()
    packed = llmat_to_smat_shifted(<LLMatObject *> llmat, row_shift, col_shift)
    svdrec = svdLAS2A(packed, k)
    svdFreeSMat(packed)
    return wrapSVDrec(svdrec, 1)
    
