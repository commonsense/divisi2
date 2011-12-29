#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "svdlib.h"
#include "svdutil.h"

char *SVDVersion = "1.34";
long SVDVerbosity = 0;
long SVDCount[SVD_COUNTERS];

void svdResetCounters(void) {
  int i;
  for (i = 0; i < SVD_COUNTERS; i++)
    SVDCount[i] = 0;
}

/********************************* Allocation ********************************/

/* Row major order.  Rows are vectors that are consecutive in memory.  Matrix
   is initialized to empty. */
DMat svdNewDMat(int rows, int cols) {
  int i;
  DMat D = (DMat) malloc(sizeof(struct dmat));
  if (!D) {perror("svdNewDMat"); return NULL;}
  D->h.transposed = svdTransposeD;
  D->h.free = svdFreeDMat;
  D->h.mat_by_vec = dense_mat_by_vec;
  D->h.mat_transposed_by_vec = dense_mat_transposed_by_vec;

  D->h.rows = rows;
  D->h.cols = cols;
  D->h.vals = rows*cols;

  D->value = (double **) malloc(rows * sizeof(double *));
  if (!D->value) {SAFE_FREE(D); return NULL;}

  D->value[0] = (double *) calloc(rows * cols, sizeof(double));
  if (!D->value[0]) {SAFE_FREE(D->value); SAFE_FREE(D); return NULL;}

  for (i = 1; i < rows; i++) D->value[i] = D->value[i-1] + cols;
  return D;
}

void svdFreeDMat(DMat D) {
  if (!D) return;
  SAFE_FREE(D->value[0]);
  SAFE_FREE(D->value);
  free(D);
}


SMat svdNewSMat(int rows, int cols, int vals) {
  SMat S = (SMat) calloc(1, sizeof(struct smat));
  if (!S) {perror("svdNewSMat"); return NULL;}
  S->h.transposed = svdTransposeS;
  S->h.free = svdFreeSMat;
  S->h.mat_by_vec = sparse_mat_by_vec;
  S->h.mat_transposed_by_vec = sparse_mat_transposed_by_vec;
  S->h.rows = rows;
  S->h.cols = cols;
  S->h.vals = vals;
  S->pointr = svd_longArray(cols + 1, TRUE, "svdNewSMat: pointr");
  if (!S->pointr) {svdFreeSMat(S); return NULL;}
  S->rowind = svd_longArray(vals, FALSE, "svdNewSMat: rowind");
  if (!S->rowind) {svdFreeSMat(S); return NULL;}
  S->value  = svd_doubleArray(vals, FALSE, "svdNewSMat: value");
  if (!S->value)  {svdFreeSMat(S); return NULL;}
  S->offset_for_row = NULL;
  S->offset_for_col = NULL;
  return S;
}

void svdFreeSMat(SMat S) {
  if (!S) return;
  SAFE_FREE(S->pointr);
  SAFE_FREE(S->rowind);
  SAFE_FREE(S->value);
  SAFE_FREE(S->offset_for_row);
  SAFE_FREE(S->offset_for_col);
  free(S);
}

SummingMat summing_mat_new(int n) {
  SummingMat  mat = (SummingMat ) calloc(1, sizeof(struct summing_mat));
  mat->h.transposed = summing_mat_transposed;
  mat->h.free = summing_mat_free;
  mat->h.mat_by_vec = summing_mat_by_vec;
  mat->h.mat_transposed_by_vec = summing_mat_transposed_by_vec;
  mat->h.rows = mat->h.cols = mat->h.vals = 0;
  mat->n = n;
  mat->mats = (Matrix*) calloc(n, sizeof(Matrix));
  return mat;
}

void summing_mat_set(SummingMat mat, int i, Matrix m) {
  mat->mats[i] = m;
  if (m->rows > mat->h.rows) mat->h.rows = m->rows;
  if (m->cols > mat->h.cols) mat->h.cols = m->cols;
  mat->h.vals += m->vals; /* vals is the number of specified entries (for sparse matrices) */
}

void summing_mat_free(SummingMat mat) {
  int i;
  for (i = 0; i<mat->n; i++) {
    Matrix m = mat->mats[i];
    m->free(m);
  }
  free(mat->mats);
  free(mat);
}

SummingMat summing_mat_transposed(SummingMat mat) {
  SummingMat t = summing_mat_new(mat->n);
  int i;
  for (i=0; i<mat->n; i++) {
    Matrix m = mat->mats[i];;
    summing_mat_set(t, i, m->transposed(m));
  }
  return t;
}

/* Creates an empty SVD record */
SVDRec svdNewSVDRec(void) {
  SVDRec R = (SVDRec) calloc(1, sizeof(struct svdrec));
  if (!R) {perror("svdNewSVDRec"); return NULL;}
  return R;
}

/* Frees an svd rec and all its contents. */
void svdFreeSVDRec(SVDRec R) {
  if (!R) return;
  if (R->Ut) svdFreeDMat(R->Ut);
  if (R->S) SAFE_FREE(R->S);
  if (R->Vt) svdFreeDMat(R->Vt);
  free(R);
}


/**************************** Conversion *************************************/

/* Converts a sparse matrix to a dense one (without affecting the former) */
DMat svdConvertStoD(SMat S) {
  int i, c;
  DMat D = svdNewDMat(S->h.rows, S->h.cols);
  if (!D) {
    svd_error("svdConvertStoD: failed to allocate D");
    return NULL;
  }
  for (i = 0, c = 0; i < S->h.vals; i++) {
    while (S->pointr[c + 1] <= i) c++;
    D->value[S->rowind[i]][c] = S->value[i];
  }
  return D;
}

/* Converts a dense matrix to a sparse one (without affecting the dense one) */
SMat svdConvertDtoS(DMat D) {
  SMat S;
  int i, j, n;
  for (i = 0, n = 0; i < D->h.rows; i++)
    for (j = 0; j < D->h.cols; j++)
      if (D->value[i][j] != 0) n++;
  
  S = svdNewSMat(D->h.rows, D->h.cols, n);
  if (!S) {
    svd_error("svdConvertDtoS: failed to allocate S");
    return NULL;
  }
  for (j = 0, n = 0; j < D->h.cols; j++) {
    S->pointr[j] = n;
    for (i = 0; i < D->h.rows; i++)
      if (D->value[i][j] != 0) {
        S->rowind[n] = i;
        S->value[n] = D->value[i][j];
        n++;
      }
  }
  S->pointr[S->h.cols] = S->h.vals;
  return S;
}

/* Transposes a dense matrix. */
DMat svdTransposeD(DMat D) {
  int r, c;
  DMat N = svdNewDMat(D->h.cols, D->h.rows);
  for (r = 0; r < D->h.rows; r++)
    for (c = 0; c < D->h.cols; c++)
      N->value[c][r] = D->value[r][c];
  return N;
}

/* Efficiently transposes a sparse matrix. */
SMat svdTransposeS(SMat S) {
  int r, c, i, j;
  SMat N = svdNewSMat(S->h.cols, S->h.rows, S->h.vals);
  /* Count number nz in each row. */
  for (i = 0; i < S->h.vals; i++)
    N->pointr[S->rowind[i]]++;
  /* Fill each cell with the starting point of the previous row. */
  N->pointr[S->h.rows] = S->h.vals - N->pointr[S->h.rows - 1];
  for (r = S->h.rows - 1; r > 0; r--)
    N->pointr[r] = N->pointr[r+1] - N->pointr[r-1];
  N->pointr[0] = 0;
  /* Assign the new columns and values. */
  for (c = 0, i = 0; c < S->h.cols; c++) {
    for (; i < S->pointr[c+1]; i++) {
      r = S->rowind[i];
      j = N->pointr[r+1]++;
      N->rowind[j] = c;
      N->value[j] = S->value[i];
    }
  }
  /* Transpose the row and column offsets also. */
  if (S->offset_for_col)
    N->offset_for_row = copyVector(S->offset_for_col, S->h.cols, "svdTransposeS: offset_for_row");
  if (S->offset_for_row)
    N->offset_for_col = copyVector(S->offset_for_row, S->h.rows, "svdTransposeS: offset_for_col");

  return N;
}

double *copyVector(double* vec, int n, const char* name) {
  int i;
  double* result = svd_doubleArray(n, FALSE, name);
  for (i = 0; i < n; i++)
    result[i] = vec[i];
  return result;
}

/* this stuff was added by Rob */
void freeVector(double *v) {
    SAFE_FREE(v);
}

double *mulDMatSlice(DMat D1, DMat D2, int index, double *weight) {
    int col, row;
    double *result;
    
    result = (double *) malloc(sizeof(double) * D2->h.cols);
    
    for (col=0; col < D2->h.cols; col++) {
        result[col] = 0.0;
        for (row=0; row < D2->h.rows; row++) {
            result[col] += D2->value[row][col] * D1->value[row][index] * weight[row];
        }
    }
    return result;
}

double *dMatNorms(DMat D) {
    int col, row;
    double *result;
    
    result = (double *) malloc(sizeof(double) * D->h.cols);
    
    for (col=0; col < D->h.cols; col++) {
        result[col] = 0.0;
        for (row=0; row < D->h.rows; row++) {
            result[col] += D->value[row][col] * D->value[row][col];
        }
        result[col] = sqrt(result[col]);
    }
    return result;
}
