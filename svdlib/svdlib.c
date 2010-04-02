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
  D->rows = rows;
  D->cols = cols;

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
  S->rows = rows;
  S->cols = cols;
  S->vals = vals;
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
  DMat D = svdNewDMat(S->rows, S->cols);
  if (!D) {
    svd_error("svdConvertStoD: failed to allocate D");
    return NULL;
  }
  for (i = 0, c = 0; i < S->vals; i++) {
    while (S->pointr[c + 1] <= i) c++;
    D->value[S->rowind[i]][c] = S->value[i];
  }
  return D;
}

/* Converts a dense matrix to a sparse one (without affecting the dense one) */
SMat svdConvertDtoS(DMat D) {
  SMat S;
  int i, j, n;
  for (i = 0, n = 0; i < D->rows; i++)
    for (j = 0; j < D->cols; j++)
      if (D->value[i][j] != 0) n++;
  
  S = svdNewSMat(D->rows, D->cols, n);
  if (!S) {
    svd_error("svdConvertDtoS: failed to allocate S");
    return NULL;
  }
  for (j = 0, n = 0; j < D->cols; j++) {
    S->pointr[j] = n;
    for (i = 0; i < D->rows; i++)
      if (D->value[i][j] != 0) {
        S->rowind[n] = i;
        S->value[n] = D->value[i][j];
        n++;
      }
  }
  S->pointr[S->cols] = S->vals;
  return S;
}

/* Transposes a dense matrix. */
DMat svdTransposeD(DMat D) {
  int r, c;
  DMat N = svdNewDMat(D->cols, D->rows);
  for (r = 0; r < D->rows; r++)
    for (c = 0; c < D->cols; c++)
      N->value[c][r] = D->value[r][c];
  return N;
}

/* Efficiently transposes a sparse matrix. */
SMat svdTransposeS(SMat S) {
  int r, c, i, j;
  SMat N = svdNewSMat(S->cols, S->rows, S->vals);
  /* Count number nz in each row. */
  for (i = 0; i < S->vals; i++)
    N->pointr[S->rowind[i]]++;
  /* Fill each cell with the starting point of the previous row. */
  N->pointr[S->rows] = S->vals - N->pointr[S->rows - 1];
  for (r = S->rows - 1; r > 0; r--)
    N->pointr[r] = N->pointr[r+1] - N->pointr[r-1];
  N->pointr[0] = 0;
  /* Assign the new columns and values. */
  for (c = 0, i = 0; c < S->cols; c++) {
    for (; i < S->pointr[c+1]; i++) {
      r = S->rowind[i];
      j = N->pointr[r+1]++;
      N->rowind[j] = c;
      N->value[j] = S->value[i];
    }
  }
  /* Transpose the row and column offsets also. */
  if (S->offset_for_col)
    N->offset_for_row = copyVector(S->offset_for_col, S->cols, "svdTransposeS: offset_for_row");
  if (S->offset_for_row)
    N->offset_for_col = copyVector(S->offset_for_row, S->rows, "svdTransposeS: offset_for_row");

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
    
    result = (double *) malloc(sizeof(double) * D2->cols);
    
    for (col=0; col < D2->cols; col++) {
        result[col] = 0.0;
        for (row=0; row < D2->rows; row++) {
            result[col] += D2->value[row][col] * D1->value[row][index] * weight[row];
        }
    }
    return result;
}

double *dMatNorms(DMat D) {
    int col, row;
    double *result;
    
    result = (double *) malloc(sizeof(double) * D->cols);
    
    for (col=0; col < D->cols; col++) {
        result[col] = 0.0;
        for (row=0; row < D->rows; row++) {
            result[col] += D->value[row][col] * D->value[row][col];
        }
        result[col] = sqrt(result[col]);
    }
    return result;
}
