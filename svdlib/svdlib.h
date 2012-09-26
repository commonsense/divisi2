#ifndef SVDLIB_H
#define SVDLIB_H

#ifndef FALSE
#  define FALSE 0
#endif
#ifndef TRUE
#  define TRUE  1
#endif

/******************************** Structures *********************************/
typedef struct smat *SMat;
typedef struct dmat *DMat;
typedef struct matrix *Matrix;
typedef struct summing_mat *SummingMat;
typedef struct svdrec *SVDRec;

/* Abstract matrix class */
struct matrix {
  Matrix (*transposed)(Matrix A);
  void (*free)(Matrix A);
  void (*mat_by_vec)(Matrix A, double *vec, double *out);
  void (*mat_transposed_by_vec)(Matrix A, double *vec, double *out);
  long rows;
  long cols;
  long vals;     /* Total specified entries. */
};

/* Harwell-Boeing sparse matrix. */
struct smat {
  struct matrix h;
  long *pointr;  /* For each col (plus 1), index of first non-zero entry. */
  long *rowind;  /* For each nz entry, the row index. */
  double *value; /* For each nz entry, the value. */
  double *offset_for_row;
  double *offset_for_col;
};

/* Row-major dense matrix.  Rows are consecutive vectors. */
struct dmat {
  struct matrix h;
  double **value; /* Accessed by [row][col]. Free value[0] and value to free.*/
};

/* summing matrix */
struct summing_mat {
  struct matrix h;
  int n;
  Matrix *mats;
};

struct svdrec {
  int d;      /* Dimensionality (rank) */
  DMat Ut;    /* Transpose of left singular vectors. (d by m)
                 The vectors are the rows of Ut. */
  double *S;  /* Array of singular values. (length d) */
  DMat Vt;    /* Transpose of right singular vectors. (d by n)
                 The vectors are the rows of Vt. */
};


/******************************** Variables **********************************/

/* Version info */
extern char *SVDVersion;

/* How verbose is the package: 0, 1 (default), 2 */
extern long SVDVerbosity;

/* Counter(s) used to track how much work is done in computing the SVD. */
enum svdCounters {SVD_MXV, SVD_COUNTERS};
extern long SVDCount[SVD_COUNTERS];
extern void svdResetCounters(void);

enum svdFileFormats {SVD_F_STH, SVD_F_ST, SVD_F_SB, SVD_F_DT, SVD_F_DB};
/*
File formats:
SVD_F_STH: sparse text, SVDPACK-style
SVD_F_ST:  sparse text, SVDLIB-style
SVD_F_DT:  dense text
SVD_F_SB:  sparse binary
SVD_F_DB:  dense binary
*/

/* True if a file format is sparse: */
#define SVD_IS_SPARSE(format) ((format >= SVD_F_STH) && (format <= SVD_F_SB))


/******************************** Functions **********************************/

/* Creates an empty dense matrix. */
extern DMat svdNewDMat(int rows, int cols);
/* Frees a dense matrix. */
extern void svdFreeDMat(DMat D);

/* Creates an empty sparse matrix. */
SMat svdNewSMat(int rows, int cols, int vals);
/* Frees a sparse matrix. */
void svdFreeSMat(SMat S);

/* Summing matrix operations */
SummingMat summing_mat_new(int n);
void summing_mat_set(SummingMat mat, int i, Matrix m);
void summing_mat_free(SummingMat m);
SummingMat summing_mat_transposed(SummingMat m);


/* Creates an empty SVD record. */
SVDRec svdNewSVDRec(void);
/* Frees an svd rec and all its contents. */
void svdFreeSVDRec(SVDRec R);

/* Converts a sparse matrix to a dense one (without affecting former) */
DMat svdConvertStoD(SMat S);
/* Converts a dense matrix to a sparse one (without affecting former) */
SMat svdConvertDtoS(DMat D);

/* Transposes a dense matrix (returning a new one) */
DMat svdTransposeD(DMat D);
/* Transposes a sparse matrix (returning a new one) */
SMat svdTransposeS(SMat S);
/* Copy a vector */
double *copyVector(double* vec, int n, const char* name);


/* Performs the las2 SVD algorithm and returns the resulting Ut, S, and Vt. */
extern SVDRec svdLAS2(Matrix A, long dimensions, long iterations, double end[2], 
                      double kappa);
/* Chooses default parameter values.  Set dimensions to 0 for all dimensions: */
extern SVDRec svdLAS2A(Matrix A, long dimensions);

void freeVector(double *v);
double *mulDMatSlice(DMat D1, DMat D2, int index, double *weight);
double *dMatNorms(DMat D);

#endif /* SVDLIB_H */
