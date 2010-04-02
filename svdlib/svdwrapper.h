#include <Python.h>
#include <numpy/arrayobject.h>
#include <svdlib.h>

void init_numpy(void);

PyObject * wrapDMat(DMat d);

PyObject * wrap_double_array(double* d, int len);

PyObject * wrapSVDrec(struct svdrec * rec, int transposed);
