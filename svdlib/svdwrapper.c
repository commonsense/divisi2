#include <Python.h>
#include <numpy/arrayobject.h>
#include <svdlib.h>
#include <stdio.h>

/*
  Mock Python object just to handle freeing the memory.
  From http://blog.enthought.com/?p=62
*/
typedef struct {
  PyObject_HEAD
  void *memory;
  int multidim; /* flag to say we should free memory[0] first. */
} _MyDeallocObject;

static void
_mydealloc_dealloc(PyObject* _self)
{
  _MyDeallocObject *self = (_MyDeallocObject*) _self;
  if (self->multidim) free(((double **)self->memory)[0]);
  free(self->memory);
  self->ob_type->tp_free((PyObject *)self);
}

static PyTypeObject _myDeallocType = {
  PyObject_HEAD_INIT(NULL)
  0,                                          /*ob_size*/
  "mydeallocator",                   /*tp_name*/
  sizeof(_MyDeallocObject),    /*tp_basicsize*/
  0,                                          /*tp_itemsize*/
  _mydealloc_dealloc,             /*tp_dealloc*/
  0,                         /*tp_print*/
  0,                         /*tp_getattr*/
  0,                         /*tp_setattr*/
  0,                         /*tp_compare*/
  0,                         /*tp_repr*/
  0,                         /*tp_as_number*/
  0,                         /*tp_as_sequence*/
  0,                         /*tp_as_mapping*/
  0,                         /*tp_hash */
  0,                         /*tp_call*/
  0,                         /*tp_str*/
  0,                         /*tp_getattro*/
  0,                         /*tp_setattro*/
  0,                         /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT,        /*tp_flags*/
  "Internal deallocator object",           /* tp_doc */
};

PyObject* make_deallocator_for(void* mem, int multidim) {
  static int dealloc_inited = 0;
  if (!dealloc_inited) {
    _myDeallocType.tp_new = PyType_GenericNew;
    PyType_Ready(&_myDeallocType);
    dealloc_inited = 1;
  }

  _MyDeallocObject* newobj = PyObject_New(_MyDeallocObject, &_myDeallocType);
  newobj->memory = mem;
  newobj->multidim = multidim;
  return (PyObject*)newobj;
}

static int _numpy_inited = 0;
void init_numpy(void) {
  if (_numpy_inited) return;
  import_array();
  _numpy_inited = 1;
}

PyObject * wrapDMat(DMat d) {
  init_numpy();
  PyArray_Descr *type = PyArray_DescrFromType(NPY_DOUBLE);
  npy_intp dim[2] = {d->h.rows, d->h.cols};
  npy_intp strides[2] = {d->h.cols*sizeof(double), sizeof(double)};
  PyObject* arr = PyArray_NewFromDescr( &PyArray_Type, type,
					2, dim, strides,
					d->value[0],
					NPY_CONTIGUOUS | NPY_WRITEABLE,
					NULL /*PyObject *obj*/ ); 
  PyArray_BASE(arr) = make_deallocator_for(d->value, 1);
  return arr;
}

PyObject * wrap_double_array(double* d, int len) {
  init_numpy();
  PyArray_Descr *type = PyArray_DescrFromType(NPY_DOUBLE);
  npy_intp dim[1] = {len};
  npy_intp strides[1] = {sizeof(double)};
  PyObject* arr = PyArray_NewFromDescr( &PyArray_Type, type,
					1, dim, strides,
					d,
					NPY_CONTIGUOUS | NPY_WRITEABLE,
					NULL /*PyObject *obj*/ ); 
  PyArray_BASE(arr) = make_deallocator_for(d, 0);
  return arr;
}


PyObject * wrapSVDrec(struct svdrec * rec, int transposed) {
  PyObject * ut = wrapDMat(rec->Ut);
  PyObject * s = wrap_double_array(rec->S, rec->d);
  PyObject * vt = wrapDMat(rec->Vt);

  PyObject * result = PyTuple_New(3);
  PyTuple_SetItem(result, 1, s);
  if (transposed) {
     PyTuple_SetItem(result, 0, vt);
     PyTuple_SetItem(result, 2, ut);
  } else {
     PyTuple_SetItem(result, 0, ut);
     PyTuple_SetItem(result, 2, vt);
  }
  return result;
}
