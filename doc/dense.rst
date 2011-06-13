.. module:: divisi2.dense

Dense Matrices and Vectors
==========================
When Divisi2 creates a dense matrix or vector, it gives you a NumPy-compatible
object. That is, the object is actually an instance of `numpy.ndarray` with
extra data on it. This makes various powerful features available, and makes
these objects compatible with lots of existing numerical code for Python.

Thus, many of the ways to use these objects are actually defined in NumPy.
See the `NumPy tutorial`_ for examples.

.. _`NumPy tutorial`: http://www.scipy.org/Tentative_NumPy_Tutorial

Divisi's objects will attempt to maintain their labels when you run operations
on them.

Unfortunately, this will fail in one case: if you run a NumPy
operation that changes the object's number of dimensions, such as
`np.mean(axis=n)`, you will end up with a malformed Divisi object. Resolving
this would require much deeper hooks into NumPy. The workaround is to convert
your array to a plain, unlabeled NumPy array first, as follows::

    >>> import numpy as np
    >>> numpy_array = np.asarray(divisi_array)
    >>> means = np.mean(numpy_array, axis=1)

API documentation
-----------------
.. autoclass:: AbstractDenseArray
   :members:

.. autoclass:: DenseMatrix
   :members:
   :show-inheritance:

.. autoclass:: DenseVector
   :members:
   :show-inheritance:

