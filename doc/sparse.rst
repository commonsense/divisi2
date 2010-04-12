.. module:: csc.divisi2.sparse

Sparse Matrices and Vectors
===========================

Divisi2 uses PySparse as a representation of sparse matrices that can be sliced
and index like NumPy matrices. Divisi's :class:`SparseMatrix` class wraps the
PysparseMatrix class to present an interface that's consistent with the rest of
Divisi. It also comes in a :class:`SparseVector` version to represent 1-D data.

The SparseMatrix class
----------------------
.. autoclass:: SparseMatrix

The SparseVector class
----------------------
.. autoclass:: SparseVector

The AbstractSparseArray class
-----------------------------
.. autoclass:: AbstractSparseArray

