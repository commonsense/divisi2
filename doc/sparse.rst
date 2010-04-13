.. module:: csc.divisi2.sparse

Sparse Matrices and Vectors
===========================

.. testsetup::

    from csc import divisi2
    mat1 = divisi2.make_sparse([
        (2, "apple", "red"),
        (2, "orange", "orange"),
        (1, "apple", "green"),
        (1, "celery", "green"),
    ])
    mat2 = divisi2.make_sparse([
        (1, "apple", "fruit"),
        (1, "orange", "fruit"),
        (1, "orange", "orange"),
        (2, "celery", "vegetable"),
    ])

Divisi2 uses PySparse as a representation of sparse matrices that can be sliced
and index like NumPy matrices. Divisi's :class:`SparseMatrix` class wraps the
PysparseMatrix class to present an interface that's consistent with the rest of
Divisi. It also comes in a :class:`SparseVector` version to represent 1-D data.

SparseMatrix
------------

A SparseMatrix is a sparse matrix whose rows and columns can
optionally have *labels*. When they do, they are referred to as *named*
rows and columns.
    
The underlying implementation represents each row as a linked list in C,
using the `pysparse` library. The linked list representation makes some
sparse matrix operations particularly fast, at the expense of making it
inefficient to update cells one at a time. So we suggest that you build a
SparseMatrix all at once, from an existing list of data.

This is easy to do with the :meth:`from_named_entries` factory method,
which we also give the name `divisi2.make_sparse` because we think it's the
best way to create a sparse matrix.  It takes in a list of tuples, each of
which contains a value, a row name, and a column name.

.. doctest::

    >>> from csc import divisi2
    >>> mat1 = divisi2.make_sparse([
    ...     (2, "apple", "red"),
    ...     (2, "orange", "orange"),
    ...     (1, "apple", "green"),
    ...     (1, "celery", "green"),
    ... ])
    >>> print mat1
    SparseMatrix (3 by 3)
             red        orange     green   
    apple    2.000000      ---     1.000000  
    orange      ---     2.000000      ---    
    celery      ---        ---     1.000000  

Let's make one more matrix to work with:

.. doctest::

    >>> mat2 = divisi2.make_sparse([
    ...     (1, "apple", "fruit"),
    ...     (1, "orange", "fruit"),
    ...     (1, "orange", "orange"),
    ...     (2, "celery", "vegetable"),
    ... ])
    >>> print mat2
    SparseMatrix (3 by 3)
             fruit      orange     vegetabl
    apple    1.000000      ---        ---    
    orange   1.000000   1.000000      ---    
    celery      ---        ---     2.000000  

Arithmetic operations
---------------------

One simple thing to do with a SparseMatrix is to multiply or divide
it by a scalar.

.. doctest::
   
    >>> print mat1/2
    SparseMatrix (3 by 3)
             red        orange     green   
    apple    1.000000      ---     0.500000  
    orange      ---     1.000000      ---    
    celery      ---        ---     0.500000  
    
    >>> print mat1*0.1
    SparseMatrix (3 by 3)
             red        orange     green   
    apple    0.200000      ---     0.100000  
    orange      ---     0.200000      ---    
    celery      ---        ---     0.100000  

More interesting are operations on multiple matrices. For example, we can
add two sparse matrices, and the label sets will be combined appropriately.

.. doctest::

    >>> print mat1+mat2
    SparseMatrix (3 by 5)
             red        orange     green      fruit      vegetabl
    apple    2.000000      ---     1.000000   1.000000      ---    
    orange      ---     3.000000      ---     1.000000      ---    
    celery      ---        ---     1.000000      ---     2.000000  

    >>> print mat1-2*mat2
    SparseMatrix (3 by 5)
             red        orange     green      fruit      vegetabl
    apple    2.000000      ---     1.000000  -2.000000      ---    
    orange      ---        ---        ---    -2.000000      ---    
    celery      ---        ---     1.000000      ---    -4.000000  

Fancy indexing
--------------

Matrices can be sliced and indexed with "fancy indexing", as if they
were NumPy matrices. Each index can be a single number, a slice
(such as `2:5` to get rows 2, 3, and 4), the slice `:` (selecting
everything), or a list of particular indices.

.. doctest::

    >>> print mat1[:2]
    SparseMatrix (2 by 3)
             red        orange     green   
    apple    2.000000      ---     1.000000  
    orange      ---     2.000000      ---    
    
    >>> print mat1[[1,2], ::-1]
    SparseMatrix (2 by 3)
             green      orange     red     
    orange      ---     2.000000      ---    
    celery   1.000000      ---        ---    

If the result of indexing has a single dimension, the result will be a
:class:`SparseVector`.

.. doctest::

    >>> print mat1[0]
    SparseVector (2 of 3 entries): [red=2, green=1]
    >>> print mat1[:,0]
    SparseVector (1 of 3 entries): [apple=2]

Sparse multiplication
---------------------

To do matrix multiplication, we need to make sure the inner labels match.
We can accomplish this by transposing `mat1`.
    
.. doctest::

    >>> print mat1.T
    SparseMatrix (3 by 3)
             apple      orange     celery  
    red      2.000000      ---        ---    
    orange      ---     2.000000      ---    
    green    1.000000      ---     1.000000  

    >>> mat1.T.col_labels == mat2.row_labels
    True

Now we can take the sparse matrix product of `mat1.T` and `mat2`, giving
us a matrix that relates the set `[red, orange, green]` to the set
`[fruit, orange, vegetable]` through the set `[apple, orange, celery]`.

.. doctest::

    >>> print divisi2.dot(mat1.T, mat2)
    SparseMatrix (3 by 3)
             fruit      orange     vegetabl
    red      2.000000      ---        ---    
    orange   2.000000   2.000000      ---    
    green    1.000000      ---     2.000000  
    
It turns out to be much more efficient to transpose as part
of the product operation, instead of transposing as a separate step.
Divisi supports this:

.. doctest::

    >>> print divisi2.transpose_dot(mat1, mat2)
    SparseMatrix (3 by 3)
             fruit      orange     vegetabl
    red      2.000000      ---        ---    
    orange   2.000000   2.000000      ---    
    green    1.000000      ---     2.000000  
    
Be sure to distinguish *matrix multiplication* from *elementwise
multiplication*, which takes in two matrices of the same shape.
We can also do sparse elementwise multiplication, which of course
has non-zero values only where the two matrices overlap:

.. doctest::

    >>> print divisi2.multiply(mat1, mat2)
    SparseMatrix (3 by 5)
             red        orange     green      fruit      vegetabl
    apple       ---        ---        ---        ---        ---    
    orange      ---     2.000000      ---        ---        ---    
    celery      ---        ---        ---        ---        ---    
    
The `*` operator is used *very* inconsistently in Python matrix classes.
To avoid confusion, sparse matrices do not support `*` -- you need to
explicitly ask for :meth:`multiply` or :meth:`dot`.

Normalization
-------------

Normalization makes sure that rows or columns have the same Euclidean
magnitude. 
    
.. doctest::

    >>> print mat1.normalize_rows()
    SparseMatrix (3 by 3)
             red        orange     green   
    apple    0.894427      ---     0.447214  
    orange      ---     1.000000      ---    
    celery      ---        ---     1.000000  

    >>> print mat1.normalize_cols()
    SparseMatrix (3 by 3)
             red        orange     green   
    apple    1.000000      ---     0.707107  
    orange      ---     1.000000      ---    
    celery      ---        ---     0.707107  

It's impossible to make such a guarantee for both directions at once,
except by throwing out most of the information and diagonalizing the
matrix. So `normalize_all` goes halfway to normalization in both
directions, instead, dividing each by the square root of the norm.
    
.. doctest::

    >>> print mat1.normalize_all()
    SparseMatrix (3 by 3)
             red        orange     green   
    apple    0.945742      ---     0.562341  
    orange      ---     1.000000      ---    
    celery      ---        ---     0.840896  

API documentation
-----------------
.. autoclass:: AbstractSparseArray
   :members:

.. autoclass:: SparseMatrix
   :members:
   :show-inheritance:

.. autoclass:: SparseVector
   :members:
   :show-inheritance:


