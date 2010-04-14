Labeling data
=============

.. module:: csc.divisi2.labels

Labeling vectors and matrices
-----------------------------
Divisi structures, whether they are vectors or matrices, sparse or dense,
have the ability to track the meaning of their data with labels. This is
enabled through the :class:`LabeledVectorMixin` and :class:`LabeledMatrixMixin`
classes.

Label lists are instances of the :class:`OrderedSet` class, described below.
This allows an efficient one-to-one correspondence between labels and *indices*
(the actual positions of rows or columns). If you have a label, you can quickly
find out its index. If you have an index, you can quickly find out its label.

The two *modes* of a matrix -- that is, rows and columns -- have their own
label lists, which are stored in the properties `matrix.row_labels` and
`matrix.col_labels`. Vectors do not distinguish rows from columns, so they have
a single label list, `vector.labels`.

Label lists may also be `None`, indicating that there are no meaningful labels
besides the indices themselves. In that case, the labels will be treated as
being identical to the indices. For example, in a matrix whose row labels are
`None`, then the top row of the matrix, whose index is 0, will also be
considered to have the label `0`. The next row will have the label `1`, and so
on.

API documentation
.................

.. autoclass:: LabeledVectorMixin
   :members:

.. autoclass:: LabeledMatrixMixin
   :members:




.. module:: csc.divisi2.ordered_set

The OrderedSet class
--------------------

An OrderedSet acts very much like a list. There are two important
differences:

- Each item appears in the list only once.
- You can look up an item's index in the list in constant time.

Use it like you would use a list. All the standard operators are defined.

    >>> from csc.divisi2.ordered_set import OrderedSet

.. testsetup::

    from csc.divisi2.ordered_set import OrderedSet

Look up the index of 'banana' in a set:

.. doctest::

    >>> s = OrderedSet(['apple', 'banana', 'pear'])
    >>> s.index('banana')
    1
    >>> s.indexFor('banana')  # (synonym)
    1

Look up an unknown index:

.. doctest::

    >>> s = OrderedSet(['apple', 'banana', 'pear'])
    >>> s.index('automobile')
    Traceback (most recent call last):
        ...
    KeyError: 'automobile'

Add a new item:

.. doctest::

    >>> s = OrderedSet(['apple', 'banana', 'pear'])
    >>> s.add('orange')
    3
    >>> s.index('orange')
    3

Add an item that's already there:

.. doctest::

    >>> s = OrderedSet(['apple', 'banana', 'pear'])
    >>> s.add('apple')
    0

Extend with some more items:

.. doctest::

    >>> s = OrderedSet(['apple', 'banana', 'pear'])
    >>> s.extend(['grapefruit', 'kiwi'])
    >>> s.index('grapefruit')
    3

See that it otherwise behaves like a list:

.. doctest::

    >>> s = OrderedSet(['apple', 'banana', 'pear'])
    >>> s[0]
    'apple'
    >>> s[0] = 'Apple'
    >>> s[0]
    'Apple'
    >>> len(s)
    3
    >>> for item in s:
    ...     print item,
    Apple banana pear

``None`` element is used as a placeholder for non-present
elements, but it is never semantically an element of the set:

.. doctest::

    >>> s = OrderedSet(['apple', 'banana', 'pear'])
    >>> del s[0]
    >>> s[0] is None
    True
    >>> s.index('banana')
    1
    >>> None in s
    False

API documentation
.................

.. autoclass:: OrderedSet
   :members:

