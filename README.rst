This project is no longer maintained
====================================

Divisi2 was a library for reasoning by analogy over semantic networks using the sparse singular-value decomposition, originating in a time when the sparse SVD was (a) the most effective source of word vectors and (b) difficult to perform in Python. Both of these situations have changed.

conceptnet5_ contains code for building multilingual word vectors based on distributional semantics and the knowledge graph ConceptNet. These word vectors can reason about words by similarity and analogy, with state-of-the-art performance as of 2017.

Other libraries that can help to accomplish the lower-level operations of Divisi2:

* SciPy_ now has built-in sparse matrices, and `scipy.sparse.linalg` can perform a sparse SVD.
* pandas_ is an excellent library for working with matrices of labeled data.

.. _conceptnet5: https://github.com/commonsense/conceptnet5
.. _SciPy: https://www.scipy.org/
.. _pandas: http://pandas.pydata.org/

Authors
=======
Divisi2 belongs to two projects with many of the same people involved:

- Open Mind Common Sense, a project of the MIT Media Lab
- the MIT Mind Machine Project

The primary developers are:

- Rob Speer <rspeer at mit dot edu>
- Ken Arnold <kcarnold at mit dot edu>

See AUTHORS.rst for a list of all authors.

License
=======

This version of Divisi is available under the GNU General Public License,
version 3.0. See COPYING.txt.
