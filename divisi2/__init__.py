"""
This is the top level of the Divisi2 package, which imports some commonly-used
functions and classes.
"""

from divisi2.exceptions import Error, LabelError, DimensionMismatch
from divisi2.fileIO import load, save
from divisi2.sparse import SparseMatrix, SparseVector, category
from divisi2.dense import DenseMatrix, DenseVector
from divisi2.blending import blend
from divisi2.operators import *
from divisi2.reconstructed import reconstruct, reconstruct_symmetric, reconstruct_similarity, reconstruct_activation
from divisi2 import dataset
from divisi2 import network
from divisi2.ordered_set import OrderedSet

make_sparse = SparseMatrix.from_named_entries
