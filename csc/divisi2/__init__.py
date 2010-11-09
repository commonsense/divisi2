"""
This is the top level of the Divisi2 package, which imports some commonly-used
functions and classes.
"""

from csc.divisi2.exceptions import Error, LabelError, DimensionMismatch
from csc.divisi2.fileIO import load, save
from csc.divisi2.sparse import SparseMatrix, SparseVector, category
from csc.divisi2.dense import DenseMatrix, DenseVector
from csc.divisi2.blending import blend
from csc.divisi2.operators import *
from csc.divisi2.reconstructed import reconstruct, reconstruct_symmetric, reconstruct_similarity, reconstruct_activation
from csc.divisi2 import dataset
from csc.divisi2 import network
from csc.divisi2.ordered_set import OrderedSet

make_sparse = SparseMatrix.from_named_entries
