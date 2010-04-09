"""
This is the top level of the Divisi2 package, which imports some commonly-used
functions and classes.
"""

from csc.divisi2.exceptions import Error, DimensionMismatch
from csc.divisi2.fileIO import load, save
from csc.divisi2.sparse import SparseMatrix, SparseVector, LabelError
from csc.divisi2.blend import blend
from csc.divisi2.operators import *
from csc.divisi2.reconstructed import reconstruct, reconstruct_symmetric, reconstruct_similarity
import network

from_named_entries = SparseMatrix.from_named_entries
from_named_lists = SparseMatrix.from_named_lists
from_entries = SparseMatrix.from_entries
from_lists = SparseMatrix.from_lists
make_sparse = from_named_entries
