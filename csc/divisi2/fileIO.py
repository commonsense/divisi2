from __future__ import with_statement
import cPickle as pickle

"""
Easy functions for loading and saving Divisi matrices.

New in Divisi2.
"""

def _meta_open(filename, mode='r'):
    if filename.endswith('.gz'):
        import gzip
        opener = gzip.open
    else:
        opener = open
    return opener(filename, mode)

def load(filename):
    """
    Load a Divisi matrix (or anything else, really) from the given pickle file.
    If the filename ends in .gz, it will be uncompressed.
    """
    with _meta_open(filename) as file:
        return pickle.load(file)

def save(matrix, filename):
    """
    Save a Divisi matrix (or anything else, really) to the given pickle file.
    The filename can end in .gz, in which case it will be compressed.
    """
    if isinstance(matrix, basestring) and not isinstance(filename, basestring):
        # correct for reversed arguments.
        filename, matrix = matrix, filename
    with _meta_open(filename, 'wb') as file:
        pickle.dump(matrix, file, -1)
    
