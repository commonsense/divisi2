from __future__ import with_statement
from pkg_resources import resource_filename
import cPickle as pickle
import codecs
"""
Easy functions for loading and saving Divisi matrices and semantic networks.

New in Divisi2.
"""

def _meta_open(filename, mode='r'):
    if filename.startswith('data:'):
        filename = resource_filename(__name__, filename[5:])
    
    if filename.endswith('.gz'):
        import gzip
        opener = gzip.open
    else:
        opener = open
    return opener(filename, mode)

def load(filename):
    """
    Load an object (most likely a Divisi matrix or a semantic network) from a
    pickle or GML file. If the filename ends in .gz, it will be uncompressed.
    """
    if filename.endswith('.gml.gz') or filename.endswith('.gml'):
        return load_gml(filename)
    else:
        return load_pickle(filename)

def load_pickle(filename):
    file = _meta_open(filename)
    return pickle.load(file)

def load_gml(filename):
    import networkx as nx
    file = _meta_open(filename)
    file_decoder = codecs.getreader("utf-7")(file)
    return nx.read_gml(file_decoder)

def save(obj, filename):
    """
    Save an object to the given filename.

    If the filename ends in .gz, the file will be compressed. If aside from
    a possible .gz, the filename ends in .gml, it will assume that your
    object is a semantic network and save it in the GML format.
    """
    if isinstance(obj, basestring) and not isinstance(filename, basestring):
        # correct for reversed arguments.
        filename, obj = obj, filename
    if filename.endswith('.gml') or filename.endswith('.gml.gz'):
        save_gml(obj, filename)
    else:
        save_pickle(obj, filename)

def save_gml(network, filename):
    import networkx as nx
    assert isinstance(network, nx.Graph)
    file = _meta_open(filename, 'w')
    file_encoder = codecs.getwriter("utf-7")(file)
    return nx.write_gml(network, file_encoder)

def save_pickle(matrix, filename):
    file = _meta_open(filename, 'wb')
    pickle.dump(matrix, file, -1)

