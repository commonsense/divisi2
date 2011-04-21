from __future__ import with_statement
from pkg_resources import resource_filename
import cPickle as pickle
import codecs
import gzip
"""
Easy functions for loading and saving Divisi matrices and semantic networks.

New in Divisi2.
"""

# Note: gzip.GzipFile is super-slow for things that read a few bytes at a time,
# but quite fast to read the whole file at once. So we do the latter.
#
# If you run out of memory reading a gzip'ed file, un-gzip it first.

def data_filename(filename):
    if filename.startswith('data:'):
        filename = resource_filename(__name__, 'data/'+filename[5:])
    return filename

def _meta_read(filename, encoding=None):
    filename = data_filename(filename)
    opener = gzip.open if filename.endswith('.gz') else open
    f = opener(filename, 'rb')
    data = f.read()
    f.close()
    if encoding is not None:
        data = data.decode(encoding)
    return data

def _meta_write(filename, data, encoding=None):
    filename = data_filename(filename)
    if encoding is not None:
        data = data.encode(encoding)
    opener = gzip.open if filename.endswith('.gz') else open
    f = opener(filename, 'wb')
    f.write(data)
    f.close()

# def _meta_open(filename, mode='rb', encoding=None):
#     if filename.endswith('.gz'):
#         raise RuntimeError('Opening gzip files is no longer supported.')
#     if encoding is None:
#         return open(filename, mode)
#     else:
#         return codecs.open(filename, mode, encoding=encoding)

def load(filename):
    """
    Load an object (most likely a Divisi matrix or a semantic network) from a
    .pickle or .graph file. If the filename ends in .gz, it will be
    uncompressed.
    """
    if filename.endswith('.graph.gz') or filename.endswith('.graph'):
        return load_graph(filename)
    else:
        return load_pickle(filename)

def load_pickle(filename):
    file = _meta_read(filename)
    return pickle.loads(file)

def load_graph(filename, encoding='utf-8'):
    import networkx as nx
    return nx.read_edgelist(data_filename(filename), encoding=encoding,
                            data=True, delimiter='\t',
                            create_using=nx.MultiDiGraph())

def save(obj, filename):
    """
    Save an object to the given filename.

    If the filename ends in .gz, the file will be compressed. If aside from
    a possible .gz, the filename ends in .graph, it will assume that your
    object is a semantic network and save it in the NetworkX edgelist format.
    """
    if isinstance(obj, basestring) and not isinstance(filename, basestring):
        # correct for reversed arguments.
        filename, obj = obj, filename
    if filename.endswith('.graph') or filename.endswith('.graph.gz'):
        save_graph(obj, filename)
    else:
        save_pickle(obj, filename)

def save_graph(network, filename, encoding='utf-8'):
    import networkx as nx
    return nx.write_edgelist(network, data_filename(filename), data=True, delimiter='\t')

def save_pickle(matrix, filename):
    if isinstance(matrix, basestring) and not isinstance(filename, basestring):
        # Catch accidentally reversing argument order.
        return save_pickle(filename, matrix)
    _meta_write(filename, pickle.dumps(matrix, -1))
