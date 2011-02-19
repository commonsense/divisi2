"""
The ``export_svdview`` module allows SVD results to be visualized using the
separate program ``svdview`` (http://launchpad.net/svdview).

Denormalization
===============

Concepts are often stored in Divisi tensors in a normalized form,
which is often not human-friendly. The ``denormalize`` callback
provides a way "undo" the normalization as concepts are returned. A
denormalizer for ConceptNet concepts is provided, which returns the
"canonical name" of concepts.

File formats
============

Binary format
-------------

The binary format is newer and faster. It consists of a header and a
body (everything is stored in big-endian (network) byte order):

Header:
 * 4 bytes: number of dimensions (integer)
 * 4 bytes: number of items (integer)

The body is a sequence of items with no separator. Each item has a
coordinate for each dimension. Each coordinate is an IEEE float
(32-bit) in big-endian order.

TSV format
----------

The old TSV format is easier to edit by hand or with simple
scripts. Each line is a sequence of fields separated by tabs. The
first field on each line is the concept name. It is followed by a
floating point number for each dimension.

"""
from itertools import izip, imap, count
import codecs

def denormalize(concept_text):
    '''
    Returns the canonical denormalized (user-visible) form of a
    concept, given its normalized text of a concept.
    '''
    from conceptnet.models import Concept

    if isinstance(concept_text, tuple):
        text, lang = concept_text
    else:
        text, lang = concept_text, 'en'
    try:
        concept = Concept.get_raw(text, lang)
        result = concept.canonical_name.lower()
    except Concept.DoesNotExist:
        result = text
    if lang != 'en': return '%s [%s]' % (result, lang)
    else: return result

def null_denormalize(x): return x

def _sorted_rowvectors(m, denormalize, num_dims):
    def fix_concept(c):
        if not c: return '_'
        return unicode(c).encode('utf-8')
    for row in xrange(m.shape[0]):
        concept = fix_concept(denormalize(m.row_label(row)))
        yield concept, m[row]

def write_tsv(matrix, outfn, denormalize=None, cutoff=40):
    '''
    Export a tab-separated value file that can be visualized
    with svdview. The data is saved to the file named _outfn_.
    '''
    if denormalize is None: denormalize = null_denormalize
    num_vecs, num_dims = matrix.shape
    if num_dims > cutoff: num_dims = cutoff

    out = open(outfn, 'wb')
    for concept, vec in _sorted_rowvectors(matrix, denormalize, num_dims):
        datastr = '\t'.join(imap(str, vec))
        out.write("%s\t%s\n" % (concept, datastr))
    out.close()

def write_packed(matrix, out_basename, denormalize=None, cutoff=40):
    '''
    Export in the new binary coordinate file format.
    '''

    import struct
    names = open(out_basename+'.names','wb')
    coords = open(out_basename+'.coords', 'wb')

    if denormalize is None: denormalize = null_denormalize

    num_vecs, num_dims = matrix.shape
    if num_dims > cutoff: num_dims = cutoff
    coords.write(struct.pack('>ii', num_dims, num_vecs))

    # Write the whole file.
    format_str = '>' + 'f'*num_dims
    for concept, vec in _sorted_rowvectors(matrix, denormalize, num_dims):
        coords.write(struct.pack(format_str, *vec[:cutoff]))
        names.write(concept+'\n')

    names.close()
    coords.close()

def feature_str(feature):
    if not isinstance(feature, tuple): return str(feature)
    if len(feature) != 3: return str(feature)
    if feature[0] == 'left':
        return "%s\%s" % (feature[2], feature[1])
    elif feature[0] == 'right':
        return "%s/%s" % (feature[1], feature[2])
    else: return str(feature)

