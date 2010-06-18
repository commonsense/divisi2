from math import log
from csc.divisi2.sparse import SparseMatrix
import networkx as nx
log_2 = log(2)

def set_conceptnet_weights(graph):
    """
    Given a ConceptNet-style graph representation, with a 'score' and 'freq'
    on each node, combine these into a single 'weight' value that grows
    linearly with the frequency and logarithmically with the score.
    """
    def conceptnet_weight(score, freq):
        return (freq/10.0) * log(max((score+1, 1)))/log_2
    
    def set_weight(edgedata):
        score = edgedata['score']
        freq = edgedata['freq']
        edgedata['weight'] = conceptnet_weight(score, freq)

    data_foreach(graph, set_weight)

def data_foreach(graph, func):
    """
    Applies a function that mutates the data dictionary of each edge in
    a graph. The return value of the function is not used.
    """
    for source, dest, data in graph.edges_iter(data=True):
        func(data)

def _extract_nodes(source, target, data):
    return (source, target)

def _extract_features(source, target, data):
    return (('left', data['rel'], source), ('right', data['rel'], target))

def _extract_pairs(source, target, data):
    return ((source, target), (target, source))

def _extract_relations(source, target, data):
    return (data['rel'], data['rel'])

def _extract_source_only(source, target, data):
    return (source, source)

def _extract_target_only(source, target, data):
    return (target, target)

def prune(graph, cutoff=1):
    '''
    Only keep nodes with a connectivity >= cutoff.

