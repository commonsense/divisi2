from math import log
from divisi2.sparse import SparseMatrix
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
        if data and data['score'] > 0:
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
    '''
    # The to_undirected function is tempting, but it copies all the
    # data, which is unnecessary for this algorithm.
    ugraph = nx.Graph()
    ugraph.add_nodes_from(graph)
    ugraph.add_edges_from(graph.edges_iter())
    cores = nx.find_cores(ugraph)
    core_nodes = [n for n in graph.nodes() if cores[n] >= cutoff]
    return graph.subgraph(core_nodes)

def relation_filter(relation):
    if isinstance(relation, basestring): relation = [relation]
    def filter(source, target, data):
        return data['rel'] in relation
    return filter

LABELERS = {
    'nodes': _extract_nodes,
    'concepts': _extract_nodes,       # synonym for 'nodes'
    'features': _extract_features,
    'pairs': _extract_pairs,
    'relations': _extract_relations,
    'source_only': _extract_source_only,
    'target_only': _extract_target_only,
}

def sparse_triples(graph, row_labeler, col_labeler, cutoff=1, filter=None):
    """
    A generator of sparse triples to put into a matrix, given a NetworkX graph.
    It is assumed that each edge of the graph yields two entries of the matrix.
    
    `row_labeler` and `col_labeler` are functions that are given each edge
    as a tuple of (source, target, data), and choose two rows and columns for
    the matrix. The first row is paired with the second column and vice versa.

    In practice, you don't need to worry abaout that, because `row_labeler`
    and `col_labeler` can also be strings choosing a predefined function.
    To get an adjacency matrix that relates nodes to nodes, for example, use::

        divisi2.network.sparse_triples(graph, 'nodes', 'nodes')

    To get an AnalogySpace concept-by-feature matrix:

        divisi2.network.sparse_triples(graph, 'nodes', 'features')

    To get a pair-relation matrix, as in Latent Relational Analysis:

        divisi2.network.sparse_triples(graph, 'pairs', 'relations')
    
    `cutoff` specifies the minimum degree of nodes to include.
    
    The edge weights should be expressed in one of two forms:

    - The standard way for NetworkX, as the entry named 'weight' in the edge
      data dictionaries.
    - As ConceptNet-style 'score' and 'freq' values in the edge data
      dictionaries, which will be transformed into appropriate weights.

    If no edge weights can be found, the edges will be given a default weight
    of 1.
    """
    first_edge = graph.edges_iter(data=True).next()
    first_data = first_edge[2]
    if 'score' in first_data and 'weight' not in first_data:
        set_conceptnet_weights(graph)

    try:
        if isinstance(row_labeler, basestring):
            row_labeler = LABELERS[row_labeler]
        if isinstance(col_labeler, basestring):
            col_labeler = LABELERS[col_labeler]
    except KeyError:
        raise KeyError("Unknown row or column type. The valid types are: %s"
          % sorted(LABELERS.keys()))
    subgraph = prune(graph, cutoff=cutoff)
    for edge in subgraph.edges_iter(data=True):
        if filter is not None and not filter(*edge): continue
        rows = row_labeler(*edge)
        cols = col_labeler(*edge)
        weight = edge[2].get('weight', 1)
        yield (weight, rows[0], cols[1])
        yield (weight, rows[1], cols[0])

def sparse_matrix(graph, row_labeler, col_labeler, cutoff=1):
    """
    Constructs a :class:`SparseMatrix` from a graph. See the documentation
    for :func:`sparse_triples` for how to choose the `row_labeler` and
    `col_labeler` to build different kinds of matrices.
    """
    matrix_builder = SparseMatrix.from_named_entries
    if row_labeler == col_labeler:
        matrix_builder = SparseMatrix.square_from_named_entries
    return matrix_builder(
      list(sparse_triples(graph, row_labeler, col_labeler, cutoff))
    )

def filter_by_relation(concept_feature_matrix, relation):
    '''
    Returns only the part of the matrix about the given relation.

    Be aware that empty rows may remain.
    '''
    from divisi2.operators import filter_labels
    all_features = concept_feature_matrix.col_labels
    features_to_keep = [feature for feature in all_features
                        if feature[1] == relation]
    return filter_labels(concept_feature_matrix, cols=features_to_keep)

def conceptnet_matrix(lang):
    # load from the included pickle file
    import divisi2
    try:
        matrix = divisi2.load('data:matrices/conceptnet_%s.gz' % lang)
        return matrix
    except IOError:
        graph = divisi2.load('data:graphs/conceptnet_%s.graph.gz' % lang)
        matrix = sparse_matrix(graph, 'concepts', 'features', 3)
        divisi2.save(matrix, 'data:matrices/conceptnet_%s.gz' % lang)
        return matrix

def conceptnet_assoc(lang):
    import divisi2
    try:
        matrix = divisi2.load('data:matrices/conceptnet_assoc_%s.gz' % lang)
        return matrix
    except IOError:
        graph = divisi2.load('data:graphs/conceptnet_%s.graph.gz' % lang)
        matrix = sparse_matrix(graph, 'concepts', 'concepts', 3)
        divisi2.save(matrix, 'data:matrices/conceptnet_assoc_%s.gz' % lang)
        return matrix

analogyspace_matrix = conceptnet_matrix   # synonym
