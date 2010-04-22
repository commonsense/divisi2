from csc.divisi2.crossbridge import *
import networkx as nx

def test_k_edge_subgraphs():
    """
    Test on a somewhat larger graph. 
    """
    g = nx.Graph()
    g.add_edges_from(zip([1,2,3,4,5,6], [2, 3, 4, 5, 6, 1]))
    g.add_edges_from([(2, 6), (2, 5)])
    #there are only two triangles in the graph: (1, 2, 6) and (2, 5, 6)


    subgraphs = k_edge_subgraphs(g, 4, 3, 1)
    assert frozenset([1,2,6]) in subgraphs[(3, 3)] # 3 vertices, 3 edges
    assert frozenset([2,5,6]) in subgraphs[(3, 3)]
    assert len(subgraphs[(3,3)]) == 2

def test_k_edge_subgraphs_using_complete_graphs():
    """
    A k-vertex complete graph has k*(k-1)*(k-2) / 3! triangles in them.
    We can get number of triangles in a graph using k-edge_subgraphs(g, 4, 3)[3,3]
    """
    numer = lambda k: k * (k - 1) * (k - 2)
    expected_triangles = lambda k : numer(k) / 6
    
    for i in xrange(5, 50):
        print "testing complete graphs of size: %s" % i
        g = nx.generators.classic.complete_graph(i)
        actual_triangles = k_edge_subgraphs(g, 4, 3)[3,3]
        assert len(actual_triangles) == expected_triangles(i)
