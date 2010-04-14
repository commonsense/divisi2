from collections import defaultdict
from csc.divisi.labeled_tensor import SparseLabeledTensor
from semantic_network import SemanticNetwork
import copy
import logging

class CrossBridge(object):
    '''
    This class implements the CrossBridge analogy algorithm.

    Typical usage:
    cnet_graph = cnet_graph_from_tensor(cnet_tensor)
    crossbridge = CrossBridge(cnet_graph) # This may compute for a bit
    # Then one of the following
    analogies, top_relations = crossbridge.analogy(set(['bird', 'wing', 'fly', 'sky']))
    analogies, top_relations = crossbridge.analogy_from_concept('bird')
    '''

    def __init__(self, graph, num_nodes=3, min_graph_edges=3,
                 min_feature_edges=2, max_feature_edges=3, svd_dims=100,
                 logging_interval=None):

        self.num_nodes = num_nodes
        self.graph = graph
        self.tensor = self._build_analogy_tensor(graph, num_nodes, min_graph_edges,
                                                 min_feature_edges, max_feature_edges,
                                                 logging_interval=logging_interval)
        self.svd = self.tensor.svd(k=svd_dims)


    def _build_analogy_tensor(self, graph, num_nodes, min_graph_edges,
                              min_feature_edges, max_feature_edges,
                              logging_interval=None):

        '''
        Builds a tensor T whose rows correspond to maximal dense subgraphs of
        graph and whose columns are relation structures (n-ary relations
        formed by combining multiple binary relations).

        graph - The graph from which the tensor is built
        num_nodes - The size of each vertex set row
        min_graph_edges - The minimum number of edges in the graphs being indexed
        min_feature_edges - The minimum number of edges to include in a graph feature
        max_feature_edges - The minimum number of edges to include in a graph feature

        Note: the min_graph_edges parameter treats the graph as if it
        is undirected and untyped, meaning there is at most 1 edge
        between 2 vertices
        '''
        k_subgraphs = k_edge_subgraphs(graph, min_graph_edges, num_nodes,
                                         logging_interval=logging_interval)
        rel_tensor = SparseLabeledTensor(ndim=2)

        logging_counter = 0
        for size, subgraphs_vertices in k_subgraphs.iteritems():
            if size[0] == num_nodes and size[1] >= min_graph_edges:
                logging.debug("Adding graphs with %d vertices and %d edges (%d total)...",
                              size[0], size[1], len(subgraphs_vertices))
                for subgraph_vertices in subgraphs_vertices:
                    subgraph = graph.subgraph_from_vertices(subgraph_vertices)
                    for subgraph_no_repeats in subgraph.enumerate_without_repeated_edges():
                        if (len(subgraph_no_repeats.edges()) >= min_feature_edges
                            and len(subgraph_no_repeats.edges()) <= max_feature_edges):
                            for vertex_order in enumerate_orders(subgraph_vertices):
                                vm = dict([(y, x) for x, y in enumerate(vertex_order)])
                                edges = frozenset([(vm[v1], vm[v2], value) for v1, v2, value in subgraph_no_repeats.edges()])
                                rel_tensor[vertex_order, edges] = 1

                    if logging_interval is not None and logging_counter % logging_interval == 0:
                        logging.debug("%r", subgraph_vertices)
                    logging_counter += 1

        return rel_tensor


    def analogy(self, source_concept_set, logging_interval=None,
                num_candidates=300, beam_width=10000):
        '''
        Takes a set of source concepts and returns a sorted, scored
        list of analogies for the source concepts and the most
        important relations.

        num_candidates and beam_width are parameters that control the
        number of candidate analogies maintained at various points during
        the search for good analogies.

        Note: The important relations are computed by a very hacky
        algorithm... I (jayant) don't trust them very much
        '''

        '''
        First search for analogies between self.num_nodes concepts.
        '''
        candidate_target_concept_sets = []
        candidate_relations = defaultdict(lambda: 0)
        for concept_triple in k_subset_iter(self.num_nodes, source_concept_set):
            try:
                concept_triple = tuple(concept_triple)
                candidates = self.svd.u_angles_to(self.svd.weighted_u[concept_triple, :]).top_items(num_candidates)
                candidate_target_concept_sets.extend([(frozenset(zip(target_triple, concept_triple)), score) for target_triple, score in candidates])

                inferred_relations = self.svd.v_angles_to(self.svd.weighted_u[concept_triple, :]).top_items(num_candidates)
                for relations, score in inferred_relations:
                    for c1, c2, rel in relations:
                        candidate_relations[(concept_triple[c1], rel, concept_triple[c2])] += score

                logging.debug("Found source concept set: %r", concept_triple)
                logging.debug("Top 10 analogies: %r", candidates[:10])
                logging.debug("Top 10 relations: %r", inferred_relations[:10])
            except KeyError:
                pass

        '''
        Iteratively merge the candidate analogies together by combining
        n-concept analogies containing 2 overlapping concepts.
        '''
        smallest_analogies = copy.copy(candidate_target_concept_sets)
        candidates = [set(candidate_target_concept_sets)]
        iteration_num = 0
        while len(candidates[-1]) != 0:
            logging.debug("Starting iteration %d (%d candidates)", iteration_num, len(candidates[-1]))
            iteration_num += 1

            new_candidates = set()
            candidate_scores = {}
            candidates_with_children = set()
            for mapping, score in candidates[-1]:
                for mapping2, score2 in smallest_analogies:
                    if (len(mapping & mapping2) == 2 and
                        len(set([x for x, y in mapping]) & set([x for x, y in mapping2])) == 2 and
                        len(set([y for x, y in mapping]) & set([y for x, y in mapping2])) == 2):
                        new_candidate = mapping | mapping2
                        if new_candidate in new_candidates:
                            candidate_scores[new_candidate] = max(score + score2, candidate_scores[new_candidate])
                        else:
                            new_candidates.add(new_candidate)
                            candidate_scores[new_candidate] = score + score2
                        candidates_with_children.add((mapping, score))

            candidates[-1] = candidates[-1] - candidates_with_children
            logging.debug("terminal candidates: %d", len(candidates[-1]))
            new_candidates = set([(x, candidate_scores[x]) for x in new_candidates])

            if beam_width is not None:
                ''' Prune the current batch of candidates '''
                new_candidates = list(new_candidates)
                new_candidates.sort(key=lambda x: x[1], reverse=True)
                new_candidates = set(new_candidates[:beam_width])

            candidates.append(new_candidates)

        all_candidates = list(reduce(lambda x, y: y.union(x), candidates, set()))

        ''' Remove duplicate target sets '''
        seen_targets = defaultdict(set)
        for candidate in all_candidates:
            seen_targets[frozenset([x for x, y in candidate[0]])].add(candidate)

        filtered_candidates = []
        for target_candidate, base_candidates in seen_targets.iteritems():
            filtered_candidates.append(max(base_candidates, key=lambda x:x[1]))

        filtered_candidates.sort(key=lambda x: x[1], reverse=True)

        return filtered_candidates, sorted(candidate_relations.items(), key=lambda x: x[1], reverse=True)


    def analogy_from_concept(self, source_concept, logging_interval=None,
                             num_candidates=100,
                             beam_width=None):
        '''
        Finds an analogy for a single concept. The algorithm simply
        uses the CrossBridge.analogy method to find analogies for
        neighbors of source_concept.

        (This procedure is hacky because not all of the neighboring
        concepts are important for the analogy)
        '''

        # Get the concepts that are a part of the chosen concept
        related_concepts = set(self.graph.get_neighbors(source_concept))
        related_concepts.add(source_concept)

        logging.debug("Concept analogy: %r", source_concept)
        logging.debug("looking up: %r", related_concepts)
        # Find an analogy with these concepts
        return self.analogy(related_concepts,
                            logging_interval=logging_interval, num_candidates=num_candidates,
                            beam_width=beam_width)


def cnet_graph_from_tensor(cnet, delete_reltypes=None,
                           assertion_weight_threshold=0):
    '''
    This is a sort of hacky way to transform the typical
    ConceptNet concept / feature matrix into a SemanticNetwork.

    assertion_weight_threshold - only assertions with scores greater than the threshold are included in the graph.
    delete_reltypes - a list of relationship types that shouldn't be included in the graph
    '''

    concepts = set(cnet.dim_keys(0))

    cnet_graph = SemanticNetwork()
    for (concept1, (typ, r, concept2)), score in cnet.iteritems():
        if (score < assertion_weight_threshold or
            (delete_reltypes is not None and r in delete_reltypes)):
            continue

        if typ == 'left':
            # left feature
            c1 = concept2
            c2 = concept1
        else:
            # right feature
            c1 = concept1
            c2 = concept2

        cnet_graph.add_edge(c1, c2, r)

    return cnet_graph

def graph_from_triples(triples, omit_relations=None,
                       min_weight=0):
    '''
    Make a SemanticNetwork out of a sequence of triples.

    You can get such a sequence from
    csc.conceptnet4.analogyspace.conceptnet_triples.
    '''
    graph = SemanticNetwork()
    omit_relations = set(omit_relations or [])
    for (c1, rel, c2), val in triples:
        if val < min_weight or rel in omit_relations: continue
        graph.add_edge(c1, c2, rel)
    return graph

''' these are utility functions used by CrossBridge '''

def k_subset_iter(k, in_set):
    '''
    Returns an iterator over all possible subsets x of in_set where x
    contains exactly k items.
    '''
    def k_subset_iter_helper(k, in_set, base_set):
        if len(base_set) == k:
            yield base_set
        else:
            remaining_concepts = in_set.difference(base_set)
            if len(remaining_concepts) == 0:
                return

            cur_concept = remaining_concepts.pop()
            # Return all subsets without this element
            for subset in k_subset_iter_helper(k, remaining_concepts, base_set):
                yield subset

            # Return all subsets with this element
            new_set = copy.copy(base_set)
            new_set.add(cur_concept)
            for subset in k_subset_iter_helper(k, remaining_concepts, new_set):
                yield subset

    return k_subset_iter_helper(k, set(in_set), set())

def enumerate_orders(in_set):
    '''
    Returns an iterator over all permutations (orderings) of in_set
    '''
    def helper(in_set):
        if len(in_set) == 0:
            yield []
        else:
            for item in in_set:
                for remainder in helper(in_set - set([item])):
                    yield [item] + remainder

    for order in helper(in_set):
        yield tuple(order)

def k_edge_subgraphs(graph, min_subgraph_edges, num_subgraph_vertices, logging_interval=None):
    '''
    Bottom-up dynamic programming approach to finding all subgraphs of
    graph with exactly num_subgraph_vertices vertices and at least
    min_subgraph_edges edges.  This algorithm treats graph as if it
    were untyped and undirected, so there is at most 1 edge between 2
    vertices.

    graph - the graph from which subgraphs should be extracted
    num_subgraph_vertices - The number of vertices in the subgraphs
    min_subgraph_edges - The minimum number of edges in each returned subgraph. min_subgraph_edges must be at least num_subgraph_vertices-1 (so the graph is connected)

    Returns a dictionary mapping (num vertices, num edges) tuples to
    sets of vertices, expressed as frozensets.
    '''
    assert(min_subgraph_edges >= num_subgraph_vertices-1)

    found_subgraphs = defaultdict(set)

    # Base case: the set of 2-vertex subgraphs connected by 1 edge
    found_subgraphs[(2, 1)] = set(frozenset([v1, v2])
                                  for v1, v2, type in graph.edges())

    logging_counter = 0
    for num_nodes in xrange(3,num_subgraph_vertices + 1):
        logging.debug("Finding graphs with %d vertices...", num_nodes)
        for i in xrange(num_nodes - 2, min(min_subgraph_edges, (num_nodes - 2)*(num_nodes - 1)/2 + 1)):
            # Fill in found_subgraphs[num_nodes, i]
            logging.debug("Checking subgraphs with %d vertices and %d edges (total: %d)...",
                          num_nodes - 1, i, len(found_subgraphs[num_nodes - 1, i]))
            for soln_subgraph in found_subgraphs[num_nodes - 1, i]:
                #Compute edge counts for vertices adjacent to soln_subgraph
                vertex_counts = defaultdict(int)
                for soln_vertex in soln_subgraph:
                    for v in graph.get_neighbors(soln_vertex):
                        if v not in soln_subgraph:
                            vertex_counts[v] += 1

                for v, count in vertex_counts.iteritems():
                    found_subgraphs[(num_nodes, count + i)].add(soln_subgraph.union(frozenset([v])))

                if logging_interval is not None and logging_counter % logging_interval == 0:
                    logging.debug("Adding vertices to %r ...", soln_subgraph)
                logging_counter += 1

    return found_subgraphs
