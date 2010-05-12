from csc.divisi2.network import conceptnet_matrix, conceptnet_assoc
from csc.divisi2.reconstructed import reconstruct, reconstruct_similarity, reconstruct_activation
import numpy as np

def analogyspace_predictions():
    cnet = conceptnet_matrix('en')
    U, S, V = cnet.normalize_all().svd(k=100)
    return reconstruct(U, S, V)

def analogyspace_similarity():
    cnet = conceptnet_matrix('en')
    U, S, V = cnet.normalize_all().svd(k=100)
    return reconstruct_similarity(U, S)

def spreading_activation():
    assoc = conceptnet_assoc('en')
    Q, S, _ = assoc.svd(k=100)
    L = np.sqrt(S)
    return reconstruct_activation(Q, L)
