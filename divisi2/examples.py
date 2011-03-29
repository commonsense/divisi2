from divisi2.network import conceptnet_matrix, conceptnet_assoc
from divisi2.reconstructed import reconstruct, reconstruct_similarity, reconstruct_activation
import divisi2
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
    U, S, _ = assoc.normalize_all().svd(k=100)
    return reconstruct_activation(U, S, post_normalize=True)

def spreading_activation_weighted():
    assoc = conceptnet_assoc('en')
    U, S, _ = assoc.normalize_all().svd(k=100)
    return reconstruct_activation(U, S, post_normalize=False)

