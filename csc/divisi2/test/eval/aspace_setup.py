from csc.conceptnet.analogyspace2 import *
en = build_matrix('en')
U, S, V = en.normalize_all().svd()
rec = divisi2.reconstruct(U, S, V)
sim = divisi2.reconstruct_similarity(U, S)

