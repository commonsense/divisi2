from csc.conceptnet.analogyspace2 import *
en = build_matrix('en', cutoff=5, identity_weight=0)
U, S, V = en.normalize_rows().svd(k=100)
rec = divisi2.reconstruct(U, S, V)
print rec.evaluate_assertions('usertest_data.pickle')

