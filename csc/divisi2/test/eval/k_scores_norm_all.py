from csc.conceptnet.analogyspace2 import *
en = build_matrix('en', cutoff=5, identity_weight=0)

out = open('k_scores_norm_all.txt', 'w')
for k in xrange(5, 300, 5):
    U, S, V = en.normalize_all().svd(k=k)
    rec = divisi2.reconstruct(U, S, V)
    
    score = rec.evaluate_assertions('usertest_data.pickle')[-1]
    print k, score
    print >> out, k, score
out.close()

