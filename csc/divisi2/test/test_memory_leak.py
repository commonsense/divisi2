from csc.conceptnet.analogyspace2 import *
def test_memory_leak():
    for w in xrange(20, 40):
        print w
        en = build_matrix('en', identity_weight=w)

        U, S, V = en.normalize_all().svd(k=150)
        #rec = divisi2.reconstruct(U, S, V)
        #del en

        #score = rec.evaluate_assertions('usertest_data.pickle')[-1]
        #del rec
        #print w, score
        #print >> out, w, score

if __name__ == '__main__': test_memory_leak()
