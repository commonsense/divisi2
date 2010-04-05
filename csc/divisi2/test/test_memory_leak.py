from csc.conceptnet.analogyspace2 import *
def test_memory_leak():
    en = build_matrix('en', identity_weight=1)
    for i in xrange(20):
        print i
        newen = en.squish()

if __name__ == '__main__': test_memory_leak()
