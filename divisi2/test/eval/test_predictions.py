from csc import divisi2
from nose.tools import *
from nose.plugins.attrib import attr

@attr('slow')
def test_all_norm():
    matrix = divisi2.network.conceptnet_matrix('en')
    U, S, V = matrix.normalize_all().svd(k=100)
    rec = divisi2.reconstruct(U, S, V)
    correct, total, accuracy = rec.evaluate_assertions('data:eval/usertest_data.pickle')

    print "accuracy =", accuracy
    assert accuracy > 0.7
    return accuracy

if __name__ == '__main__':
    test_all_norm()
