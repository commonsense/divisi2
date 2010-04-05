from csc import divisi2
import cPickle as pickle

mat_4x3 = divisi2.make_sparse([
    (2, "apple", "red"),
    (2, "orange", "orange"),
    (1, "apple", "green"),
    (1, "celery", "green"),
    (-1, "apple", "orange"),
    (-1, "banana", "orange")
])

def pickle_bounce(obj):
    s = pickle.dumps(obj)
    objcopy = pickle.loads(s)
    return objcopy

def test_sparse_pickle():
    mat2 = divisi2.SparseMatrix.from_state(mat_4x3.to_state())
    assert mat2 == mat_4x3
    assert pickle_bounce(mat_4x3) == mat_4x3
    assert pickle_bounce(mat_4x3[0]) == mat_4x3[0]
    assert pickle_bounce(mat_4x3[:,0]) == mat_4x3[:,0]

def test_dense_pickle():
    dmat = mat_4x3.to_dense()
    assert pickle_bounce(dmat).equals(dmat)
    assert pickle_bounce(dmat[0]).equals(dmat[0])
    assert pickle_bounce(dmat[:,0]).equals(dmat[:,0])

