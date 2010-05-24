from csc import divisi2
from csc.divisi2.operators import *
from csc.divisi2.sparse import *
from csc.divisi2.dense import *
from nose.tools import *

mat_4x3 = SparseMatrix.from_named_entries([
    (2, "apple", "red"),
    (2, "orange", "orange"),
    (1, "apple", "green"),
    (1, "celery", "green"),
    (-1, "apple", "orange"),
    (-1, "banana", "orange")
])

# A second matrix, differing only in features, to test blending.
second_mat_4x3 = SparseMatrix.from_named_entries([
    (2, "apple", "Red"),
    (2, "orange", "Orange"),
    (1, "apple", "Green"),
    (1, "celery", "Green"),
    (-1, "apple", "Orange"),
    (-1, "banana", "Orange")
])


def test_full_svd():
    U_sparse, S_sparse, V_sparse = mat_4x3.svd(3)
    rec = dot(U_sparse * S_sparse, V_sparse.T)
    assert rec.same_labels_as(mat_4x3)
    assert np.allclose(mat_4x3.to_dense(), rec)

def test_truncated_svd():
    # FIXME: this doesn't actually test against NumPy's SVD now that
    # dense SVDs use svdlibc too.
    U_sparse, S_sparse, V_sparse = mat_4x3.svd(2)
    U_dense, S_dense, V_dense = mat_4x3.to_dense().svd(2)
    rec_sparse = dot(U_sparse * S_sparse, V_sparse.T)
    rec_dense = dot(U_dense * S_dense, V_dense.T)
    assert np.allclose(rec_sparse, rec_dense)

@raises(ValueError)
def test_zero_row():
    matcopy = mat_4x3.copy()
    matcopy[2,2] = 0
    matcopy.svd(3)

def test_k_too_large():
    U, S, V = mat_4x3.svd(50)
    assert len(S) == 3

def test_blend():
    U, S, V = blend_svd([mat_4x3, second_mat_4x3])
