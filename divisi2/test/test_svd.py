import divisi2
from divisi2.operators import *
from divisi2.sparse import *
from divisi2.dense import *
from nose.tools import *
from nose.plugins.attrib import attr

entries = [
    (2, "apple", "red"),
    (2, "orange", "orange"),
    (1, "apple", "green"),
    (1, "celery", "green"),
    (-1, "apple", "orange"),
    (-1, "banana", "orange")
]
mat_4x3 = SparseMatrix.from_named_entries(entries)

# A second matrix, differing only in features, to test blending.
second_mat_4x3 = SparseMatrix.from_named_entries([
    (2, "apple", "Red"),
    (2, "orange", "Orange"),
    (1, "apple", "Green"),
    (1, "celery", "Green"),
    (-1, "apple", "Orange"),
    (-1, "banana", "Orange")
])

# A third matrix, differing only in concepts, to test blending.
third_mat_4x3 = SparseMatrix.from_named_entries([
    (2, "Apple", "red"),
    (2, "Orange", "orange"),
    (1, "Apple", "green"),
    (1, "Celery", "green"),
    (-1, "Apple", "orange"),
    (-1, "Banana", "orange")
])

def assert_singular_triple(mat, u, s, v):
    assert np.allclose(
        u.multiply(s),
        mat.dot(v))
    assert np.allclose(
        v.multiply(s),
        mat.T.dot(u))

def test_full_svd():
    U_sparse, S_sparse, V_sparse = mat_4x3.svd(3)
    rec = dot(U_sparse * S_sparse, V_sparse.T)
    assert rec.same_labels_as(mat_4x3)
    assert np.allclose(mat_4x3.to_dense(), rec)
    for i in range(3):
        assert_singular_triple(mat_4x3, U_sparse[:,i], S_sparse[i], V_sparse[:,i])

def test_truncated_svd():
    # FIXME: this doesn't actually test against NumPy's SVD now that
    # dense SVDs use svdlibc too.
    U_sparse, S_sparse, V_sparse = mat_4x3.svd(2)
    U_dense, S_dense, V_dense = mat_4x3.to_dense().svd(2)
    rec_sparse = dot(U_sparse * S_sparse, V_sparse.T)
    rec_dense = dot(U_dense * S_dense, V_dense.T)
    assert np.allclose(rec_sparse, rec_dense)

def test_zero_row():
    matcopy = mat_4x3.copy()
    matcopy[2,2] = 0
    U, S, V = matcopy.svd(2)
    rec = dot(U*S, V.T)
    assert np.allclose(matcopy.to_dense(), rec)

def test_k_too_large():
    U, S, V = mat_4x3.svd(50)
    assert len(S) == 3

def test_blend():
    from divisi2.blending import blend, blend_svd
    Uref, Sref, Vref = blend([mat_4x3, second_mat_4x3]).svd(k=2)
    U, S, V = blend_svd([mat_4x3, second_mat_4x3], k=2)
    rec_ref = dot(Uref * Sref, Vref.T)
    rec_opt = dot(U * S, V.T)
    assert np.allclose(rec_ref, rec_opt)

def test_blend_2():
    from divisi2.blending import blend, blend_svd
    Uref, Sref, Vref = blend([mat_4x3, third_mat_4x3]).svd(k=2)
    U, S, V = blend_svd([mat_4x3, third_mat_4x3], k=2)
    rec_ref = dot(Uref * Sref, Vref.T)
    rec_opt = dot(U * S, V.T)
    assert np.allclose(rec_ref, rec_opt)
    
@attr('slow')
def test_cnet_blend():
    from divisi2.blending import blend, blend_svd
    matrix = divisi2.network.conceptnet_matrix('en')
    isa = divisi2.network.filter_by_relation(matrix, 'IsA').squish().normalize_all()
    atloc = divisi2.network.filter_by_relation(matrix, 'AtLocation').squish().normalize_all()
    Uref, Sref, Vref = blend([isa, atloc]).svd(k=3)
    U, S, V = blend_svd([isa, atloc], k=3)
    rec_ref = divisi2.reconstruct(Uref, Sref, Vref)
    rec_opt = divisi2.reconstruct(U, S, V)

    # Check a random sampling of the items.
    import random
    for row in random.sample(rec_ref.row_labels, 50):
        for col in random.sample(rec_ref.col_labels, 50):
            assert np.allclose(rec_ref.entry_named(row, col),
                               rec_opt.entry_named(row, col))
    
