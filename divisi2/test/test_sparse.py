import divisi2
from divisi2.sparse import *
from divisi2.dense import *
from nose.tools import *
import numpy as np

mat1 = SparseMatrix.from_named_entries([
    (2, "apple", "red"),
    (2, "orange", "orange"),
    (1, "apple", "green"),
    (1, "celery", "green"),
])

mat2 = SparseMatrix.from_named_entries([
    (1, "apple", "fruit"),
    (1, "orange", "fruit"),
    (1, "orange", "orange"),
    (2, "celery", "vegetable"),
])

def test_vector_entries():
    assert mat1[0].keys() == [0,2]
    assert mat1[0].zero_entries() == [1]

def test_matrix_identities():
    "Test that various operations cancel out to give the same matrix or vector."
    for mat in (mat1, mat2):
        assert mat == mat.copy()
        assert mat == mat.match_labels(SparseMatrix.from_named_entries(mat.named_entries()))[0]
        assert mat == mat.match_labels(SparseMatrix.from_named_lists(*mat.named_lists()))[0]
        assert mat.unlabeled() == SparseMatrix.from_entries(mat.entries())
        assert mat.unlabeled() == SparseMatrix.from_lists(*mat.lists())
        assert mat.unlabeled() == SparseMatrix.from_lists(*mat.find())
        assert mat == mat*1
        assert mat == mat/1
        assert mat == 1*mat
        assert mat != mat.T
        assert mat == mat.T.T
        assert mat == mat.to_dense().to_sparse()

def test_vector_identities():
    """
    Test that various operations cancel out to give the same matrix or vector.
    """
    for vec in (mat1[0], mat1[:,0], mat2[0], mat2[:,0]):
        assert vec == vec.copy()
        assert vec == vec.match_labels(SparseVector.from_named_entries(vec.named_entries()))[0]
        assert vec == vec.match_labels(SparseVector.from_named_lists(*vec.named_lists()))[0]
        assert vec == vec*1
        assert vec == vec/1
        assert vec == 1*vec
        assert vec == vec.T
        assert vec == vec.to_dense().to_sparse()

def test_ellipsis():
	assert mat1[..., 0] == mat1[:, 0]
	assert mat1[0, ...] == mat1[0]

def test_sparse_vs_dense():
    assert np.allclose(mat1.T.dot(mat2).to_dense(), mat1.T.to_dense().dot(mat2))
    assert np.allclose(mat1.T.dot(mat2).to_dense(), mat1.T.dot(mat2.to_dense()))

def test_overlap_add():
    """
    Ensure that duplicated indices add their values.
    """
    mat3 = SparseMatrix.from_named_entries([
        (2, "apple", "fruit"),
        (1, "celery", "vegetable"),
        (1, "apple", "fruit")
    ])
    assert mat3.entry_named('apple', 'fruit') == 3

def test_iadd():
    mat3 = mat1.copy()
    mat3 += mat1
    assert mat3 == mat1*2

@raises(divisi2.LabelError)
def test_unaligned_iadd():
    mat3 = mat1.copy()
    mat3 += mat2

def test_subtract():
    """
    Test subtraction that isn't covered by the doctests.
    """
    sub1 = mat1-mat2
    assert sub1.entry_named('orange', 'orange') == 1
    assert sub1.entry_named('apple', 'red') == 2
    assert sub1.entry_named('apple', 'fruit') == -1

    mat3, mat4 = mat1.match_labels(mat2)
    mat3 -= mat4
    assert sub1 == mat3

def test_inplace_mul_div():
    mat3 = mat1.copy()
    mat3 *= -0.1
    assert mat3 == mat1.cmul(-0.1)
    mat3 /= -0.1
    assert mat3 == mat1

@raises(ZeroDivisionError)
def test_silly_division():
    1.0/mat1

@raises(ValueError)
def test_ambiguous_multiply():
    mat1 * mat2

@raises(NotImplementedError)
def test_abstract():
    AbstractSparseArray(mat1)

def test_normalize_tfidf():
    m = divisi2.SparseMatrix([[1,-1],[0,1]])
    tfidf = m.normalize_tfidf().to_dense()
    assert np.allclose(np.exp(tfidf*2), divisi2.DenseMatrix([[1,1],[1,2]]))

def test_empty_entries():
    '''Empty entry lists should yield empty matrices.'''
    m = SparseMatrix.from_named_entries(())
    eq_(m.shape, (0, 0))

def test_empty_square_entries():
    '''Empty entry lists should yield empty square matrices.'''
    m = SparseMatrix.square_from_named_entries(())
    eq_(m.shape, (0, 0))

def test_category():
    vec = category('cat', 'dog')
    assert vec.entry_named('cat') == 1
    assert vec.entry_named('dog') == 1
    assert len(vec) == 2

    vec2 = category(happy=1, sad=-1)
    assert vec2.entry_named('happy') == 1
    assert vec2.entry_named('sad') == -1
    assert len(vec2) == 2
