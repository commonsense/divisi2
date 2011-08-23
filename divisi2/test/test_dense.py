import divisi2
from divisi2.dense import *
import numpy as np
from nose.tools import eq_
mat1 = DenseMatrix([[0, 1], [2, 3]], ['A', 'B'], ['C', 'D'])
mat2 = DenseMatrix([[0, 1], [2, 3]], ['A', 'B'], None)
mat3 = DenseMatrix([[0, 1], [2, 5], [0, 1], [2, 5]], ['A', 'B', 'C', 'D'], None)
mat4 = DenseMatrix([[0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, 0.5]], ['A', 'B', 'C', 'D'], None)

def test_arithmetic():
    assert np.allclose(mat1+mat1, 2*mat1)

def test_vector_slice():
    row0 = DenseVector([0, 1], ['C', 'D'])
    col0 = DenseVector([0, 2], ['A', 'B'])
    assert mat1[0].equals(row0)
    assert mat1[0,:].equals(row0)
    assert mat1[:,0].equals(col0)
    assert mat1.get_row(0).equals(row0)
    assert mat1.get_col(0).equals(col0)
    
def test_equality():
    same = DenseMatrix([[0, 1], [2, 3]], ['A', 'B'], ['C', 'D'])
    different_data = DenseMatrix([[0, 1], [2, 4]], ['A', 'B'], ['C', 'D'])
    different_row_labels = DenseMatrix([[0, 1], [2, 3]], ['A', 'b'], ['C', 'D'])
    different_col_labels = DenseMatrix([[0, 1], [2, 3]], ['A', 'B'], ['C', 'd'])
    not_dense_matrix = np.asarray(mat1)
    assert mat1.equals(same)
    assert not mat1.equals(different_data)
    assert not mat1.equals(different_row_labels)
    assert not mat1.equals(different_col_labels)
    assert not mat1.equals(not_dense_matrix)

def test_normalize():
    assert mat1.normalize_rows().equals( mat1.to_sparse().normalize_rows().to_dense())
    assert mat1.normalize_cols().equals( mat1.to_sparse().normalize_cols().to_dense())
    assert mat1.normalize_all().equals( mat1.to_sparse().normalize_all().to_dense())

def test_degenerate_normalize():
    assert (mat1*0).normalize_all(offset=0.001).equals( mat1*0 )

def test_unlabeled_convert():
    unlabeled = DenseMatrix(np.asarray(mat1))
    assert np.allclose(unlabeled, unlabeled.to_sparse().to_dense())

def test_mean_center():
    centered, row_means, col_means, total_mean = mat3.mean_center()
    assert centered.equals(mat4)
    rec = centered + row_means[:,np.newaxis] + col_means + total_mean
    assert rec.equals(mat3)

def test_lookups():
    assert np.all(mat1.row_named('A') == mat1[0])
    assert np.all(mat1.col_named('D') == mat1[:,1])
    assert np.all(mat2.row_named('A') == mat2[0])
    assert np.all(mat2.col_named(1) == mat2[:,1])
    assert mat1.entry_named('B', 'C') == mat1[1,0]
    assert mat2.entry_named('B', 0) == mat2[1,0]

def test_sparse_by_dense():
    vec = divisi2.SparseVector.from_dict(dict(A=-1, B=1))
    mul = divisi2.matrixmultiply(vec, mat1)
    mul2 = divisi2.matrixmultiply(mat1.T, vec)
    assert mul.equals(mul2)
    assert np.allclose(mul, np.array([2, 2]))

def test_top_items():
    vec = DenseVector([0, 1, 2, 3], ['A', 'B', 'a', 'b'])
    eq_(vec.top_items(2), [('b', 3.0), ('a', 2.0)])
    eq_(vec.top_items(2, lambda label: label == label.upper()),
        [('B', 1.0), ('A', 0.0)])
    eq_(vec.top_items(2, lambda label: False), [])
