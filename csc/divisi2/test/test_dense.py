from csc import divisi2
from csc.divisi2.dense import *
import numpy as np
mat1 = DenseMatrix([[0, 1], [2, 3]], ['A', 'B'], ['C', 'D'])

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
    assert mat1.normalize_all().equals( mat1.to_sparse().normalize_all().to_dense())

def test_degenerate_normalize():
    assert (mat1*0).normalize_all().equals( mat1*0 )

def test_sparse_by_dense():
    vec = divisi2.SparseVector.from_dict(dict(A=-1, B=1))
    mul = divisi2.matrixmultiply(vec, mat1)
    mul2 = divisi2.matrixmultiply(mat1.T, vec)
    assert mul.equals(mul2)
    assert np.allclose(mul, np.array([2, 2]))
