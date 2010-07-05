from csc import divisi2
from csc.divisi2.dense import *
import numpy as np
mat1 = DenseMatrix([[0, 1], [2, 3]], ['A', 'B'], ['C', 'D'])

def test_arithmetic():
    assert np.allclose(mat1+mat1, 2*mat1)

def test_vector_slice():
    assert isinstance(mat1[0], DenseVector)
    assert isinstance(mat1[0,:], DenseVector)
    assert isinstance(mat1[:,0], DenseVector)

def test_normalize():
    assert mat1.normalize_rows().equals( mat1.to_sparse().normalize_rows().to_dense())
    assert mat1.normalize_cols().equals( mat1.to_sparse().normalize_cols().to_dense())
    assert mat1.normalize_all().equals( mat1.to_sparse().normalize_all().to_dense())

def test_unlabeled_convert():
    unlabeled = DenseMatrix(np.asarray(mat1))
    assert np.allclose(unlabeled, unlabeled.to_sparse().to_dense())

def test_sparse_by_dense():
    vec = divisi2.SparseVector.from_dict(dict(A=-1, B=1))
    mul = divisi2.matrixmultiply(vec, mat1)
    mul2 = divisi2.matrixmultiply(mat1.T, vec)
    assert mul.equals(mul2)
    assert np.allclose(mul, np.array([2, 2]))
