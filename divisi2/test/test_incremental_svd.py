from divisi2.sparse import SparseMatrix
from divisi2.reconstructed import ReconstructedMatrix
from divisi2.operators import dot
import numpy as np

mat_4x3 = SparseMatrix.from_named_entries([
    (2, "apple", "red"),
    (2, "orange", "orange"),
    (1, "apple", "green"),
    (1, "celery", "green"),
    (-1, "apple", "orange"),
    (-1, "banana", "orange")
])

def test_incremental_svd():
    U_sparse, S_sparse, V_sparse = mat_4x3.svd(2)
    rec = dot(U_sparse * S_sparse, V_sparse.T)
    rec2 = ReconstructedMatrix.make_random(mat_4x3.row_labels,
                                           mat_4x3.col_labels,
                                           2,
                                           learning_rate = 0.01)
    for iter in xrange(1000):
        for row in xrange(4):
            for col in xrange(3):
                rec2.hebbian_step(row, col, mat_4x3[row, col])
        print np.linalg.norm(rec2.to_dense() - rec)
    dense = rec2.to_dense()
    assert rec.same_labels_as(rec2)
    assert np.linalg.norm(rec2.to_dense() - rec) < 0.1
