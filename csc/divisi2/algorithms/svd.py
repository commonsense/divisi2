from csc.divisi2.dense import DenseMatrix
from csc.divisi2.reconstructed import ReconstructedMatrix
from csc.divisi2._svdlib import svd_llmat
from csc.divisi2 import operators

def svd(matrix, k=50):
    """
    Calculate the truncated singular value decomposition
    :math:`A = U * Sigma * V^T` using SVDLIBC.

    Returns a triple of:
    
    - U as a dense labeled matrix
    - S, a dense vector representing the diagonal of Sigma
    - V as a dense labeled matrix
    
    This matrix must not contain any empty rows or columns. If it does,
    use the .squish() method first.
    """
    assert matrix.ndim == 2
    if isinstance(matrix, DenseMatrix):
        # FIXME: we can probably feed this into SVDLIBC directly.
        matrix = matrix.to_sparse()
    if matrix.shape[1] >= matrix.shape[0] * 1.2:
        # transpose the matrix for speed
        V, S, U = matrix.T.svd(k)
        return U, S, V

    Ut, S, Vt = svd_llmat(matrix.llmatrix, k)
    U = DenseMatrix(Ut.T, matrix.row_labels, None)
    V = DenseMatrix(Vt.T, matrix.col_labels, None)

    return (U, S, V)


