from divisi2.sparse import SparseMatrix
import numpy as np

def blend_factor(mat):
    """
    Use the blending heuristic to suggest an appropriate weight for combining
    this matrix with others.
    """
    U, S, V = mat.svd(1)
    return 1.0/S[0]

def blend(mats, factors=None, symmetric=False, post_weights=None):
    """
    Combine multiple labeled matrices into one, with weighted data from
    all the matrices.

    mats: a list of matrices to blend.
    factors: List of scaling factor for each matrix.
      If None, the reciprocal of the first singular value is used.
    post_weights: List of weights to apply to each scaled matrix.
      You can use this to, for example, say that one matrix is twice as
      important as another. If None, no post-weighting is performed.
    symmetric: Use square_from_named_lists.
    """
    assert len(mats) > 0
    if len(mats) == 1:
        if factors is None: return mats[0]
        else: return mats[0] * factors[0]
    
    b_values = []
    b_row_labels = []
    b_col_labels = []
    
    if factors is None:
        factors = [blend_factor(mat) for mat in mats]

    if post_weights is not None:
        factors = [factor*post_weight for factor, post_weight in zip(factors, post_weights)]
    
    for mat, factor in zip(mats, factors):
        # FIXME: using bare find(), multiplying in numpy form, and
        # translating the labels manually would be a bit faster
        values, row_labels, col_labels = mat.named_lists()
        b_values.extend([v*factor for v in values])
        b_row_labels.extend(row_labels)
        b_col_labels.extend(col_labels)
    
    if symmetric:
        return SparseMatrix.square_from_named_lists(b_values, b_row_labels, b_col_labels)
    else:
        return SparseMatrix.from_named_lists(b_values, b_row_labels, b_col_labels)

def blend_svd(mats, factors=None, k=50):
    '''
    Special optimized version of blend for doing just an SVD.

    Like matrix.svd, returns a triple of:

    - U as a dense labeled matrix
    - S, a dense vector representing the diagonal of Sigma
    - V as a dense labeled matrix

    '''
    
    if factors is None:
        factors = [blend_factor(mat) for mat in mats]

    # Align matrices.
    # FIXME: only works for fully labeleed matrices right now.
    # TODO: could micro-optimize by using the first ordered set's indices.
    from csc_utils.ordered_set import OrderedSet
    row_labels, row_mappings = OrderedSet(), []
    for mat in mats:
        row_mappings.append(np.array([row_labels.add(item) for item in mat.row_labels], dtype=np.uint64))
    col_labels, col_mappings = OrderedSet(), []
    for mat in mats:
        col_mappings.append(np.array([col_labels.add(item) for item in mat.col_labels], dtype=np.uint64))

    # Elide zero row tests, etc.

    from divisi2._svdlib import svd_sum
    from divisi2 import DenseMatrix
    Ut, S, Vt = svd_sum(mats, k, factors, row_mappings, col_mappings)
    U = DenseMatrix(Ut.T, row_labels, None)
    V = DenseMatrix(Vt.T, col_labels, None)
    return U, S, V

