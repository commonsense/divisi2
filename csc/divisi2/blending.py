from csc.divisi2.sparse import SparseMatrix

def blend_factor(mat):
    """
    Use the blending heuristic to suggest an appropriate weight for combining
    this matrix with others.
    """
    U, S, V = mat.svd(1)
    return 1.0/S[0]

def blend(mats, factors=None, symmetric=False):
    """
    Combine multiple labeled matrices into one, with weighted data from
    all the matrices.
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
    
    for mat, factor in zip(mats, factors):
        # FIXME: using bare find(), multiplying in numpy form, and
        # translating the labels manually would be a bit faster
        values, row_labels, col_labels = mat.named_lists()
        b_values.extend([v*factor for v in values])
        b_row_labels.extend(row_labels)
        b_col_labels.extend(col_labels)
    
    if symmetric:
        return SparseMatrix.square_from_named_lists(b_values, b_row_labels,
        b_col_labels)
    else:
        return SparseMatrix.from_named_lists(b_values, b_row_labels,
        b_col_labels)

