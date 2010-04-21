def blend_factor(mat):
    """
    Use the blending heuristic to suggest an appropriate weight for combining
    this matrix with others.
    """
    U, S, V = mat.svd(1)
    return 1.0/S[0]

def blend(mats, factors=None):
    """
    Combine multiple labeled matrices into one, with weighted data from
    all the matrices.
    """
    assert len(mats) > 0
    if factors is None:
        factors = [blend_factor(mat) for mat in mats]
    total = mats[0] * factors[0]
    for i in xrange(1, len(mats)):
        total = total + (mats[i] * factors[i])
    return total

