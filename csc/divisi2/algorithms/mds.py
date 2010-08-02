import numpy as np
import random
from csc.divisi2.sparse import SparseMatrix, SparseVector
from csc.divisi2.dense import DenseMatrix, DenseVector
from csc.divisi2.operators import dot, multiply
from csc.divisi2.fileIO import save

def _getLandmarkPoints(size):
    numpoints = int(np.sqrt(size))
    return random.sample(xrange(size), numpoints)

def normalize(mat):
    if mat.ndim == 1:
        return mat.hat()
    else:
        return mat.normalize_rows()

def compute_distances(matrix1, matrix2):
    nmatrix1 = normalize(matrix1)
    nmatrix2 = normalize(matrix2)
    distances = np.arccos(np.maximum(-1, np.minimum(1, dot(nmatrix1, nmatrix2.T))))
    assert isinstance(distances, DenseMatrix)
    assert distances.same_row_labels_as(nmatrix1)
    return multiply(distances, distances)

def lmds(matrix, k=2):
    # Find Landmark points
    N = matrix.shape[0]
    landmarks = _getLandmarkPoints(N)
    num_landmarks = len(landmarks)
    
    landmark_matrix = matrix[landmarks]
    sqdistances = compute_distances(landmark_matrix, landmark_matrix)

    # Do normal MDS on landmark points
    means = np.mean(sqdistances, axis=1)      # this is called mu_n in the paper
    global_mean = np.mean(means)              # this is called mu in the paper

    # this is called B in the paper
    distances_balanced = -(sqdistances - means[np.newaxis,:] - means[:,np.newaxis] + global_mean)/2

    # find the eigenvectors and eigenvalues with our all-purpose hammer
    # for the paper, Lambda = lambda, Q = V
    Q, Lambda, _ = np.linalg.svd(distances_balanced)
    k = min(k, len(Lambda))
    mdsarray_sharp = Q[:,:k]                       # called L^sharp transpose in the paper
    mdsarray = multiply(mdsarray_sharp, np.sqrt(Lambda)[np.newaxis,:k])  # called L transpose in the paper

    # Make Triangulation Object
    return LMDSProjection(landmark_matrix, mdsarray_sharp, means)

class LMDSProjection(object):
    def __init__(self, landmarks, mdsarray_sharp, means):
        self.landmarks = landmarks
        self.mdsarray_sharp = mdsarray_sharp
        self.means = means
        self.N, self.k = self.mdsarray_sharp.shape
    def project(self, vector):
        # vector can in fact be a matrix of many vectors

        # Dimensions:
        # vector = (m x ndim) or possibly just (ndim),
        #   with ndim = K from the SVD
        # dist = (m x N)
        # means = (N)
        # mdsarray_sharp = (N x k)
        dist = (compute_distances(vector, self.landmarks) - self.means)/2
        return dot(dist, -self.mdsarray_sharp)

def aspace_mds():
    from csc.divisi2.network import analogyspace_matrix

    cnet = analogyspace_matrix('en')
    
    U, S, V = cnet.normalize_all().svd(k=100)
    concept_axes = U
    proj = mds(concept_axes)
    result = proj.project(concept_axes)
    save(result, '/tmp/mds.pickle')
    return result

if __name__ == '__main__':
    aspace_mds()
