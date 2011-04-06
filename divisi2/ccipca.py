import logging
import numpy as np
from divisi2.operators import projection
logger = logging.getLogger(__name__)
EPSILON = 1e-12

# The algorithm is based on:
#   Book Series - Lecture Notes in Computer Science
#   Book Title  - Intelligent Data Engineering and Automated Learning
#   Chapter Title  - A Fast Algorithm for Incremental Principal Component Analysis
#   First Page  - 876
#   Last Page  - 881
#   Copyright  - 2003
#   Author  - Juyang Weng
#   Author  - Yilu Zhang
#   Author  - Wey-Shiuan Hwang
#   DOI  - 
#   Link  - http://www.springerlink.com/content/cd8br967h808bw7h
#
# This is a purely dense, unlabeled version of ccipca.py, with the intent that
# it can be much faster as a result.

class CCIPCA(object):
    """A Candid Covariance-free Incremental Principal Component Analysis
    implementation"""

    def __init__(self, matrix, iteration=0, bootstrap=20, amnesia=3.0,
                 remembrance=100000, auto_baseline=True):
        """
        Construct an object that incrementally computes a CCIPCA, given a
        matrix that should hold the eigenvectors. If you want to make
        such a matrix from scratch, try the `CCIPCA.make` factory method.

        Parameters:

        - *matrix*: The matrix of eigenvectors to start with. (It can be all
          zeroes at the start.) Each column is an eigenvector, and rows
          represent the different entries an eigenvector can have.
        - *iteration*: the current time step.
        - *bootstrap*: The actual CCIPCA computation will begin after this time
          step. If you are starting from a zero matrix, this should be larger
          than the number of eigenvectors, so that the eigenvectors can be
          initialized properly.
        - amnesia: A parameter that weights the present more strongly than the
          past. amnesia=1 makes the present count the same as anything else.
        - remembrance: inputs that are more than this many steps old will begin
          to decay.
        - auto_baseline: if true, the CCIPCA will calculate and subtract out a
          moving average of the data. Otherwise, it will subtract out the
          constant vector in column 0.

        Construct a CCIPCA computation with k initial eigenvectors ev at
        iteration i, using simple averaging until the iteration given by
        bootstrap, afterward using CCIPCA given amnesic parameter amnesia and
        rememberance parameter remembrance.
        
        """
        self.matrix = matrix
        self.iteration = iteration
        self.bootstrap = bootstrap
        self.amnesia = amnesia
        self.remembrance = remembrance
        self.auto_baseline = auto_baseline

    @property
    def shape(self):
        return self.matrix.shape

    def zero_column(self):
        """
        Get a vector shaped like a column of the CCIPCA matrix, all of whose
        entries are zero.
        """
        return np.zeros((self.shape[0],), self.matrix.dtype)

    def get_weighted_eigenvector(self, index):
        """
        Get the weighted eigenvector with a specified index. "Weighted" means
        that its magnitude will correspond to its eigenvalue.

        Real eigenvectors start counting at 1. The 0th eigenvector represents
        the moving average of the input data.
        """
        return self.matrix[:,index]

    def get_unit_eigenvector(self, index):
        """
        Get the eigenvector with a specified index, as a unit vector.

        Real eigenvectors start counting at 1. The 0th eigenvector represents
        the moving average of the input data.
        """
        eig = self.get_weighted_eigenvector(index)
        return eig / (np.linalg.norm(eig) + EPSILON)

    def set_eigenvector(self, index, vec):
        """
        Sets eigenvector number `index` to the specified vector.
        """
        self.matrix[:,index] = vec

    def eigenvectors(self):
        return self.matrix / self.eigenvalues()

    def get_eigenvalue(self, index):
        return np.linalg.norm(self.get_weighted_eigenvector(index))
    
    def eigenvalues(self):
        return np.sqrt(np.sum(self.matrix * self.matrix, axis=0))

    def compute_attractor(self, index, vec):
        """
        Compute the attractor vector for the eigenvector with index
        `index` with the new vector `vec`: the projection of the
        eigenvector onto `vec`.
        """
        if index == 0:
            # special case for the mean vector
            return vec
        eigvec = self.get_unit_eigenvector(index)
        return projection(vec, eigvec)

    def eigenvector_loading(self, index, vec):
        """
        Returns the "loading" (the magnitude of the projection) of `vec` onto
        the eigenvector with index `index`. If `vec` forms an obtuse angle
        with the eigenvector, the loading will be negative.
        """
        #if index == 0:
        #    # handle the mean vector case
        #    return self.get_eigenvalue(0)
        #else:
        return np.dot(self.get_unit_eigenvector(index).conj(), vec)
    
    def eigenvector_projection(self, index, vec):
        # Do we actually need this?
        return self.get_unit_eigenvector(index) * self.eigenvector_loading(index, vec)
    
    def eigenvector_residue(self, index, vec):
        """
        Projects `vec` onto the eigenvector with index `index`. Returns the
        projection as a multiple of the unit eigenvector, and the remaining
        component that is orthogonal to the eigenvector.
        """
        loading = self.eigenvector_loading(index, vec)
        orth = vec - (loading * self.get_unit_eigenvector(index))
        if index > 0:
            assert np.abs(np.dot(orth.conj(), self.get_unit_eigenvector(index))) < 0.001
        return loading, orth

    def update_eigenvector(self, index, vec):
        """
        Performs the learning step of CCIPCA to update an eigenvector toward
        an input vector. Returns the magnitude of the eigenvector component,
        and the residue vector that is orthogonal to the eigenvector.
        """
        if self.iteration < index:
            # there aren't enough eigenvectors yet
            return 0.0, self.zero_column()

        if self.iteration == index:
            # create a new eigenvector
            self.set_eigenvector(index, vec)
            return np.linalg.norm(vec), self.zero_column()

        n = min(self.iteration, self.remembrance)
        if n < self.bootstrap:
            old_weight = float(n-1) / n
            new_weight = 1.0/n

        else:
            L = self.amnesia
            old_weight = float(n-L) / n
            new_weight = float(L)/n
        
        attractor = self.compute_attractor(index, vec)
        new_eig = ((self.get_weighted_eigenvector(index) * old_weight) +
                   (attractor * new_weight))
        self.set_eigenvector(index, new_eig)
        return self.eigenvector_residue(index, vec)

    def sort_vectors(self):
        """
        Sorts the eigenvector table in decreasing order, in place,
        keeping the special eigenvector 0 first. Returns the mapping
        from old to new eigenvectors.
        """
        eigs = self.eigenvalues()
        
        # keep eigenvector 0 in front (eigs was a new list)
        eigs[0] = np.inf
        sort_order = np.asarray(np.argsort(-eigs))

        self.matrix[:] = self.matrix[:,sort_order]
        return sort_order

    def learn_vector(self, vec):
        """
        Updates the eigenvectors to account for a new vector. Returns the
        amount of error (the magnitude of the vector outside the space of the
        eigenvectors).
        """
        current_vec = vec.copy()

        self.iteration += 1
        magnitudes = np.zeros((self.shape[1],), self.matrix.dtype)
        for index in xrange(min(self.shape[1], self.iteration+1)):
            mag, new_vec = self.update_eigenvector(index, current_vec)
            current_vec = new_vec
            magnitudes[index] = mag

        return np.linalg.norm(new_vec)
        sort_order = self.sort_vectors()
        magnitudes = magnitudes[sort_order]
        return magnitudes

    def project_vector(self, vec):
        """
        Projects `vec` onto each eigenvector in succession. Returns
        the magnitude of each eigenvector.
        """
        current_vec = vec.copy()
        magnitudes = np.zeros((self.shape[1],), self.matrix.dtype)
        for index in xrange(min(self.shape[1], self.iteration+1)):
            mag, new_vec = self.eigenvector_residue(index, current_vec)
            current_vec = new_vec
            magnitudes[index] = mag

        return magnitudes

    def reconstruct(self, weights):
        sum = self.zero_column()
        for index, w in enumerate(weights):
            sum += self.get_weighted_eigenvector(index) * w
        return sum

    def smooth(self, vec, k_max=None):
        mags = self.project_vector(vec)
        if k_max is not None:
            mags = mags[:k_max]
        
        return self.reconstruct(mags)
    
    def train_matrix(self, matrix):
        for col in xrange(matrix.shape[1]):
            print col, '/', matrix.shape[1]
            self.learn_vector(matrix[:,col])

