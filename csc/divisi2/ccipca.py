import logging
import numpy as np
logger = logging.getLogger(__name__)

from csc.divisi2.ordered_set import OrderedSet
from csc.divisi2.recycling_set import RecyclingSet

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

class CCIPCA(object):
    """A Candid Covariance-free Incremental Principal Component Analysis
    implementation"""

    def __init__(self, matrix, i=0, bootstrap=20, amnesia=3.0, remembrance=100000, auto_baseline=True):
        """
        Construct an object that incrementally computes a CCIPCA, given a
        matrix that should hold the eigenvectors. If you want to make
        such a matrix from scratch, try the `CCIPCA.make` factory method.

        Parameters:

        - *matrix*: The matrix of eigenvectors to start with. (It can be all
          zeroes at the start.) Each column is an eigenvector, and rows
          represent the different entries an eigenvector can have. Rows may
          have labels on them.
        - *i*: the current time step.
        - *bootstrap*: The actual CCIPCA computation will begin after this time
          step. If you are starting from a zero matrix, this should be larger
          than the number of eigenvectors, so that the eigenvectors can be
          initialized properly.
        - amnesia: A parameter that weights the present more strongly than the
          past.
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

        if isinstance(self.matrix.row_labels, RecyclingSet):
            self.matrix.row_labels.listen_for_drops(self.forget_row)

    @property
    def shape(self):
        return self.matrix.shape

    def zero_column(self):
        """
        Get a vector labeled like a column of the CCIPCA matrix, all of whose
        entries are zero.
        """
        return DenseVector(np.zeros((self.shape[0],)), self.matrix.row_labels)

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
        return self.get_weighted_eigenvector(index).hat()

    def set_eigenvector(self, index, vec):
        """
        Sets eigenvector number `index` to the specified vector.
        """
        self.matrix[:,index] = vec

    def eigenvectors(self):
        return self.matrix.normalize_cols()

    def get_eigenvalue(self, index):
        return np.linalg.norm(self.get_weighted_eigenvector(index))
    
    def eigenvalues(self):
        return np.sqrt(np.sum(self.matrix * self.matrix, mode=0))

    def compute_attractor(self, index, vec):
        """
        Compute the attractor vector for the eigenvector with index `index`
        with the new vector `vec`.
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
        if index == 0:
            # handle the mean vector case
            return self.eigenvalue(0)
        else:
            return dot(self.get_unit_eigenvector(index), vec)
    
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
        return loading, vec - (loading * self.get_unit_eigenvector(index))

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

        n = min(self._iteration, self._remembrance)
        if n < self._bootstrap:
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

    def learn(self, vector):
        current_v = vector[:]

