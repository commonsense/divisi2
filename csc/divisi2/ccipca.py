import logging
import numpy as np
logger = logging.getLogger(__name__)

from csc.divisi2.ordered_set import OrderedSet
from csc.divisi2.recycling_set import RecyclingSet

class CCIPCA(object):
    """A Candid Covariance-free Incremental Principal Component Analysis
    implementation"""

    def __init__(self, matrix, i=0, bootstrap=20, amnesia=3.0, remembrance=100000.0, auto_baseline=True):
        """
        TODO: fix docs

        Construct a CCIPCA computation with k initial eigenvectors ev at
        iteration i, using simple averaging until the iteration given by
        bootstrap, afterward using CCIPCA given amnesic parameter amnesia,
        rememberance parameter remembrance, and a weight vector and subspace
        criteria for simultaneous vector presentation
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

    def zerovec(self):
        """
        Get an appropriately-shaped column of all zeros.
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

    def get_eigenvector(self, index):
        """
        Get the eigenvector with a specified index, as a unit vector.

        Real eigenvectors start counting at 1. The 0th eigenvector represents
        the moving average of the input data.
        """
        return self.get_weighted_eigenvector(index).hat()

    def compute_attractor(self, index, vec):
        """
        Compute the attractor vector for the eigenvector with index `index`
        with the new vector `vec`.
        """
        if index == 0:
            # special case for the mean vector
            return vec
        eigvec = self.get_eigenvector(index)
        return self.projection(self, eigvec, vec)

    def projection_onto(self, v, u):
        """
        Compute the projection of `v` onto `u`, scaled by the magnitude of `u`.
        """
        return u * dot(u, v)
