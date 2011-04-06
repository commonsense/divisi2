import logging
import numpy as np
logger = logging.getLogger(__name__)

from csc.divisi2.ordered_set import OrderedSet, RecyclingSet
from csc.divisi2.dense import DenseMatrix, DenseVector
from csc.divisi2.operators import projection, dot

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

    def __init__(self, matrix, iteration=0, bootstrap=20, amnesia=3.0, remembrance=100000, auto_baseline=True):
        """
        Construct an object that incrementally computes a CCIPCA, given a
        matrix that should hold the eigenvectors. If you want to make
        such a matrix from scratch, try the `CCIPCA.make` factory method.

        Parameters:

        - *matrix*: The matrix of eigenvectors to start with. (It can be all
          zeroes at the start.) Each column is an eigenvector, and rows
          represent the different entries an eigenvector can have. Rows may
          have labels on them.
        - *iteration*: the current time step.
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
    
    @staticmethod
    def make(k, labels, amnesia=3.0, remembrance=100000):
        """
        Makes a k-dimensional CCIPCA for a set of labels (or a set of m
        unlabeled indices, if an integer m is given instead).

        Some of CCIPCA's fiddlier options are given reasonable defaults.
        """
        if isinstance(labels, int):
            # no actual labels, just standard indices
            m = labels
            labels = None
        else:
            if not isinstance(labels, OrderedSet):
                labels = OrderedSet(labels)
            m = len(labels)
        matrix = DenseMatrix(np.zeros((m, k)), labels)
        return CCIPCA(matrix, 0, k*2, amnesia, remembrance, True)

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
        return self.matrix.get_col(index)

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
        return self.matrix.col_norms()

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
        if index == 0:
            # handle the mean vector case
            return self.get_eigenvalue(0)
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

        #self.matrix[:] = self.matrix[:,sort_order]
        self.matrix = DenseMatrix(np.asarray(self.matrix)[:,sort_order], self.matrix.row_labels, None)
        return sort_order
    
    def match_labels(self, vec, touch=False):
        """
        Returns a new vector with the data of `vec` but aligned to the
        current labels.
        """
        result = self.zero_column()
        for key, value in vec.named_items():
            if touch:
                index = result.labels.add(key)
            else:
                index = result.labels.index(key, touch=False)
            result[index] = value
        return result

    def learn_vector(self, vec):
        """
        Updates the eigenvectors to account for a new vector. Returns
        the magnitudes of each eigenvector that would (approximately)
        reconstruct the given vector.
        """
        if vec.labels is not self.matrix.row_labels:
            current_vec = self.match_labels(vec, touch=True)
        else:
            current_vec = vec

        self.iteration += 1
        magnitudes = np.zeros((self.shape[1],))
        for index in xrange(min(self.shape[1], self.iteration+1)):
            mag, new_vec = self.update_eigenvector(index, current_vec)
            current_vec = new_vec
            magnitudes[index] = mag

        sort_order = self.sort_vectors()
        magnitudes = magnitudes[sort_order]
        return magnitudes

    def project_vector(self, vec):
        """
        Projects `vec` onto each eigenvector in succession. Returns
        the magnitude of each eigenvector.  (Like learn_vector, but
        doesn't change the state.)
        """
        if vec.labels is not self.matrix.row_labels:
            current_vec = self.match_labels(vec, touch=False)
        else:
            current_vec = vec
        magnitudes = np.zeros((self.shape[1],))
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
    
    def forget_row(self, slot, label):
        """
        Called by RecyclingSet when an index gets reused. Clears the
        old data out of the eigenvector table.
        """
        logger.debug("forgetting row %d" % slot)
        self.matrix[slot,:] = 0

    def train_matrix(self, matrix):
        for col in xrange(matrix.shape[1]):
            print col, '/', matrix.shape[1]
            self.learn_vector(matrix[:,col])

def for_profiling(A, n):
    c = CCIPCA.make(100, A.row_labels, amnesia=1.0)
    for col in xrange(n):
        c.learn_vector(A[:,col])


def evaluate_assertions(input_data, test_filename):
    """
    Evaluate the predictions that this matrix makes against a matrix of
    test data.
    """
    
    def order_compare(s1, s2):
        assert len(s1) == len(s2)
        score = 0.0
        total = 0
        for i in xrange(len(s1)):
            for j in xrange(i+1, len(s1)):
                if s1[i] < s1[j]:
                    if s2[i] < s2[j]: score += 1
                    elif s2[i] > s2[j]: score -= 1
                    total += 1
                elif s1[i] > s1[j]:
                    if s2[i] < s2[j]: score -= 1
                    elif s2[i] > s2[j]: score += 1
                    total += 1
        # move onto 0-1 scale
        score += (total-score)/2.0
        return (float(score) / total, score, total)
    
    from csc import divisi2
    import time
    testdata = divisi2.load(test_filename)
    values1 = []
    values2 = []
    row_labels = input_data.row_labels
    col_labels = input_data.col_labels
    c = CCIPCA.make(100, row_labels, amnesia=1.0)
    c.train_matrix(input_data)
    start_time = time.time()
    c.train_matrix(input_data)
    duration = time.time() - start_time
    print "Elapsed time:", duration
    print "Per entry:", duration/input_data.shape[1]

    for value, label1, label2 in testdata.named_entries():
        if label1 in row_labels and label2 in col_labels:
            smooth = c.smooth(input_data.column_named(label2))
            entry = smooth.entry_named(label1)
            values1.append(value)
            values2.append(entry)
    s1, s1s, s1t = order_compare(values1, values2)
    s2, s2s, s2t = order_compare(values1, values1)
    return s1s, s2s, s1/s2
