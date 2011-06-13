class LabeledVectorMixin(object):
    def label(self, idx):
        "Get the label of the entry at a given numeric index."
        if self.labels is None: return idx
        else: return self.labels[idx]

    def index(self, label):
        "Get the numeric index for the entry with a given label."
        return self.labels.index(label)

    def entry_named(self, label):
        "Get the value with a given label."
        return self[self.index(label)]
    
    def all_labels(self):
        """
        Returns a one-element list containing `self.labels`.
        
        This is useful when mapping lists of indices one-to-one with lists
        of labels.
        """
        return [self.labels]

    def same_labels_as(self, other):
        """
        Does this vector have the same coordinate labels as another vector?
        (If so, they can be added very efficiently.)
        """
        return (self.shape[0] == other.shape[0] and
                self.labels == other.labels)
    
    def unlabeled(self):
        """
        Get a copy of this matrix with no labels.
        """
        return self.__class__(self, None)

class LabeledMatrixMixin(object):
    def row_label(self, idx):
        "Get the label of the row at a given numeric index."
        if self.row_labels is None: return idx
        else: return self.row_labels[idx]

    def col_label(self, idx):
        "Get the label of the column at a given numeric index."
        if self.col_labels is None: return idx
        else: return self.col_labels[idx]

    def row_index(self, label):
        "Get the numeric index for the row with a given label."
        if self.row_labels is None: return label
        return self.row_labels.index(label)

    def col_index(self, label):
        "Get the numeric index for the column with a given label."
        if self.col_labels is None: return label
        return self.col_labels.index(label)

    def row_named(self, label):
        "Get the row with a given label as a vector."
        return self[self.row_index(label)]

    def col_named(self, label):
        "Get the column with a given label as a vector."
        return self[:,self.col_index(label)]
    
    def entry_named(self, row_label, col_label):
        "Get the entry with a given row and column label."
        return self[self.row_index(row_label), self.col_index(col_label)]

    def set_entry_named(self, row_label, col_label, value):
        "Set a new value in the entry with a given row and column label."
        self[self.row_index(row_label), self.col_index(col_label)] = value
    set = set_entry_named

    def same_row_labels_as(self, other):
        return (self.shape[0] == other.shape[0] and
                self.row_labels == getattr(other, 'row_labels', None))
    
    def same_col_labels_as(self, other):
        return (self.shape[1] == other.shape[1] and
                self.col_labels == getattr(other, 'col_labels', None))
    
    def same_labels_as(self, other):
        """
        Does this matrix have the same coordinate labels as another matrix?
        (If so, they can be added very efficiently.)
        """
        return self.same_row_labels_as(other) and self.same_col_labels_as(other)
    
    def all_labels(self):
        """
        Returns the two-element list of `[self.row_labels, self.col_labels]`.

        This is useful when mapping lists of indices one-to-one with lists
        of labels.
        """
        return [self.row_labels, self.col_labels]

    def unlabeled(self):
        """
        Get a copy of this matrix with no labels.
        """
        return self.__class__(self, None, None)

def format_label(label):
    """
    This allows special handling for ConceptNet features expressed as tuples.
    """
    if isinstance(label, tuple) and len(label) == 3:
        if label[0] == 'left':
            return r'%s\%s' % (label[2], label[1])
        elif label[0] == 'right':
            return r'%s/%s' % (label[1], label[2])
    return label
