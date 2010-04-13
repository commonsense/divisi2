from itertools import izip

class OrderedSet(object):
    """
    An OrderedSet acts very much like a list. There are two important
    differences:

    - Each item appears in the list only once.
    - You can look up an item's index in the list in constant time.
    """
    index_is_efficient = True

    __slots__ = ['items', 'indices', 'index', 'indexFor', '__contains__',
                 '__len__']

    def __init__(self, origitems=None):
        '''Initialize a new OrderedSet.'''
        self.items = []     # list of all keys
        self.indices = {}   # maps known keys to their indices in the list
        for item in origitems or []:
            assert not isinstance(item, OrderedSet)
            self.add(item)

        self._setup_quick_lookup_methods()

    def _setup_quick_lookup_methods(self):
        self.index = self.indices.__getitem__
        self.indexFor = self.index
        self.__contains__ = self.indices.__contains__
        self.__len__ = self.indices.__len__
    
    def __getitem__(self, index):
        if hasattr(index, '__index__') or isinstance(index, slice):
            result = self.items[index]
            if isinstance(result, list):
                return OrderedSet(result)
            else:
                return result
        elif isinstance(index, basestring):
            raise TypeError("Can't use a string as an OrderedSet index -- did you mean to use .index?")
        elif isinstance(index, type(None)):
            raise TypeError("Can't index an OrderedSet with None")
        else:
            # assume it's a fancy index list
            return OrderedSet([self.items[i] for i in index])

    def copy(self):
        """
        Efficiently make a copy of this OrderedSet.
        """
        newset = OrderedSet()
        newset.items = self.items[:]
        newset.indices = self.indices.copy()
        newset._setup_quick_lookup_methods()
        return newset

    def __repr__(self):
        if len(self) < 10:
            return u'OrderedSet(%r)' % [x for x in self.items if x is not None]
        else:
            return u'<OrderedSet of %d items like %s>' % (len(self), self[0])

    def __getstate__(self):
        return self.items
    def __setstate__(self, state):
        self.items = state
        self.indices = dict((item, index)
                            for index, item in enumerate(self.items)
                            if item is not None)
        self._setup_quick_lookup_methods()


    def add(self, key):
        """
        Add an item to the set (unless it's already there),
        returning its index.

        ``None`` is never an element of an OrderedSet.
        """

        if key in self.indices: return self.indices[key]
        n = len(self.items)
        self.items.append(key)
        if key is not None:
            self.indices[key] = n
        return n
    append = add

    def extend(self, lst):
        "Add a collection of new items to the set."
        for item in lst: self.add(item)
    __iadd__ = extend

    def merge(self, other):
        """
        Returns a new OrderedSet that merges this with another. The indices
        from this OrderedSet will remain the same, and this method will return
        a mapping of the new indices for the other OrderedSet.

        Returns a tuple of `merged`, which is the combined OrderedSet, and
        `indices`, a list the length of `other` giving the new index for each
        of its entries.
        
            >>> set1 = OrderedSet(['red', 'orange', 'yellow', 'green', 'blue'])
            >>> set2 = OrderedSet(['cyan', 'magenta', 'yellow'])
            >>> merged, indices = set1.merge(set2)
            >>> for item in merged:
            ...     print item,
            red orange yellow green blue cyan magenta
            >>> print indices
            [5, 6, 2]
        """
        merged = self.copy()
        indices = [merged.add(item) for item in other]
        return merged, indices

    def __setitem__(self, n, newkey):
        assert hasattr(n, '__index__')
        oldkey = self.items[n]
        del self.indices[oldkey]
        self.items[n] = newkey
        self.indices[newkey] = n

    def __delitem__(self, n):
        """
        Deletes an item from the OrderedSet.

        This is a bit messy. It'll just leave a hole in the list. Do you
        really want to do that?
        """
        oldkey = self.items[n]
        del self.indices[oldkey]
        self.items[n] = None

    def __iter__(self):
        for item in self.items:
            if item is not None:
                yield item

    def __eq__(self, other):
        '''Two OrderedSets are equal if their items are equal.

            >>> a = OrderedSet(['a', 'b'])
            >>> b = OrderedSet(['a'])
            >>> b.add('b')
            1
            >>> a == b
            True
        '''
        if self is other: return True
        if len(self) != len(other): return False
        if not isinstance(other, OrderedSet): return False

        for (s, o) in izip(self, other):
            if s != o: return False
        return True

    def __ne__(self, other):
        return not self == other

class IdentitySet(object):
    '''
    An object that behaves like an :class:`OrderedSet`, but simply contains
    the range of numbers up to *len*. Thus, every number is its own index.

    IdentitySets were used in Divisi1 classes to label :class:`Tensors
    <divisi.tensor.Tensor>` on axes where labels would be meaningless or
    unnecessary.
    
    In Divisi2, using "None" as a label set does the same thing, making
    IdentitySets obsolete. However, they are still useful for testing fancy
    indexing.
    '''
    index_is_efficient = True
    __slots__ = ['len']

    def __init__(self, len):
        self.len = len

    def __repr__(self): return 'IdentitySet(%d)' % (self.len,)
    def __len__(self): return self.len

    # Doesn't check for out-of-range or even that it's an integer.
    def __getitem__(self, x): return x
    def index(self, x): return x
    def __iter__(self): return iter(xrange(self.len))
    def add(self, key):
        if key + 1 > self.len:
            self.len = key + 1
        return key
    @property
    def items(self): return range(self.len)
    def __eq__(self, other):
        return self.items == getattr(other, 'items', other)
    def __ne__(self, other): return not self == other

    def __getstate__(self): return dict(len=self.len)
    def __setstate__(self, state): self.len = state['len']

def indexable_set(x, dim=None):
    if x is None:
        return IdentitySet(dim)
    if getattr(x, 'index_is_efficient', False):
        return x
    return OrderedSet(x)

IndexTester = IdentitySet(0)
newaxis = None
ALL = slice(None, None, None)

def apply_indices(indices, indexables):
    """
    This function reverse-engineers what NumPy's indexing does, turning the
    argument to __getitem__ into a list of individual expressions that will be
    used by classes that aren't fancy and multidimensional, such as
    OrderedSets. `indices` are the list of indices, and `indexables` are
    the things to be indexed.

    The number of dimensions of data you have should be the length of
    `indexables`. The number of dimensions you get from indexing may be
    different -- for example, you can choose a 0-dimensional cell from a
    2-dimensional matrix. Call this number of dimensions *D*: this function
    will always return you a list of *D* results.

    This means that indices that are simple integers aren't looked up, they're
    just dropped from the results. This makes sense when you're labeling axes
    of a matrix: labels get dropped when there's no axis left to label. Scalars
    don't need row and column labels, they're just scalars. Vectors have only
    one list of labels.

    The following examples index a silly object called `IndexTester`, whose
    __getitem__ function is the identity.

    >>> labels0d = []
    >>> labels1d = [IndexTester]
    >>> labels2d = [IndexTester, IndexTester]
    
    A single index can be specified as a singleton tuple or a scalar:
    
    >>> apply_indices(1, labels1d)
    []
    >>> apply_indices((1,), labels1d)
    []

    A list is different, however:

    >>> apply_indices([1], labels1d)
    [[1]]

    If you index fewer dimensions than you have, the indices are padded to the
    appropriate number of dimensions with `:` slices, which select everything,
    and print as `slice(None, None, None)`.

    >>> apply_indices((), labels2d)
    [slice(None, None, None), slice(None, None, None)]
    >>> apply_indices((), labels1d)
    [slice(None, None, None)]
    >>> apply_indices((), labels0d)
    []
    >>> apply_indices(slice(None), labels1d)
    [slice(None, None, None)]

    The Ellipsis object expands into as many `:`s as possible:

    >>> apply_indices(Ellipsis, labels2d)
    [slice(None, None, None), slice(None, None, None)]
    >>> apply_indices((Ellipsis, [3, 4]), labels2d)
    [slice(None, None, None), [3, 4]]
    >>> apply_indices((Ellipsis, [3, 4]), labels1d)
    [[3, 4]]
    >>> apply_indices((Ellipsis, 3), labels2d)
    [slice(None, None, None)]
    >>> apply_indices((Ellipsis, 3), labels1d)
    []

    You can get more dimensions than you started with using `np.newaxis`
    (which is equal to None). The new dimensions will have None as their
    index results.

    >>> apply_indices((newaxis, newaxis), labels0d)
    [None, None]
    >>> apply_indices((newaxis, newaxis), labels1d)
    [None, None, slice(None, None, None)]
    >>> apply_indices((Ellipsis, newaxis, newaxis), labels2d)
    [slice(None, None, None), slice(None, None, None), None, None]

    One of the *indexables* can be None as well, in which case the result of
    indexing it will always be None.

    >>> apply_indices((Ellipsis, newaxis, newaxis), [[3, 4], None])
    [[3, 4], None, None, None]
    >>> apply_indices(([1], [2]), [OrderedSet([3, 4]), None])
    [OrderedSet([4]), None]
    >>> apply_indices((1, 2), [[3, 4], None])
    []
    """
    
    # Make indices into a list
    if isinstance(indices, tuple):
        indices = list(indices)
    else:
        indices = [indices]
    indexables = list(indexables)

    num_axes_in_data = len(indexables)
    num_axes_known = len(indices)
    for index in indices:
        # .count doesn't work over things that might be NumPy arrays
        if index is None or index is Ellipsis:
            num_axes_known -= 1

    if num_axes_known > num_axes_in_data:
        raise IndexError("Too many indices")

    # Expand ellipses... from right to left, it turns out.
    for i in reversed(xrange(len(indices))):
        if indices[i] is Ellipsis:
            indices[i:i+1] = [ALL] * (num_axes_in_data - num_axes_known)
            num_axes_known = num_axes_in_data

    while num_axes_known < num_axes_in_data:
        indices.append(ALL)
        num_axes_known += 1

    results = []
    which_indexable = 0
    # Now step through the axes and get stuff
    for index in indices:
        if index is newaxis:
            results.append(None)
        else:
            indexable = indexables[which_indexable]
            if hasattr(index, '__index__'):
                # simple index: drop this result
                pass
            elif indexable is None:
                results.append(None)
            else:
                results.append(indexable[index])
            which_indexable += 1
    return results

