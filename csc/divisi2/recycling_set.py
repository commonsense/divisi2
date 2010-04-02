from csc.divisi2.ordered_set import OrderedSet
from csc.divisi2.priodict import priorityDictionary

class RecyclingSet(OrderedSet):
    __slots__ = ['items', 'indices', 'index', 'indexFor', '__contains__',
                 '__getitem__', '__len__', 'count', 'maxsize',
                 'drop_listeners', 'priority']
    def __init__(self, maxsize, origitems=None):
        self.count = 0
        self.maxsize = maxsize
        self.priority = priorityDictionary()
        self.drop_listeners = []
        OrderedSet.__init__(self, origitems)

    def __getstate__(self):
        return (self.items, self.priority, self.maxsize, self.count)
    def __setstate__(self, state):
        items, self.priority, self.maxsize, self.count = state
        OrderedSet.__setstate__(self, items)

    def add(self, key):
        """
        Add an item to the set (unless it's already there),
        returning its index. Drop an old item if necessary.

        ``None`` is never an element of an OrderedSet.
        """

        if key in self.indices:
            self.touch(key)
            return self.indices[key]
        n = len(self.items)
        if n < self.maxsize:
            self.items.append(key)
            if key is not None:
                self.indices[key] = n
            self.touch(key)
            return n
        else:
            newindex = self.drop_oldest()
            self.items[newindex] = key
            self.indices[key] = newindex
            self.touch(key)
            return newindex
    append = add

    def __delitem__(self, n):
        """
        Deletes an item from the RecyclingSet.
        """
        oldkey = self.items[n]
        del self.indices[oldkey]
        self.items[n] = None
        self.announce_drop(n, oldkey)

    def drop_oldest(self):
        """
        Drop the least recently used item, to make room for a new one. Return
        the number of the slot that just became free.
        """
        slot = self.priority.smallest()
        oldest = self.items[slot]
        del self[slot]
        return slot

    def listen_for_drops(self, callback):
        """
        If an object needs to know when a slot becomes invalid because its
        key gets dropped, it should register a callback with listen_for_drops.
        """
        self.drop_listeners.append(callback)

    def announce_drop(self, index, key):
        """
        Tell all registered listeners that we dropped a key.
        """
        print "dropping key:", key
        for listener in self.drop_listeners:
            listener(index, key)

    def touch(self, key):
        """
        Remember that this key is useful.
        """
        if key not in self: raise IndexError
        else:
            self.count += 1
            self.priority[self.index(key, False)] = self.count

    def index(self, key, touch=True):
        if touch: self.touch(key)
        return self.indices[key]
    indexFor = index

    def __contains__(self, key):
        return key in self.indices

    def __getitem__(self, key):
        if key < self.maxsize and key >= len(self.items):
            return None
        return self.items[key]

    def __len__(self):
        return len(self.indices)

    def _setup_quick_lookup_methods(self):
        pass
