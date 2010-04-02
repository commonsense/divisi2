from nose.tools import *
from csc.divisi2.recycling_set import RecyclingSet
from csc.divisi2.ordered_set import OrderedSet, IdentitySet

import cPickle as pickle

def test_reprOfEmpty():
    '''
    repr() of an empty OrderedSet should not fail.
    '''
    repr(RecyclingSet(10))

def test_identity():
    '''
    Identity sets are just ranges of numbers.
    '''
    iset = IdentitySet(10)
    eq_(iset[5], 5)
    eq_(iset.index(2), 2)
    eq_(len(iset), 10)
    assert iset == RecyclingSet(10, range(10))

    iset = pickle.loads(pickle.dumps(iset))
    eq_(iset[5], 5)
    eq_(iset.index(2), 2)
    eq_(len(iset), 10)
    assert iset == RecyclingSet(10, range(10))

def test_pickle():
    '''
    Test that RecyclingSets can be pickled.
    '''
    s = RecyclingSet(10, ['dog','cat','banana'])
    import cPickle as pickle
    s2 = pickle.loads(pickle.dumps(s))

    eq_(s, s2)
    eq_(s2[0], 'dog')
    eq_(s2.index('cat'), 1)


def test_delete_and_pickle():
    '''
    Deleting an element doesn't affect the remaining elements'
    indices.
    '''
    s = RecyclingSet(10, ['dog','cat','banana'])
    del s[1]
    eq_(s[1], None)
    eq_(s.index('banana'), 2)

    # Pickling doesn't change things.
    s2 = pickle.loads(pickle.dumps(s))

    eq_(s, s2)
    eq_(s2[1], None)
    eq_(s2.index('banana'), 2)

    assert None not in s2
    assert None not in s2

def test_recycle():
    r = RecyclingSet(5)
    for item in ('one', 'two', 'three', 'four', 'five', 'six'): r.add(item)
    assert r[0] == 'six'
    assert 'one' not in r

def test_recycle2():
    r = RecyclingSet(5)
    for item in ('one', 'two', 'three', 'four', 'five', 'six'): r.add(item)
    r.add('two')
    assert r[0] == 'six'
    assert r[1] == 'two'

    r.add('seven')
    assert r[1] == 'two'
    assert r[2] == 'seven'

def test_duplicates():
    r = RecyclingSet(5)
    for i in xrange(10):
        r.add('two')
    assert len(r) == 1
