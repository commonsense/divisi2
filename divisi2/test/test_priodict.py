from csc.divisi2.priodict import priorityDictionary
from StringIO import StringIO
import cPickle as pickle

def test_priodict():
    picklestr = StringIO()
    p = priorityDictionary()
    p[1] = 1
    p['foo'] = 2
    pickle.dump(p, picklestr)
    picklestr.seek(0)
    p2 = pickle.load(picklestr)
    assert p2[1] == 1
    assert p2['foo'] == 2
