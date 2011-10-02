import sys

def delayed_plugin(module, fname, package='divisi2.algorithms'):
    modname = package+'.'+module
    def plugin_method(*args, **kw):
        """
        Sorry, this meta-code won't be too informative. If you're seeing this in
        IPython's ?? mode, try a single question mark instead.
        """
        __import__(modname)
        mod = sys.modules[modname]
        func = getattr(mod, fname)
        return func(*args, **kw)
    plugin_method.__name__ = fname
    plugin_method.__doc__ = """
    Import the %(fname)s algorithm and perform it on this matrix.

    More information is available in the documentation for
    :func:`%(modname)s.%(fname)s`.
    """ % locals()
    return plugin_method
    
class LearningMixin(object):
    """
    A class that inherits from LearningMixin will
    gain methods that perform various algorithms that learn from sparse data.

    The class should support the :meth:`to_sparse` and :meth:`to_dense`
    methods, to work with algorithms that expect one form of input or the
    other.
    """
    _methods = []
    svd = delayed_plugin('svd', 'svd')
    lmds = delayed_plugin('mds', 'lmds')
    nmf = delayed_plugin('nmf', 'fnmai')
    rsvd = delayed_plugin('randomized_svd', 'randomized_svd')

    @classmethod
    def available_methods(cls):
        return cls._methods

