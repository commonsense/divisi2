#!/usr/bin/env python

"""Divisi2: Commonsense Reasoning over Semantic Networks

Divisi2 is a library for reasoning by analogy and association over
semantic networks, including common sense knowledge. Divisi uses a
sparse higher-order SVD and can help find related concepts, features,
and relation types in any knowledge base that can be represented as a
semantic network. By including common sense knowledge from ConceptNet,
the results can include relationships not expressed in the original
data but related by common sense. See http://divisi.media.mit.edu/ for
more info."""

VERSION = "2.2.5"

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
import os.path, sys
from stat import ST_MTIME

classifiers=[
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License (GPL)',
    'Natural Language :: English',
    'Operating System :: MacOS',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Programming Language :: C',
    'Programming Language :: Python :: 2.5',
    'Programming Language :: Python :: 2.6',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development',
    'Topic :: Text Processing :: Linguistic',]

### Check for the existence of NumPy.
try:
    import numpy
except ImportError:
    print >>sys.stderr, """This package requires NumPy.

On a Debian / Ubuntu system, you can run:
  sudo apt-get install python-numpy python-dev

Otherwise it will probably suffice to:
  sudo easy_install numpy
"""
    sys.exit(1)


CYTHON_OUT = 'svdlib/_svdlib.c'
CYTHON_SRC = 'svdlib/_svdlib.pyx'

### Update the Cython file, if necessary.
def get_modification_time(filename):
    return os.stat(filename)[ST_MTIME]
try:
    if not os.path.exists(CYTHON_OUT) or get_modification_time(CYTHON_SRC) > get_modification_time(CYTHON_OUT):
        try:
            # Try building the Cython file
            print 'Building Cython source'
            from Cython.Compiler.Main import compile
            res = compile(CYTHON_SRC)
            if res.num_errors > 0:
                print >>sys.stderr, "Error building the Cython file."
                sys.exit(1)
        except ImportError:
            print >>sys.stderr, 'Warning: Skipped building the Cython file.'
            print >>sys.stderr, ' The svdlib source file is more recent than the Cython output file, but'
            print >>sys.stderr, ' you seem to lack Cython, so skipping rebuilding it.'
            raw_input('Press Enter to acknowledge. ')
except OSError:
    print >>sys.stderr, 'Warning: Skipped building the Cython file.'

svdlibc = Extension(
    name='divisi2._svdlib',
    sources=[
        CYTHON_OUT,
        'svdlib/svdwrapper.c',
        'svdlib/las2.c',
        'svdlib/svdlib.c',
        'svdlib/svdutil.c',
        ],
    include_dirs=[numpy.get_include(), 'svdlib'],
    extra_compile_args=['-g'],
    extra_link_args=['-g'],
    )

doclines = __doc__.split("\n")

setup(
    name="Divisi2",
    version = VERSION,
    maintainer='MIT Media Lab, Software Agents group',
    maintainer_email='conceptnet@media.mit.edu',
    url='http://divisi.media.mit.edu/',
    license = "http://www.gnu.org/copyleft/gpl.html",
    platforms = ["any"],
    description = doclines[0],
    classifiers = classifiers,
    long_description = "\n".join(doclines[2:]),
    ext_modules = [svdlibc],
    packages=['divisi2', 'divisi2.algorithms', 'divisi2.test', 'divisi2.test.eval'],
    package_data = {'divisi2': ['data/graphs/*', 'data/eval/*', 'data/matrices/*']},
    install_requires=['csc-utils >= 0.6.1', 'networkx', 'csc-pysparse'],
)
