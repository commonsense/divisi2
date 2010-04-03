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

VERSION = "0.7.0"

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils import setup, Extension
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

### Update the Cython file, if necessary.
def get_modification_time(filename):
    return os.stat(filename)[ST_MTIME]

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
    ext_modules = [],
    packages=['csc', 'csc.divisi2'],
    namespace_packages = ['csc'],
    install_requires=['csc-utils >= 0.4.1',],
)
