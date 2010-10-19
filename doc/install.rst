Installing Divisi2
==================

The easiest way to install Divisi2 is to use the ``pip`` package installer::

    pip install divisi2 csc-pysparse

.. note::

   We've contributed some changes to the PySparse project that haven't made it
   into an official release yet, so for now we've made our own unofficial PyPI
   package called ``csc-pysparse``. There's also something wonky about its
   installer which makes it not work when installed as a dependency, so you
   have to ask for it explicitly for now. Hopefully this will be sorted out
   soon.

If this doesn't work for you, keep reading.

Setting up GCC on Windows
-------------------------

Divisi2 and PySparse contain C code that needs to be compiled using GCC.
The easiest way to get GCC on Windows is to download and install the MinGW_
suite. 

.. _MinGW: http://www.mingw.org/wiki/HOWTO_Install_the_MinGW_GCC_Compiler_Suite

After you do this, you need to tell Python to use it. Create a text file called
``C:\Python26\Lib\distutils\distutils.cfg`` containing the following::

    [build]
    compiler=mingw32

Installing with easy_install
----------------------------
If you only have ``easy_install``, and not ``pip``, you may be able to do
this::

    easy_install csc-pysparse
    easy_install divisi2

(Change ``easy_install`` to ``sudo easy_install`` if you're on Mac or Linux and
you're installing into the global Python environment.)

On the other hand, you may be happier in the long run if you just do this::

    easy_install pip

