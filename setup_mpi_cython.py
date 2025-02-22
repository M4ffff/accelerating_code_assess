from distutils.core import setup
from Cython.Build import cythonize

setup(name="mpi4pyCythonLebwohlLasher",
      ext_modules=cythonize("mpi4pyCythonLebwohlLasher.pyx"))