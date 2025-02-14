from distutils.core import setup
from Cython.Build import cythonize

setup(name="CythonLebwohlLasher",
      ext_modules=cythonize("CythonLebwohlLasher.pyx"))