from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

ext_modules = [
    Extension(
        "ParallelCythonLebwohlLasher",
        ["ParallelCythonLebwohlLasher.pyx"],
         extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
    )
]

setup(name="ParallelCythonLebwohlLasher",
      ext_modules=cythonize(ext_modules))


