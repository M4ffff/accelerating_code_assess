# accelerating_code_assess
Commit history of software engineering and high performance computing assessment. 

Files with different methods for accelerating the provided code.


Files of interest:
- LebwohlLasher.py 
provided script tasked with accelerating. Edited slightly to make more readable but main code unchanged.

- NumpyVecLebwohlLasher.py 
script vectorised using numpy replacing for loops. Significant speedups, although less so at larger lattice sizes.

- NumbaLebwohlLasher.py 
main bulk of code based on NumpyVecLebwohlLasher, but multiple functions compiled using numba. generally fast. 


- CythonLebwohlLasher.pyx / run_cython.py / setup_cython.py
Main bulk of code based on LebwohlLasher. Multiple functions compiled using Cython. Generally fast

- ParallelCythonLebwohlLasher.pyx / run_parallel_cython.py / setup_parallel_cython.py
Very similar to CythonLebwohlLasher.pyx, but parallelised. 

- mpi4pyLebwohlLasher.py
Script based on LebwohlLasher.py but edited to run on multiple threads/workers. No improvement

- mpiCythonLebwohlLasher.pyx / run_mpi_cython.py / setup_mpi_cython.py
Script based on mpi4pyLebwohlLasher.py but with some functions cythonised. Major improvement on mpi4py, but due to cytonisation rather than mpi. 



