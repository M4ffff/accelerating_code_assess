# accelerating_code_assess
Commit history of software engineering and high performance computing assessment. 

Files with different methods for accelerating the provided code.


Files of interest:
- LebwohlLasher.py 
provided script tasked with accelerating. Edited slightly to make more readable but main code unchanged.

SETUP INSTRUCTIONS
python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>




- NumpyVecLebwohlLasher.py 
script vectorised using numpy replacing for loops. Significant speedups, although less so at larger lattice sizes.

SETUP INSTRUCTIONS
python NumpyVecLebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>




- NumbaLebwohlLasher.py 
main bulk of code based on NumpyVecLebwohlLasher, but multiple functions compiled using numba. generally fast. 
SETUP INSTRUCTIONS
python NumbaLebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>




- CythonLebwohlLasher.pyx / run_cython.py / setup_cython.py
Main bulk of code based on LebwohlLasher. Multiple functions compiled using Cython. Generally fast

SETUP INSTRUCTIONS
python setup_cython.py build_ext -fi
python run_cython.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> 




- ParallelCythonLebwohlLasher.pyx / run_parallel_cython.py / setup_parallel_cython.py
Very similar to CythonLebwohlLasher.pyx, but parallelised. 

SETUP INSTRUCTIONS
python setup_parallel_cython.py build_ext -fi
python run_parallel_cython.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <THREADS>
*** note number of threads needed ***




- mpi4pyLebwohlLasher.py
Script based on LebwohlLasher.py but edited to run on multiple threads/workers. No improvement

SETUP INSTRUCTIONS
mpiexec -n <num_cores> python mpi4pyLebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>




- mpiCythonLebwohlLasher.pyx / run_mpi_cython.py / setup_mpi_cython.py
Script based on mpi4pyLebwohlLasher.py but with some functions cythonised. Major improvement on mpi4py, but due to cytonisation rather than mpi. 

SETUP INSTRUCTIONS
python setup_mpi_cython.py build_ext -fi
mpiexec -n <num_cores> python run_mpi_cython.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>



- making_pkls.ipynb
Jupyter notebook to produce and save files to be plotted.



- plotting_script.ipynb
Script used to load in relevant data files and plot them. 

