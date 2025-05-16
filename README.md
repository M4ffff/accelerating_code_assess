# accelerating_code_assess
Commit history of software engineering and high performance computing assessment. 

Files with different methods for accelerating the provided code.


Files of interest:

LebwohlLasher.py was the script with which I was tasked with accelerating, written by Simon Hanna. 

Run by typing this in the commandline:

```python
python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>
```

It has been edited to make it slightly more readable than the original, but the structure of the code remains unchanged.
The code contains many nested loops, which are slow to run in Python.
A Profiler was used to determine the functions of the code which needed accelerating the most.

There were four different methods used to accelerate this script.

- Vectorisation of the nested loop calculations using NumPy arrays.

- Compilation of certain Python functions using Numba.

- Many functions rewritten with Cython. 

- Parallelisation of loops using MPI system.

There was also a final script which combined Cython with the MPI system. 

Below, I will briefly run through how each script has changed, and how to run them. 

- NumpyVecLebwohlLasher.py
Script vectorised using NumPy arrays, replacing the for loops. Significant speedups, although less so at larger lattice sizes.

``` python
python NumpyVecLebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>
```

- NumbaLebwohlLasher.py 
main bulk of code based on NumpyVecLebwohlLasher, but multiple functions compiled using numba. Generally fast.

```python
python NumbaLebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>
```

- CythonLebwohlLasher.pyx / run_cython.py / setup_cython.py
Main bulk of code based on the original LebwohlLasher script. Multiple functions compiled using Cython. Generally fast.

```python
python setup_cython.py build_ext -fi
python run_cython.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> 
```

- mpi4pyLebwohlLasher.py
Script based on LebwohlLasher.py but edited to run on multiple threads/workers. Very little improvement.

```python
mpiexec -n <num_cores> python mpi4pyLebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>
```

- ParallelCythonLebwohlLasher.pyx / run_parallel_cython.py / setup_parallel_cython.py
  
Very similar to CythonLebwohlLasher.pyx, but parallelised. 

```python
python setup_parallel_cython.py build_ext -fi
python run_parallel_cython.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <THREADS>
```

*** note number of threads needs to be included in command line ***

- mpiCythonLebwohlLasher.pyx / run_mpi_cython.py / setup_mpi_cython.py
Script based on mpi4pyLebwohlLasher.py but with some functions cythonised. 
Major improvement comapred mpi4py, but the improvement is due to the cythonised functions, rather than the use of mpi.

SETUP INSTRUCTIONS

```python
python setup_mpi_cython.py build_ext -fi
mpiexec -n <num_cores> python run_mpi_cython.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>
```

- making_pkls.ipynb

Jupyter notebook to produce and save files to be plotted.

- plotting_script.ipynb
Script used to load in relevant data files and plot them.
