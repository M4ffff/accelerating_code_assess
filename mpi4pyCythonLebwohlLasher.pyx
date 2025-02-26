import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


from libc.math cimport sqrt, cos, sin, exp
from cython.parallel cimport prange
cimport openmp
# cimport numpy as cnp
from cython import boundscheck, wraparound, cdivision


@boundscheck(False)
@wraparound(False)
@cdivision(True)
def one_energy_loop(double[:,:] arr, double[:,:] aran, double[:,:] randarr, int i, int nmax, double Ts):
  """
  arr (arr): worker lattice array
  aran (arr): array of random angle changes
  randarr (arr): random array of values between 0 and 1
  i (int): value of row in lattice
  nmax (int): size of lattice
  Ts (double): temp of system
  

  reutnr number of accepted changes
  """

  cdef:
    int j
    double en0, en1, ang
    int accept = 0

  for j in range(nmax):
      # pick random angle
      ang = aran[i,j]

      # old_energy
      en0 = one_energy_cythonised(arr,i,j,nmax)
      
      # new energy
      arr[i,j] += ang
      en1 = one_energy_cythonised(arr,i,j,nmax)
          
          
          
      if en1<=en0:
          accept += 1
      else:
      # Now apply the Monte Carlo test - compare
      # exp( -(E_new - E_old) / T* ) >= rand(0,1)
          boltz = exp( -(en1 - en0) / Ts )

          if boltz >= randarr[i,j]:
              accept += 1
          else:
              arr[i,j] -= ang
  return accept
