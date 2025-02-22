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


#=======================================================================
def one_energy_mpi_cythonised(double[:] current_row, double[:] above_row, double[:] below_row, int ix, int nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  ix (int) = x lattice coordinate of cell;
	  iy (int) = y lattice coordinate of cell;
      nmax (int) = side length of square lattice.
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
	Returns:
	  en (float) = reduced energy of cell.
    """
    cdef:
        float en = 0.0
        float ang
# HERE
# PARALLELISE
    ang = current_row[ix]-current_row[(ix+1)%nmax]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    ang = current_row[ix]-current_row[(ix-1)%nmax]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    ang = current_row[ix]-above_row[ix]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    ang = current_row[ix]-below_row[ix]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    return en



cdef double calc_boltz(double diff, double Ts) nogil:
  cdef:
    double boltzval 
  boltzval = exp( -(diff) / Ts )
  return boltzval



def update_cythonised(double[:] current_row, double[:] above_row, double[:] below_row, double[:] aran, double[:] rand_row, int nmax, double Ts):
    cdef:
        int accept = 0
        int iy
        double ang, en0, en1, boltz


    for iy in range(nmax):
        ang = aran[iy]
        en0 = one_energy_mpi_cythonised(current_row, above_row, below_row,iy,nmax)
        current_row[iy] += ang
        en1 = one_energy_mpi_cythonised(current_row, above_row, below_row,iy,nmax)
        if en1<=en0:
            accept += 1
        else:
        # Now apply the Monte Carlo test - compare
        # exp( -(E_new - E_old) / T* ) >= rand(0,1)
            boltz = calc_boltz(en1-en0, Ts)

            if boltz >= rand_row[iy]:
                accept += 1
            else:
                current_row[iy] -= ang
                
    return accept