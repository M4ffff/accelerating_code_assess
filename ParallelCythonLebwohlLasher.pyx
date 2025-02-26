"""
See run_parallel_cython.py
"""

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
from cython import boundscheck


  


#=======================================================================
@boundscheck(False)
cdef double one_energy_cythonised(double[:,:] arr, int ix, int iy, int nmax) nogil:

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
      double en = 0.0
      int ixr = (ix+1)%nmax # These are the coordinates
      int ixl = (ix-1)%nmax # of the neighbours
      int iyu = (iy+1)%nmax # with wraparound
      int iyd = (iy-1)%nmax #
      double ang
      
    ang = arr[ix,iy]-arr[ixr,iy]
    en += 0.5*(1.0 - 3.0*cos(ang)*cos(ang))
    ang = arr[ix,iy]-arr[ixl,iy]
    en += 0.5*(1.0 - 3.0*cos(ang)*cos(ang))
    ang = arr[ix,iy]-arr[ix,iyu]
    en += 0.5*(1.0 - 3.0*cos(ang)*cos(ang))
    ang = arr[ix,iy]-arr[ix,iyd]
    en += 0.5*(1.0 - 3.0*cos(ang)*cos(ang))

    return en
  
  
#=======================================================================
@boundscheck(False)
def all_energy_cythonised(double[:,:] arr, int nmax):

    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
	Returns:
	  enall (float) = reduced energy of lattice.
    """
    cdef:
      double enall = 0.0
      int i 
      int j
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy_cythonised(arr,i,j,nmax)
    return enall
  
  
#=======================================================================
@boundscheck(False)
def get_order_loop(double[:,:] Qab, int nmax, double[:,:,:] lab, double[:,:] delta, int factor, int threads):
  """
  calculate order tensor

  Qab (arr): order tensor
  nmax (int): side length of lattice
  lab (arr): 3D unit vector for each cell (i,j)
  delta (arr): identity matrix
  factor (int): 2*nmax*nmax
  
  return Qab
  """
  cdef:
    int a, b, i, j

  
  for a in range(2):
      for b in range(2):
          #for i in prange(nmax, nogil=True, num_threads=threads):
          for i in range(nmax):
              for j in range(nmax):
                  Qab[a,b] += ( 3*lab[a,i,j]*lab[b,i,j] - delta[a,b] ) / factor
  return Qab


@boundscheck(False)
def get_lab(double[:,:,:] lab, double[:,:] arr, int nmax, int threads):
    """
    calcualte lab

    lab (arr): 3D unit vector for each cell (i,j)
    arr (arr): lattice array
    nmax (int): side length of lattice

    return lab
    """
    
    cdef: 
      int i, j
    for i in prange(nmax, nogil=True, num_threads=threads):
    #for i in range(nmax):
      for j in range(nmax):
        lab[0, i, j] = cos(arr[i,j])
        lab[1, i, j] = sin(arr[i,j])
    return lab

 
cdef double calc_boltz(double diff, double Ts) nogil:
  """
  calculate boltzmann factor
  diff (double): difference between old and new energy
  Ts (double): temperature of system
  """
  
  cdef:
    double boltzval 
  boltzval = exp( -(diff) / Ts )
  return boltzval

@boundscheck(False)
def MC_step_loop(double[:,:] aran, int nmax, double[:,:] arr, double Ts, double[:,:] randarr, int threads):
      """
      determine new energies and whether to keep them

      aran (arr): array of random angles
      nmax (int):  side length of square lattice
      arr (arr): lattice array
      Ts (double): temperature of system
      randarr (arr): array of random values between 0 and 1.
      

      return number of accepted energies
      """
      
      cdef: 
        int accept = 0
        int i, j
        double diff, boltz
        double ang, en0, en1

      for i in prange(0, nmax, 2, nogil=True, num_threads=threads):
        for j in range(nmax):
            # pick random angle

            ang = aran[i,j]
            
            # old_energy
            en0 = one_energy_cythonised(arr,i,j,nmax)
            
            # new energy
            arr[i,j] += ang
            en1 = one_energy_cythonised(arr,i,j,nmax)
            diff = en1 - en0
            if diff<=0:
                accept += 1
            else:
            # Now apply the Monte Carlo test - compare
            # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                boltz = calc_boltz(diff, Ts)

                if boltz >= randarr[i,j]:
                  with gil:
                    accept += 1
                else:
                    arr[i,j] -= ang

      for i in prange(1, nmax, 2, nogil=True, num_threads=threads):
        for j in range(nmax):
            # pick random angle

            ang = aran[i,j]
            
            # old_energy
            en0 = one_energy_cythonised(arr,i,j,nmax)
            
            # new energy
            arr[i,j] += ang
            en1 = one_energy_cythonised(arr,i,j,nmax)
            diff = en1 - en0
            if diff<=0:
              with gil:
                accept += 1
            else:
            # Now apply the Monte Carlo test - compare
            # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                boltz = calc_boltz(diff, Ts)

                if boltz >= randarr[i,j]:
                    accept += 1
                else:
                    arr[i,j] -= ang
      return accept


#=======================================================================
def MC_step_cythonised(double[:,:] arr, double Ts, int nmax, double scale, int threads):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
      scale (double) = sets the width of the distribution for the angle changes
    Description:
      Function to perform one MC step, which consists of an average
      of 1 attempted change per lattice site.  Working with reduced
      temperature Ts = kT/epsilon.  Function returns the acceptance
      ratio for information.  This is the fraction of attempted changes
      that are successful.  Generally aim to keep this around 0.5 for
      efficient simulation.
	Returns:
	  accept/(nmax**2) (float) = acceptance ratio for current MCS.
    """
    #
    # Pre-compute some random numbers.  This is faster than
    # using lots of individual calls.  "scale" sets the width
    # of the distribution for the angle changes - increases
    # with temperature.
    # std
    cdef:
      int accept = 0
      double[:,:] aran
      double[:,:] boltzran

    aran = np.random.normal(scale=scale, size=(nmax,nmax))
    boltzran = np.random.uniform(0.0, 1.0, size=(nmax, nmax))
    
    accept = MC_step_loop(aran, nmax, arr, Ts, boltzran, threads)
    
    # print(arr)                
    # print(f"accepted: {accept}")
    return accept/(nmax*nmax)


def main_loop(double[:,:] lattice,double temp, int nmax, double scale):
    """
    calculate ratio and energy for one Monte Carlo step

    lattice (arr): lattice array
    temp (double): temperature of system
    nmax (int):  side length of square lattice
    scale (double): sets the width of the distribution for the angle changes

    return ratio,energy for this MCS
    
    """
    cdef:
      double ratio
      double energy
    ratio = MC_step_cythonised(lattice,temp,nmax, scale) #, threads)
    energy = all_energy_cythonised(lattice,nmax)
    return ratio, energy




