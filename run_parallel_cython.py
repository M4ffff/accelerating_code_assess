"""
Parallelised Cythonised Python Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

compile
python setup_parallel_cython.py build_ext -fi

python run_parallel_cython.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <THREADS>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  THREADS = number of threads
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  

"""


import sys
from ParallelCythonLebwohlLasher import all_energy_cythonised, get_order_loop, get_lab, MC_step_loop, MC_step_cythonised, main_loop

import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#=======================================================================
def initdat(nmax):
    """ 
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr
  
  
#=======================================================================
def plot_reduced_e(energy, nsteps, temp):
  """
  Plot reduced energy against number of monte carlo steps

  Args:
      energy (arr): array of reduced energies
      nsteps (int): number of steps
      temp (float): temperature of simulation
  """
  fig, ax = plt.subplots()
  steps = np.arange(0,nsteps+1)
  ax.plot(steps, energy)
  ax.set_xlabel("MCS")
  ax.set_ylabel("Reduced Energy")
  ax.set_title(f"Reduced Temperature, T* = {temp}")
  plt.show()
  
  
def plot_order(order, nsteps, temp):
  """
  Plot of order against number of monte carlo steps

  Args:
      order (arr): array of order throughout simulation
      nsteps (int): number of steps
      temp (float): temperature of simulation
  """
  fig, ax = plt.subplots()
  steps = np.arange(0,nsteps+1)
  ax.plot(steps, order)
  ax.set_xlabel("MCS")
  ax.set_ylabel("Order Parameter")
  ax.set_title(f"Reduced Temperature, T* = {temp}")
  plt.show()
  
 
  
   
#=======================================================================
def one_energy(arr,ix,iy,nmax):
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
	  energy (float) = reduced energy of cell.
    """
    energy = 0.0
    
    # adjacent in x direction (right/left)
    ixr = (ix+1)%nmax # These are the coordinates
    ixl = (ix-1)%nmax # of the neighbours with wraparound
    
    # adjacent in y direction (up/down)
    iyu = (iy+1)%nmax # 
    iyd = (iy-1)%nmax #
#
    ang = arr[ix,iy]-arr[ixr,iy]
    energy += 0.5*(1.0 - 3.0*np.cos(ang)*np.cos(ang))
    ang = arr[ix,iy]-arr[ixl,iy]
    energy += 0.5*(1.0 - 3.0*np.cos(ang)*np.cos(ang))
    ang = arr[ix,iy]-arr[ix,iyu]
    energy += 0.5*(1.0 - 3.0*np.cos(ang)*np.cos(ang))
    ang = arr[ix,iy]-arr[ix,iyd]
    energy += 0.5*(1.0 - 3.0*np.cos(ang)*np.cos(ang))
    return energy
  
  
#=======================================================================
def plotdat(arr,pflag,nmax, final_data=False, energy=None, temp=None, order=None, nsteps=None):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    # no plotting
    if pflag==0:
        return
      
    # x and y components of quivers
    u = np.cos(arr)
    v = np.sin(arr)
    
    # x and y positions of quivers
    x = np.arange(nmax)
    y = np.arange(nmax)
    
    # colours
    cols = np.zeros((nmax,nmax))
    
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        
        # calc colour of each quiver
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
        
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
        
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    
    fig, ax = plt.subplots()
    ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
    
    if final_data and pflag != 0:
      plot_reduced_e(energy, nsteps, temp)
      plot_order(order, nsteps, temp)
      # plot_order_vs_temp(order, temp, nmax)
    
#=======================================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
    
    

      
#=======================================================================
def get_order(arr,nmax, delta, factor, threads):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
      delta (arr) = identity matrix
      factor (int) = 2*nmax*nmax
      threads (int) = number of threads
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    Qab = np.zeros((2,2), dtype=np.float64)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    lab = np.zeros((2, nmax, nmax), dtype=np.float64)
    lab = get_lab(lab, arr, nmax, threads)
    Qab = get_order_loop(Qab, nmax, lab, delta, factor, threads)
    eigenvalues = np.linalg.eig(Qab)[0]
    return eigenvalues.max()
  
  
#=======================================================================
def MC_step(arr,Ts,nmax, scale, threads):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
      scale (double) = sets the width of the distribution for the angle changes
      threads (int) = number of threads
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
    
    accept = 0
    aran = np.random.normal(scale=scale, size=(nmax,nmax))
    boltzran = np.random.uniform(0.0, 1.0, size=(nmax, nmax))
    
    accept = MC_step_loop(aran, nmax, arr, Ts, boltzran, threads)
    
    # print(arr)                
    # print(f"accepted: {accept}")
    return accept/(nmax*nmax)
  
  

  
#=======================================================================
def main(program, nsteps, nmax, temp, pflag, threads):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting.
    threads (int) = number of threads
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """
    # Create and initialise lattice
    lattice = initdat(nmax)
    # Plot initial frame of lattice
    plotdat(lattice,pflag,nmax)
    # Create arrays to store energy, acceptance ratio and order parameter
    energy = np.zeros(nsteps+1,dtype=np.float64)
    ratio = np.zeros(nsteps+1,dtype=np.float64)
    order = np.zeros(nsteps+1,dtype=np.float64)
    # Set initial values in arrays
    energy[0] = all_energy_cythonised(lattice,nmax)
    ratio[0] = 0.5 # ideal value
    
    # only need to be calculated once (not in loop)
    delta = np.eye(2,2, dtype = np.float64)
    scale=0.1+temp
    factor = 2*nmax*nmax
    order[0] = get_order(lattice,nmax, delta, factor, threads)

    # Begin doing and timing some MC steps.
    initial = time.time()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(lattice,temp,nmax, scale, threads)
        energy[it] = all_energy_cythonised(lattice,nmax)
        order[it] = get_order(lattice, nmax, delta, factor, threads)
        
    # plot_reduced_e(energy, nsteps, temp)
    # plot_order(order, nsteps, temp)
    # plot_order_vs_temp(order, temp, nmax)
    final = time.time()
    runtime = final-initial
    
    # Final outputs
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
    # Plot final frame of lattice and generate output file
    # savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
    plotdat(lattice,pflag,nmax, True, energy, temp, order, nsteps)
    
    
#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 6:             ######### 6
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        THREADS = int(sys.argv[5])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, THREADS)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <THREADS>".format(sys.argv[0]))
        # print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================


    
