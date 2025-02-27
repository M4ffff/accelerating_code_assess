"""
Numpy vectorised Python Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python NumpyVecLebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation. 
A checkerboard method is used to update the array in each timestep. 


"""

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
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        
        # HERE - CYTHONISE OR MORE EFFICIENT WAY OF CALCULATING MIN/MAX
        cols = one_energy_vectorised(arr)
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
    
    # WHY IS THIS SET EQUAL TO q?
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
    
    # if doing a plot of cell, also show how energy and order vary over course of simulation
    if final_data and pflag != 0:
      plot_reduced_e(energy, nsteps, temp)
      plot_order(order, nsteps, temp)
      
        
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
    
   
   
def calc_angles(arr):
    """
    calculate angles between cell and adjacent cells

    Args:
        arr (float(nmax,nmax)) = array that contains lattice data;

    Returns:
        (float(4)): array that contains angle between cell and each adjacent cell
    """

    ang1 = arr - np.roll(arr, -1, axis=1)
    ang2 = arr - np.roll(arr, 1, axis=1)
    ang3 = arr - np.roll(arr, -1, axis=0)
    ang4 = arr - np.roll(arr, 1, axis=0)

    angs = np.array([ang1, ang2, ang3, ang4])
    
    return angs
    

#=======================================================================
def one_energy_vectorised(arr):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;

    Description:
      Function that computes the energy of every cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
	Returns:
	  en (arr(nmax*nmax)) = reduced energy of cell.
    """
    en = np.zeros(arr.shape, dtype=np.float64)

    # need to change to checkerboard
    angs = calc_angles(arr)
    
    # SLOW?
    ens = 0.5*(1.0 - 3.0*np.cos(angs)**2)
    en = ens[0] + ens[1] + ens[2] + ens[3]

    return en
    
    
  
#=======================================================================
def all_energy(arr):
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
    enall = np.sum(one_energy_vectorised(arr))
    return enall
  

  
#=======================================================================
def get_order(arr, nmax, norm_val, delta):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    Qab = np.zeros((3,3), dtype=np.float64)
    
    lab = np.vstack((np.cos(arr),np.sin(arr))).reshape(2,nmax,nmax)
    
    lab_square = (3*lab*lab)
    lab_01_sum = np.sum(3*lab[0]*lab[1])
    
    # calculate separately to reduce time
    Qab[0,0] = np.sum(lab_square[0] - delta[0,0])
    Qab[0,1] = Qab[1,0] = lab_01_sum 
    Qab[1,1] = np.sum(lab_square[1] - delta[1,1])
    Qab[2,2] = 0 - delta[2,2]
    
    Qab = Qab/(norm_val)
    
    eigenvalues = np.linalg.eig(Qab)[0]
    return eigenvalues.max()
  
#=======================================================================
def MC_step(arr,Ts,scale,nmax, checkerboards ):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
      checkerboards(list(arr)) = list of checkerboards: ( white squares, black squars)
    Description:
      Function to perform one MC step. Function returns the acceptance
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
    

    # Calculate current energy of each cell
    en0 = one_energy_vectorised(arr)
    
    aran = np.random.normal(scale=scale, size=(nmax,nmax))
    
    num_accepted = 0
    for board in checkerboards:
      
      # Change each cell by a random angle
      arr += aran*board
      
      # calculate new energy of each cell, using the old angles for the adjacent cells
      en1 = one_energy_vectorised(arr)
      
      # calculate difference in energy
      diff = en1 - en0
      
      rand_arr = np.random.uniform(0.0, 1.0, (diff.shape))
      boltz = np.exp( -(diff) / Ts )
      
      # accept new arrangement if energy is lower OR energy is higher and boltz calculation is greater than a random number between 0 and 1
      accepted = (diff <= 0) | ( (diff > 0) & (boltz >= rand_arr) )
      # print(arr[accepted].shape)
      arr[~accepted] -= aran[~accepted]*board[~accepted]
      
    num_accepted += np.sum(accepted)
    # print(num_accepted)
    

    return num_accepted/(nmax*nmax)
  
  
#=======================================================================
def main(program, nsteps, nmax, temp, pflag):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting.
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
    energy[0] = all_energy(lattice)

    ratio[0] = 0.5 # ideal value
    
    # take calculation outside loop
    norm_val = 2*nmax*nmax
    delta = np.eye(3,3)
    order[0] = get_order(lattice,nmax,norm_val,delta)
    
    checkerboard1 = ( np.indices((nmax,nmax))[0] + np.indices((nmax,nmax))[1] ) % 2 
    checkerboard2 = np.where((checkerboard1==0)|(checkerboard1==1), checkerboard1^1, checkerboard1)
    checkerboards = [checkerboard1, checkerboard2]
    scale=0.1+temp
    # Begin doing and timing some MC steps.
    initial = time.time()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(lattice,temp,scale,nmax, checkerboards)
        energy[it] = all_energy(lattice)
        order[it] = get_order(lattice,nmax,norm_val, delta)

    final = time.time()
    runtime = final-initial
    
    # Final outputs
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
    # Plot final frame of lattice and generate output file
    # savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
    plotdat(lattice,pflag,nmax)
    
    
#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================
