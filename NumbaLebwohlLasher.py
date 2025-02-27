"""
Numba vectorised Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python NumbaLebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <THREADS>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  THREADS = number of parallel threads to run simulation with.
  
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

import numba as nb


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
    ax.set_title(f"Reduced Temperature, T* = {temp}")
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
    
    
 
#=======================================================================
@nb.njit
def calculate_energies_dummy(angle: float) -> float:
  """
  Calculate energy from angle

  Args:
      angle (float): angle of cell

  Returns:
      float: energy of cell
  """
  en = 0.5*(1.0 - 3.0*np.cos(angle)*np.cos(angle))
  return en


@nb.vectorize([nb.float64(nb.float64)], target='parallel')
def calculate_energies(angle: float) -> float:
  """
  calculate energies of an array

  Args:
      angle (float): angle of cell

  Returns:
      float: energy of cell
  """
  en = calculate_energies_dummy(angle)
  return en
  

#=======================================================================
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
@nb.njit
def calc_sum(ens0: float, ens1: float, ens2: float, ens3: float) -> float:
  """
  calculate sum of energies of 4 adjacent cells

  Args:
      ens0 (float): energy of adjacent cell
      ens1 (float): energy of adjacent cell
      ens2 (float): energy of adjacent cell
      ens3 (float): energy of adjacent cell

  Returns:
      float: reduced energy
  """
  en = ens0 + ens1 + ens2 + ens3
  return en


@nb.vectorize([nb.float64(nb.float64, nb.float64, nb.float64, nb.float64)], target='parallel')
def sum_ens(ens0,ens1,ens2,ens3):
  """
  calculate reduced energy of cell

  Args:
      ens0 (float): energy of adjacent cell
      ens1 (float): energy of adjacent cell
      ens2 (float): energy of adjacent cell
      ens3 (float): energy of adjacent cell

  Returns:
      float: reduced energy
  """
  en = calc_sum(ens0,ens1,ens2,ens3)
  return en

 
#=======================================================================
def one_energy_vectorised(arr):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
	Returns:
	  en (float) = reduced energy of cell.
    """
    en = np.zeros(arr.shape, dtype=np.float64)
    
    angs = np.zeros((4,arr.shape[0], arr.shape[0]), dtype=np.float64)
    angs = calc_angles(arr)
    
    ens = calculate_energies(angs)
    en = sum_ens(ens[0], ens[1], ens[2], ens[3])
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
@nb.njit
def get_order_calc(lab, delta, norm_val):
    """
    calculate order of system

    Args:
        lab (arr(2*nmax*nmax)): 2D unit vector for each cell     
        delta (arr): 2x2 identity matrix
        norm_val (float): = 2*nmmax*nmax

    Returns:
        float(2,2): order?
    """
    Qab = np.zeros((2,2)) 
    
    lab_square = (3*lab*lab)
    lab_01_sum = np.sum(3*lab[0]*lab[1])
    
    # calculate separately to reduce time
    Qab[0,0] = np.sum(lab_square[0] - delta[0,0])
    Qab[0,1] = Qab[1,0] = lab_01_sum 
    Qab[1,1] = np.sum(lab_square[1] - delta[1,1])
    
    Qab = Qab/(norm_val)
    
    return Qab

  
#=======================================================================
def get_order(arr,nmax, norm_val, delta):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
      norm_val (float): = 2*nmmax*nmax
      delta (arr): 2x2 identity matrix
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    # Qab = np.zeros((3,3))
    
    #
    # Generate a 2D unit vector for each cell (i,j) and
    # put it in a (2,i,j) array.

    lab = np.vstack((np.cos(arr),np.sin(arr))).reshape(2,nmax,nmax)
    
    Qab = get_order_calc(lab, delta, norm_val)

    eigenvalues = np.linalg.eig(Qab)[0]
    return eigenvalues.max()
  

#=======================================================================
@nb.vectorize([nb.float64(nb.float64, nb.float64)], target='parallel') 
def calc_boltz(arr, Ts):
  """
  calculate boltzmann constant

  Args:
      arr (float): element of energy difference
      Ts (float): temperature of system

  Returns:
      _type_: _description_
  """
  boltz = np.exp( -(arr) / Ts )
  return boltz
  
  
#=======================================================================
@nb.njit
def accept_torf(diff: float, boltz: float, rand_arr: float):
    """
    determine if angle should be accepted or not 

    Args:
        diff (float): energy difference
        boltz (float): boltzmann factor value
        rand_arr (float): random value between 0 and 1

    Returns:
        int: 0 if accepted, 1 if rejected
    """
    # accept
    if (diff <= 0) | ( (diff > 0) & (boltz >= rand_arr) ):
      return 0
    # reject
    else:
      return 1
    
    
@nb.vectorize([nb.int64(nb.float64, nb.float64, nb.float64)], target='parallel')
def calc_accepted(diff, boltz, rand_arr):
  """
  determine if angle should be accepted or not 

  Args:
        diff (float): energy difference
        boltz (float): boltzmann factor value
        rand_arr (float): random value between 0 and 1

  Returns:
      int: 0 if accepted, 1 if rejected
  """
  accept_bool = accept_torf(diff, boltz, rand_arr)
  return accept_bool
  
  
#=======================================================================
def MC_step(arr,Ts,scale,nmax, checkerboards ):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
	  scale (float) = scale for tem;
      nmax (int) = side length of square lattice.
	  checkerboards(list(arr)) = list of checkerboards: ( white squares, black squars)
   
    Description:
      Function to perform one MC step.  Function returns the acceptance
      ratio for information.  This is the fraction of attempted changes
      that are successful.  Generally aim to keep this around 0.5 for
      efficient simulation.
	Returns:
	  accept/(nmax**2) (float) = acceptance ratio for current MCS.
    """
   

    # Calculate current energy of each cell
    # LONG
    en0 = one_energy_vectorised(arr)
    
    # Change each cell by a random angle
    aran = np.random.normal(scale=scale, size=(nmax,nmax))
   
    rand_arr = np.random.uniform(0.0, 1.0, (arr.shape))
    
        
    num_accepted = 0
    for board in checkerboards:
      
      # Change each cell by a random angle
      arr += aran*board
      
      # calculate new energy of each cell, using the old angles for the adjacent cells
      en1 = one_energy_vectorised(arr)
      
      # calculate difference in energy
      diff = en1 - en0
      
      boltz = calc_boltz(diff, Ts)
      
      # accept new arrangement if energy is lower OR energy is higher and boltz calculation is greater than a random number between 0 and 1
      accepted = (calc_accepted(diff, boltz, rand_arr))
      
      arr -= aran*board*accepted
    
    num_accepted = np.sum(accepted)
    
    return num_accepted/(nmax*nmax)
  
  

  
  
  
#=======================================================================
def main(program, nsteps, nmax, temp, pflag, threads):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting.
	  threads (int) = number of threads to run script with
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """
    
    nb.set_num_threads(threads)
    numthreads = nb.get_num_threads()
    print(f"THREADS: {numthreads}")
    
    
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
    plotdat(lattice,pflag,nmax, True, energy, temp, order, nsteps)
    
    
#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 6:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        THREADS = int(sys.argv[5])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, THREADS)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <THREADS>".format(sys.argv[0]))
#=======================================================================
