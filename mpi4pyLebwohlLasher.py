"""
Basic Python Lebwohl-Lasher code made multi threaded using MPI.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

mpiexec -n <THREADS> python mpi4pyLebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation. 

"""

import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import sys



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
    
    # if doing a plot of cell, also show how energy and order vary over course of simulation
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
    energy += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ixl,iy]
    energy += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyu]
    energy += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyd]
    energy += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return energy
  
  
#=======================================================================
def all_energy(arr,rows,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
    rows (int): number of rows in workers section of array
      nmax (int) = side length of square lattice.
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
	Returns:
	  enall (float) = reduced energy of lattice.
    """
    enall = 0.0
    for i in range(1,rows+1):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall
  
  
#=======================================================================
def get_order(arr,rows,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
    rows (int): number of rows in workers section of array
      nmax (int) = side length of square lattice.
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
	Returns:
	  max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    Qab = np.zeros((3,3))
    delta = np.eye(3,3)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,rows,nmax)
    for a in range(3):
        for b in range(3):
            for i in range(rows):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    Qab = Qab/(2*rows*nmax) 
    eigenvalues = np.linalg.eig(Qab)[0]
    return eigenvalues.max()
  
  
#=======================================================================
def MC_step(arr, Ts,rows, nmax, comm, above, below, LTAG, RTAG):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
    rows (int): number of rows in workers section of array
      nmax (int) = side length of square lattice.
      comm (?) = connection for MPI
      above (int) = worker above label
      below (int) = worker below label
      LTAG (int) = sending above tag
      RTAG (int) = receiving from below tag
    Description:
      Function to perform one MC step, which consists of an average
      of 1 attempted change per lattice site.  Function returns the acceptance
      ratio for information.  This is the fraction of attempted changes
      that are successful.  Generally aim to keep this around 0.5 for
      efficient simulation.
      
      Sends first row to worker above once updated
	Returns:
	  accept/(nmax**2) (float) = acceptance ratio for current MCS.
    """

    scale=0.1+Ts
    accept = 0
    xran = np.random.randint(1,high=rows+1, size=(nmax,nmax))
    yran = np.random.randint(0,high=nmax, size=(nmax,nmax))
    aran = np.random.normal(scale=scale, size=(nmax,nmax))
    for i in range(1, rows+1):
        if i == 3:
            # receive first row from below - needed in calculation of last line
            # i set to 3 to ensure time for first row below to be calculated
            comm.Irecv([arr[rows+1], MPI.DOUBLE], source=below, tag=LTAG)
        for j in range(nmax):
            # pick random x coordinate
            ix = xran[i,j]
            # pick random y coordinate
            iy = yran[i,j]
            # pick random angle
            ang = aran[i,j]

            # old_energy
            en0 = one_energy(arr,ix,iy,nmax)
            
            # new energy
            arr[ix,iy] += ang
            en1 = one_energy(arr,ix,iy,nmax)
                
                
                
            if en1<=en0:
                accept += 1
            else:
            # Now apply the Monte Carlo test - compare
            # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                boltz = np.exp( -(en1 - en0) / Ts )

                if boltz >= np.random.uniform(0.0,1.0):
                    accept += 1
                else:
                    arr[ix,iy] -= ang
                    
        if i == 1:
            # send freshly calculated first row to worker above
            req=comm.Isend([arr[i], MPI.DOUBLE], dest=above, tag=RTAG)
            
    return accept/(nmax*nmax)
  


        
        
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
    
    MAXWORKER  = 17          # maximum number of worker tasks
    
    # need multiple workers otherwise sending rows doesnt work.
    MINWORKER  = 2          # minimum number of worker tasks
    BEGIN      = 1          # message tag
    LTAG       = 2          # message tag
    RTAG       = 3          # message tag
    DONE       = 4          # message tag
    MASTER     = 0          # taskid of first process
    
    comm = MPI.COMM_WORLD
    taskid = comm.Get_rank()
    numtasks = comm.Get_size()
    numworkers = numtasks-1
    
    #************************* master code *******************************/
    if taskid == MASTER:
            # Check if numworkers is within range - quit if not
        if (numworkers > MAXWORKER) or (numworkers < MINWORKER):
            print("ERROR: the number of tasks must be between %d and %d." % (MINWORKER+1,MAXWORKER+1))
            print("Quitting...")
            comm.Abort()
            
        # Create and initialise lattice
        lattice = initdat(nmax)
        
        # Plot initial frame of lattice
        plotdat(lattice,pflag,nmax)
        
        # Create arrays to store energy, acceptance ratio and order parameter
        ratio = np.zeros(nsteps+1,dtype=np.float64)
        energy = np.zeros(nsteps+1,dtype=np.float64)
        order = np.zeros(nsteps+1,dtype=np.float64)
        

        # Distribute work to workers.  Must first figure out how many rows to
        # send and what to do with extra rows.
        averow = nmax // numworkers
        extra = nmax % numworkers
        offset = 0
        
        initial = MPI.Wtime()
        
        for i in range(1,numworkers+1):
            rows = averow
            if i <= extra:
                rows+=1

        # Tell each worker who its neighbors are, since they must exchange
        # data with each other.
            above = (i - 1) % numworkers
            below = (i + 1) % numworkers

            # send importatn information
            comm.send(offset, dest=i, tag=BEGIN)
            comm.send(rows, dest=i, tag=BEGIN)
            comm.send(above, dest=i, tag=BEGIN)
            comm.send(below, dest=i, tag=BEGIN)

            comm.send(temp, dest=i, tag=BEGIN)
            comm.send(nsteps, dest=i, tag=BEGIN)
            comm.send(nmax, dest=i, tag=BEGIN)
            
            # roll lattice up/ down so that first/lst rows are in block being sent 
            if offset==0:
                lattice = np.roll(lattice, 1, axis=0)
                comm.Send(([lattice[offset-1:offset+rows+1, :], (rows+2)*nmax, MPI.DOUBLE]), dest=i, tag=BEGIN)
                # unroll
                lattice = np.roll(lattice, -1, axis=0)
            elif offset+rows == nmax:
                lattice = np.roll(lattice, -1, axis=0)
                comm.Send(([lattice[offset-1:offset+rows+1, :], (rows+2)*nmax, MPI.DOUBLE]), dest=i, tag=BEGIN)
                # unroll
                lattice = np.roll(lattice, 1, axis=0)
            else:
                # send segment of lattice
                comm.Send([lattice[offset-1:offset+rows+1, :], (rows+2)*nmax, MPI.DOUBLE], dest=i, tag=BEGIN)
            
            offset += rows
          
        # receive data from each worker
        for i in range(1,numworkers+1):
            offset = comm.recv(source=i, tag=DONE)
            rows = comm.recv(source=i, tag=DONE)
            comm.Recv(lattice[offset:offset+rows, :], source=i, tag=DONE)
        
        
        # Reduce values from each worker into single result 
        comm.Reduce(MPI.IN_PLACE, ratio, op=MPI.SUM, root=MASTER)
        comm.Reduce(MPI.IN_PLACE, energy, op=MPI.SUM, root=MASTER)
        comm.Reduce(MPI.IN_PLACE, order, op=MPI.SUM, root=MASTER)
        
        
        final = MPI.Wtime()
        runtime = final-initial
        
        # Final outputs
        print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
        # Plot final frame of lattice and generate output file
        # savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
        plotdat(lattice,pflag,nmax, True, energy, temp, order, nsteps)
        
    #************************* workers code **********************************/
    elif taskid != MASTER:
        
        # receive relevatn info
        offset = comm.recv(source=MASTER, tag=BEGIN)
        rows = comm.recv(source=MASTER, tag=BEGIN)
        above = comm.recv(source=MASTER, tag=BEGIN)
        below = comm.recv(source=MASTER, tag=BEGIN)
        
        temp = comm.recv(source=MASTER, tag=BEGIN)
        nsteps = comm.recv(source=MASTER, tag=BEGIN)
        nmax = comm.recv(source=MASTER, tag=BEGIN)
        
        # receive section of array with extra 2 rows to store above/below rows
        local_lattice = np.zeros((rows+2,nmax), dtype=np.float64)
        comm.Recv([local_lattice, (rows+2)*nmax, MPI.DOUBLE], source=MASTER, tag=BEGIN)
        
        
        # ratio, energy and order values to be filled by this worker
        worker_ratio = np.zeros(nsteps+1, dtype=np.float64)
        worker_energy = np.zeros(nsteps+1, dtype=np.float64)
        worker_order = np.zeros(nsteps+1, dtype=np.float64)
        
        
        # Set initial values in arrays
        worker_energy[0] = all_energy(local_lattice,rows,nmax)
        worker_ratio[0] = 0.5 # ideal value
        worker_order[0] = get_order(local_lattice[1:-1,:],rows,nmax)
        
        # calculate new within each step
        for it in range(1,nsteps+1):
            # first row sent up once updated by each worker in MC_step function
            worker_ratio[it] = MC_step(local_lattice, temp, rows, nmax, comm, above, below, LTAG, RTAG)
            worker_energy[it] = all_energy(local_lattice,rows,nmax)
            worker_order[it] = get_order(local_lattice[1:-1,:],rows,nmax)
        # return ratio, energy, order
            
          # send relevant results back to MASTER
        comm.send(offset, dest=MASTER, tag=DONE)
        comm.send(rows, dest=MASTER, tag=DONE)
            
        # send lattice back
        comm.Send(local_lattice[1:-1, :], dest=MASTER, tag=DONE)

        # Send values back to master
        comm.Reduce(worker_ratio, None, op=MPI.SUM, root=MASTER)
        comm.Reduce(worker_energy, None, op=MPI.SUM, root=MASTER)
        comm.Reduce(worker_order, None, op=MPI.SUM, root=MASTER)


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
