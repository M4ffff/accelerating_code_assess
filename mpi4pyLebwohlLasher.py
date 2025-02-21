"""
Basic Python Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

SH 16-Oct-23
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
def plotdat(arr,pflag,nmax):
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
    
    # WHY IS THIS SET EQUAL TO q?
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
    
    
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
	  en (float) = reduced energy of cell.
    """
    en = 0.0
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #
#
# Add together the 4 neighbour contributions
# to the energy
#

# HERE
# PARALLELISE
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return en
  
  
#=======================================================================
def one_energy_mpi(current_row, above_row, below_row,ix,nmax):
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
    en = 0.0
#
# Add together the 4 neighbour contributions
# to the energy
#

# HERE
# PARALLELISE
    ang = current_row[ix]-current_row[(ix+1)%nmax]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = current_row[ix]-current_row[(ix-1)%nmax]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = current_row[ix]-above_row[ix]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = current_row[ix]-below_row[ix]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return en
  
  
  
  
#=======================================================================
def all_energy(arr,nmax):
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
    enall = 0.0
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall
  
  
#=======================================================================
def get_order(arr,nmax):
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
    Qab = np.zeros((3,3))
    delta = np.eye(3,3)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    Qab = Qab/(2*nmax*nmax)
    eigenvalues = np.linalg.eig(Qab)[0]
    return eigenvalues.max()
  
  
def update(current_row, above_row, below_row, aran, nmax, Ts):
    accept = 0
    for iy in range(nmax):
        ang = aran[iy]
        en0 = one_energy_mpi(current_row, above_row, below_row,iy,nmax)
        current_row[iy] += ang
        en1 = one_energy_mpi(current_row, above_row, below_row,iy,nmax)
        if en1<=en0:
            accept += 1
        else:
        # Now apply the Monte Carlo test - compare
        # exp( -(E_new - E_old) / T* ) >= rand(0,1)
            boltz = np.exp( -(en1 - en0) / Ts )

            if boltz >= np.random.uniform(0.0,1.0):
                accept += 1
            else:
                current_row[iy] -= ang
                
    return accept
  
  
  
#=======================================================================
def MC_step(arr,Ts,nmax, comm):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
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
    scale=0.1+Ts
    accept = 0
    aran = np.random.normal(scale=scale, size=(nmax,nmax))
    # print("0")
  
    MAXWORKER  = 17          # maximum number of worker tasks
    MINWORKER  = 1          # minimum number of worker tasks
    BEGIN      = 1          # message tag
    LTAG       = 2          # message tag
    RTAG       = 3          # message tag
    NONE       = 0          # indicates no neighbour
    DONE       = 4          # message tag
    MASTER     = 0          # taskid of first process

        # u = np.zeros((2,NXPROB,NYPROB))        # array for grid

# First, find out my taskid and how many tasks are running
    numtasks = comm.Get_size()
    numworkers = numtasks-1
    # print("1")


    # print("Starting mpi script with %d worker tasks." % numworkers)

# Distribute work to workers.  Must first figure out how many rows to
# send and what to do with extra rows.
    averow = nmax // numworkers
    extra = nmax % numworkers
    offset = 0
    for i in range(1,numworkers+1):
        rows = averow
        if i <= extra:
            rows+=1

    # Tell each worker who its neighbors are, since they must exchange
    # data with each other.
        above = (i - 1) % numworkers
        below = (i + 1) % numworkers

    # Now send startup information to each worker 
        # print("sending")
        comm.send(offset, dest=i, tag=BEGIN)
        comm.send(rows, dest=i, tag=BEGIN)
        comm.send(above, dest=i, tag=BEGIN)
        comm.send(below, dest=i, tag=BEGIN)
        comm.Send([arr[offset:offset+rows], MPI.DOUBLE], dest=i, tag=BEGIN)
        comm.Send([aran[offset:offset+rows], MPI.DOUBLE], dest=i, tag=BEGIN)
        # print("sending 2")
        offset += rows
        
        
    # print("all workers sent")
# Now wait for results from all worker tasks 
# NEED TO DO
    for i in range(1,numworkers+1):
        offset = comm.recv(source=i, tag=DONE)
        rows = comm.recv(source=i, tag=DONE)
        local_accept = comm.recv(source=i, tag=DONE)
        comm.Recv([arr[offset:offset+rows], MPI.DOUBLE], source=i, tag=DONE)
        accept += int(local_accept)

    # print(f"accept: {accept}")
    return accept/(nmax*nmax)
    # End of master code
    

    
    
      
      
      
  
  
  
  
  
  
  
  
  
  
  
  
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
    MINWORKER  = 1          # minimum number of worker tasks
    BEGIN      = 1          # message tag
    LTAG       = 2          # message tag
    RTAG       = 3          # message tag
    NONE       = 0          # indicates no neighbour
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
            
        print("running main")
        # Create and initialise lattice
        lattice = initdat(nmax)
        for i in range(1, numworkers+1):
            comm.Send([lattice, MPI.DOUBLE], dest=i, tag=BEGIN)
            comm.send(temp, dest=i, tag=BEGIN)
            comm.send(nsteps, dest=i, tag=BEGIN)
        
        print("lattice and temp sent")
        
        # Plot initial frame of lattice
        plotdat(lattice,pflag,nmax)
        # Create arrays to store energy, acceptance ratio and order parameter
        energy = np.zeros(nsteps+1,dtype=np.float64)
        ratio = np.zeros(nsteps+1,dtype=np.float64)
        order = np.zeros(nsteps+1,dtype=np.float64)
        # Set initial values in arrays
        energy[0] = all_energy(lattice,nmax)
        ratio[0] = 0.5 # ideal value
        order[0] = get_order(lattice,nmax)

        # Begin doing and timing some MC steps.
        initial = MPI.Wtime()
        for it in range(1,nsteps+1):
            ####################################################################################################################
            ratio[it] = MC_step(lattice,temp,nmax, comm)
            ####################################################################################################################
            energy[it] = all_energy(lattice,nmax)
            order[it] = get_order(lattice,nmax)
        final = MPI.Wtime()
        runtime = final-initial
        
        # Final outputs
        print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
        # Plot final frame of lattice and generate output file
        # savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
        plotdat(lattice,pflag,nmax)
    
    
    
    #************************* workers code **********************************/
    elif taskid != MASTER:
      
        lattice = np.zeros((nmax, nmax))
        comm.Recv([lattice, MPI.DOUBLE], source=MASTER, tag=BEGIN)
      
        Ts = comm.recv(source=MASTER, tag=BEGIN)
        nsteps = comm.recv(source=MASTER, tag=BEGIN)
      
        # print("lattice and temp received")
        # print(f"Ts: {Ts}")
        # print(f"lattice shape: {lattice.shape}")
      
        # print("entering worker")
    # Array is already initialized to zero - including the borders
    # Receive my offset, rows, neighbors and grid partition from master
        for i in range(nsteps):
            offset = comm.recv(source=MASTER, tag=BEGIN)
            rows = comm.recv(source=MASTER, tag=BEGIN)
            above = comm.recv(source=MASTER, tag=BEGIN)
            below = comm.recv(source=MASTER, tag=BEGIN)
            # print(f"below: {below}")
            comm.Recv([lattice[offset:offset+rows], MPI.DOUBLE], source=MASTER, tag=BEGIN)
            
            
            aran_rows = np.zeros_like(lattice[offset:offset+rows])
            comm.Recv([aran_rows, MPI.DOUBLE], source=MASTER, tag=BEGIN)
            
            # print(f"Worker {taskid}: Received aran!")
            
            min_rows = offset
            max_rows = rows+offset
        
            
            start = min_rows
            # if min_rows % 2 == 0:
            #     print("do even rows first")
            # else:
            #     print("do odd rows first")
            #     start += 1

        # Determine border elements.  Need to consider first and last columns.
        # Obviously, row 0 can't exchange with row 0-1.  Likewise, the last
        # row can't exchange with last+1.
        # DELETE
            # start=offset
            # end=offset+rows
            # if offset==0:
            #     start=1
            # if (offset+rows)==nmax:
            #     end-=1

        # Begin doing STEPS iterations.  Must communicate border rows with
        # neighbours.  If I have the first or last grid row, then I only need
        # to  communicate with one neighbour
        
            above_row_recieved = np.zeros_like(lattice[0])
            below_row_recieved = np.zeros_like(lattice[0])
        
            # print(f"above: {above}")
        
            if above % 2 == 0:
                # print("send first row above")
                # send first row above
                req=comm.Isend([lattice[offset], MPI.DOUBLE], dest=above, tag=RTAG)
                
                # print("receive first row from below")
                comm.Irecv([below_row_recieved, MPI.DOUBLE], source=above, tag=LTAG)
                
                        #####
                # send last row below
                # print("send last row below")
                req=comm.Isend([lattice[offset+rows-1], MPI.DOUBLE], dest=below, tag=LTAG)
            
                # print("receive last row from above")
                comm.Irecv([above_row_recieved, MPI.DOUBLE], source=below, tag=RTAG)
                
            else:
                # print("receive first row from below")
                comm.Irecv([below_row_recieved, MPI.DOUBLE], source=above, tag=LTAG)
                
                # print("send first row above")
                # send first row above
                req=comm.Isend([lattice[offset], MPI.DOUBLE], dest=above, tag=RTAG)
                
                # print("receive last row from above")
                comm.Irecv([above_row_recieved, MPI.DOUBLE], source=below, tag=RTAG)
                
                # send last row below
                # print("send last row below")
                req=comm.Isend([lattice[offset+rows-1], MPI.DOUBLE], dest=below, tag=LTAG)
            
                
            
            # #####
            # # send last row below
            # print("send last row below")
            # req=comm.Isend([arr[offset+rows-1], MPI.DOUBLE], dest=below, tag=LTAG)
        
            # print("receive last row from above")
            # comm.IRecv([above_row_recieved, MPI.DOUBLE], source=below, tag=RTAG)
        
        
            # print("entering rows iteration")
            for row in range(start, max_rows):
                above_row = np.zeros_like(lattice[row])
                below_row = np.zeros_like(lattice[row])
                if row == min_rows:
                    
                    # receive row from above neighbour
                    
                    above_row = above_row_recieved
                else:
                    above_row = lattice[row-1]
                    
                if row == max_rows - 1:
                    
                    # receive row from below neighbour
                    below_row = below_row_recieved
                else:
                    below_row = lattice[row+1]
                    
                aran_row = aran_rows[row-offset]
                        
            # Now call update to update the value of grid points

                # update(start,end,nmax,u[iz],u[1-iz]);

                # print("entering update")
                accept = update(lattice[row], above_row, below_row, aran_row, nmax, Ts)

                # print("leaving update")
                
            
            # if min_rows % 2 == 0:
            #     start += 1
            # else:
            #     start -= 1
            


        # Finally, send my portion of final results back to master
            comm.send(offset, dest=MASTER, tag=DONE)
            comm.send(rows, dest=MASTER, tag=DONE)
            comm.send(accept, dest=MASTER, tag=DONE)
            comm.Send([lattice[offset:offset+rows], MPI.DOUBLE], dest=MASTER, tag=DONE)
        
        

    
    
    
    
    
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
