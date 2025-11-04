#!/usr/bin/env/python3

import numpy as np
import matplotlib.pyplot as plt 

plt.ion()

def forest_fire(isize=3, jsize=3, nstep=4, pspread=1.0):
    '''
    Create a forest fire model.

    Parameters
    ----------
    isize, jsize : int, defaults to 3
        Set size of forest in x and y direction, respectively
    nstep : int, defaults to 4
        Set number of time steps 
    pspread : float, defaults to 1.0
        Set chance (from 0.0 to 1.0, ie 0%-100%) that fire can spread in any direction

    Returns
    ----------

    '''

    # Creating a forest and making all spots have trees.
    forest = np.zeros((nstep, isize, jsize)) + 2 #note odd indexing order in 3+ dimensions: [k,i,j]

    # 1=bare, 2=untouched 3=fire
    # Set initial fire 
    forest[0,isize//2,jsize//2] = 3 #hardcoded, bad - update for lab

    for n in range(nstep):
        # debug 
        print(f'Timestep: {n}')

        # duplicate forest for modification 
        if n+1 < nstep:
            forest[n+1,:,:] = forest[n,:,:] 
            
        # test fire
        for i in range(isize):
            for j in range(jsize):
                if not (forest[n,i,j] == 3):
                    continue 
                else:
                    print(f"FUCK theres a fire at {[i,j]}")
                    if n+1<nstep:
                        forest[n+1,i,j] = 1
                    prob = np.random.rand()
                    if pspread>prob and n+1<nstep: #determine likelihood of spread
                        if i+1<isize and j+1<jsize and i+1==2 and j+1==2:
                            forest[n+1,i+1,j] = 3 
                        if i+1<isize and j+1<jsize and i+1==2 and j+1==2:
                            forest[n+1,i-1,j] = 3 
                        if i+1<isize and j+1<jsize and i+1==2 and j+1==2:
                            forest[n+1,i,j+1] = 3 
                        if i+1<isize and j+1<jsize and i+1==2 and j+1==2:
                            forest[n+1,i,j-1] = 3 
    
    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    contour = ax.pcolor(range(isize),range(jsize),forest[nstep-1,:,:])
    cbar = plt.colorbar(contour)
    plt.show()

    return forest 


# discussion of monte carlo simulation
## random numbers aren't great once other factors are involved 

