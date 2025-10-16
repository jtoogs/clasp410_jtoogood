#!/usr/bin/env python3

'''
Workspace for completing the homework titled "Neumann vs. Dirichlet"
Built around code from Dan's Github file "lab04_diffuse.py"
Includes lecture notes from 10.16 onwards
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
plt.ion()


def solve_heat(xstop=1., tstop=0.2, dx=0.02, dt=0.0002, c2=1, fx=None, lowerbound=None, upperbound=None):
    '''
    A function for solving the heat equation

    Parameters
    ----------
    xstop : float
        Furthest point in space to run the model
    tstop : float
        Furthest moment in time to run the model 
    dx : float 
        Spacial step size
    dt : float
        Time step 
    c2 : float
        c^2, the square of the diffusion coefficient.        
    fx : function, defaults to None 
        determines initial conditions #SOMETHING BWOKEN 
    lowerbound, upperbound : float, defaults to None 
        determines boundary conditions 
        Neumann boundary conditions use dU/dx=0 in this case 

    Returns
    -------
    x, t : 1D Numpy arrays
        Space and time values, respectively.
    U : Numpy array
        The solution of the heat equation, size is nSpace x nTime

    Comments
    ----------
    Lab04 uses dirichlet boundary conditions but the UB value changes as a function of time
    '''
    # Get grid sizes:
    N = int(tstop / dt)
    M = int(xstop / dx)

    # Set up space and time grid:
    t = np.linspace(0, tstop, N)
    x = np.linspace(0, xstop, M)

    # Create solution matrix; set initial conditions
    U = np.zeros([M, N])
    # U[:, 0] = fx
    U[:, 0] = 4*x - 4*x**2   #hardcoded override for hw       

    # Get our "r" coeff:
    r = c2 * (dt/dx**2)

    # announce boundary condition type
    if lowerbound is None: 
        print("Using Neumann (lower)")
    elif callable(lowerbound):
        print("Using Dirichlet (lower bound changes)")
    else: 
        print("Using Dirichlet (lower bound constant)")
    if upperbound is None: 
        print("Using Neumann (upper)")
    elif callable(upperbound): 
        print("Using Dirichlet (upper bound changes)")
    else:
        print("Using Dirichlet (upper bound constant)")

    # Solve our equation!
    for j in range(N-1):
        U[1:M-1, j+1] = (1-2*r) * U[1:M-1, j] + r*(U[2:M, j] + U[:M-2, j])
        if lowerbound is None: 
            U[0,j+1] = U[1,j+1] 
        elif callable(lowerbound):
            U[0,:] = lowerbound(t[j+1])
        else: 
            U[0,:] = lowerbound
        if upperbound is None: 
            U[-1,j+1] = U[-2,j+1]
        elif callable(upperbound): 
            U[-1,:] = upperbound(t[j+1])
        else:
            U[-1,:] = upperbound

    # Return our pretty solution to the caller:
    return t, x, U

def climate(t):
    '''t in secs, returns temp'''
    return 10 * np.sin(t/np.pi) #something wrong when passing this into solve_heat.. not sure why 

def plot_heatsolve(t, x, U, title=None, **kwargs):
    '''
    Plot the 2D solution for the `solve_heat` function.

    Extra kwargs handed to pcolor.

    Paramters
    ---------
    t, x : 1D Numpy arrays
        Space and time values, respectively.
    U : Numpy array
        The solution of the heat equation, size is nSpace x nTime
    title : str, default is None
        Set title of figure.

    Returns
    -------
    fig, ax : Matplotlib figure & axes objects
        The figure and axes of the plot.

    cbar : Matplotlib color bar object
        The color bar on the final plot
    '''

    # Check our kwargs for defaults:
    # Set default cmap to hot
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'hot'

    # Create and configure figure & axes:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Add contour to our axes:
    contour = ax.pcolor(t, x, U, **kwargs)
    cbar = plt.colorbar(contour)

    # Add labels to stuff!
    cbar.set_label(r'Temperature ($^{\circ}C$)')
    ax.set_xlabel('Time ($s$)')
    ax.set_ylabel('Position ($m$)')
    ax.set_title(title)

    fig.tight_layout()

    return fig, ax, cbar

t1,x1,U1 = solve_heat(lowerbound=0,upperbound=0)
t2,x2,U2 = solve_heat()

# Create and configure figure & axes:
fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(8, 8))

# Add contour to our axes:
contour = ax1.pcolor(t1, x1, U1,cmap="hot")
cbar = fig.colorbar(contour)

# Add labels to stuff!
cbar.set_label(r'Temperature ($^{\circ}C$)')
ax1.set_xlabel('Time ($s$)')
ax1.set_ylabel('Position ($m$)')
ax1.set_title('Dirichlet Boundary Conditions')

# Add contour to our axes:
contour = ax2.pcolor(t2, x2, U2,cmap="hot")
cbar = fig.colorbar(contour)

# Add labels to stuff!
cbar.set_label(r'Temperature ($^{\circ}C$)')
ax2.set_xlabel('Time ($s$)')
ax2.set_ylabel('Position ($m$)')
ax2.set_title('Neumann Boundary Conditions')

fig.tight_layout()