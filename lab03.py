#!/usr/bin/env/python3
'''
Lab #3: Heat Diffusion & Permafrost
Julian Toogood

This file solves the 1D diffusion equation for Lab 03 and all subparts. 

Hi Sarah :) 
To reproduce the values and plots in my report, do this: 
- For question 1, run the function question_1() with no inputs.
- For question 2, run the function question_2() with no inputs.
- For question 3, run the function question_3() with no inputs.
Running this script will also call these functions in order. 

'''

# import packages 
import numpy as np 
import matplotlib.pyplot as plt
from tabulate import tabulate #remove for submission

# turn on interactive plotting, set stylesheet
plt.ion()
plt.style.use('seaborn-v0_8-poster')

#####

# define global variables
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4, 10.7, 8.5, 3.1, -6.0, -12.0, -16.9]) # Kangerlussuaq average temperature

def solve_heat(xstop=1., tstop=0.2, dx=0.02, dt=0.0002, c2=1, initial=None, lowerbound=None, upperbound=None):
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
    initial : function, defaults to None 
        determines initial conditions #SOMETHING BWOKEN 
        Must accept an array of positions and return temperature at those
        positions as an equally sized array.
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
    Lab03 uses dirichlet boundary conditions but the UB value changes as a function of time
    '''

    # Check our stability criterion:
    dt_max = dx**2 / (2*c2)
    if dt > dt_max:
        raise ValueError(f'DANGER: dt={dt} > dt_max={dt_max}.')
    
    # Get grid sizes:
    N = int(tstop / dt) + 1
    M = int(xstop / dx) + 1

    # Set up space and time grid:
    t = np.linspace(0, tstop, N)
    x = np.linspace(0, xstop, M)

    # Create solution matrix; set initial conditions
    U = np.zeros([M, N])
    # U[:, 0] = initial
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

def temp_kanger(t):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()

    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean()

def question_1():
    '''
    Run this code to reproduce all results for question 1.

    Parameters: none

    Returns: none
    '''
    # Get solution using your solver:
    time, x, heat = solve_heat(xstop=1., tstop=0.2, dx=0.2, dt=0.02, c2=1, initial=None, upperbound=0,lowerbound=0)

    # Create a figure/axes object
    fig, axes = plt.subplots(1, 1)

    # Create a color map and add a color bar.
    map = axes.pcolor(time, x, heat, cmap='hot') #, vmin=-25, vmax=25
    plt.colorbar(map, ax=axes, label='Temperature ($C$)')

    print(tabulate(heat))

def question_2():
    '''
    Run this code to reproduce all results for question 2.

    Parameters: none

    Returns: none
    '''
    # Set indexing for the final year of results:
    loc = int(-365/dt) # Final 365 days of the result.

    # Extract the min values over the final year:
    winter = heat[:, loc:].min(axis=1)

    # Create a temp profile plot:
    fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    ax2.plot(winter, x, label='Winter')

def question_3():
    '''
    Run this code to reproduce all results for question 3.

    Parameters: none

    Returns: none
    ''' 


print('Question 1:')
question_1()

print('Question 2:')
#question_2()

print('Question 3:')
#question_3()



