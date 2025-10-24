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
# from tabulate import tabulate #uncomment for table 1b formatting

# turn on interactive plotting, set stylesheet
plt.ion()
plt.style.use('seaborn-v0_8-poster')

# define global variables
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4, 10.7, 8.5, 3.1, -6.0, -12.0, -16.9]) # Kangerlussuaq average temperature

def validation_initial(x):
    '''
    Function defining the initial conditions for validating solve_heat

    Parameters
    ---------
    x : numpy array
        Array of positions at which to define the starting temperature values
    
    Returns
    ---------
    temp : numpy array
        Array of starting temperature values corresponding to U[:,0]
    '''
    temp = 4*x - 4*x**2
    return temp

def temp_kanger(t,warming=0):
    '''
    For an array of times in YEARS, return timeseries of temperature for
    Kangerlussuaq, Greenland.

    Parameters
    ----------
    t : numpy array
        Array of times in years 
    warming : float, defaults to 0
        Uniform temperature shift (degrees C)
    
    Returns
    ----------
    temp : numpy array
        Array of temperature values corresponding to each time in t
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()

    temp = t_amp*np.sin(np.pi/180 * (t*365.25) - np.pi/2) + t_kanger.mean() + warming

    return temp

def zeros_initial(x):
    '''
    Function defining uniform 0 degrees C initial conditions for solve_heat

    Parameters
    ---------
    x : numpy array
        Array of positions at which to define the starting temperature values
    
    Returns
    ---------
    temp : numpy array
        Array of starting temperature values corresponding to U[:,0]
    '''
    temp = np.zeros(np.size(x))
    return temp

def solve_heat(xstop=1., tstop=0.2, dx=0.02, dt=0.0002, c2=1, initial=None, lowerbound=None, upperbound=None, warming=0, suppressoutput=False):
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
        determines initial conditions 
        Must accept an array of positions and return temperature at those
        positions as an equally sized array.
    lowerbound, upperbound : float or function, defaults to None 
        determines boundary conditions 
        Neumann boundary conditions use dU/dx=0 in this case 
    warming : float, defaults to 0
        Uniform temperature shift (degrees C) to pass to temp_kanger
    suppressoutput : boolean, defaults to False
        determines whether to announce boundary condition type

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
    U[:, 0] = initial(x)

    # Get our "r" coeff:
    r = c2 * (dt/dx**2)

    # announce boundary condition type
    if suppressoutput is not True:
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
            U[0,j+1] = lowerbound(t[j+1],warming)
        else: 
            U[0,:] = lowerbound
        if upperbound is None: 
            U[-1,j+1] = U[-2,j+1]
        elif callable(upperbound): 
            U[-1,j+1] = upperbound(t[j+1],warming)
        else:
            U[-1,:] = upperbound

    # Return our pretty solution to the caller:
    return t, x, U

def question_1(plot=False):
    '''
    Run this code to reproduce all results for question 1.

    Parameters
    ---------
    plot : boolean, defaults to False   
        Determines whether to plot the heat equation solution (for debugging)

    Returns: none
    '''
    # Get solution using your solver:
    time, x, heat = solve_heat(xstop=1., tstop=0.2, dx=0.2, dt=0.02, c2=1, initial=validation_initial, 
                               upperbound=0, lowerbound=0, suppressoutput=True)

    if plot: 
        # Create a figure/axes object
        fig, ax = plt.subplots(1, 1)

        # Create a color map and add a color bar.
        map = ax.pcolor(time, x, heat, cmap='hot') #, vmin=-25, vmax=25)
        plt.colorbar(map, ax=ax, label='Temperature ($C$)')

        # label axes, add title 
        ax.set_xlabel('Time $(seconds)$')
        ax.set_ylabel('Position $(meters)$')
        ax.set_title(f'Validation of Diffusion Equation Solver')

    print("Heat Equation Solution for Validation: ")
    # print(tabulate(heat)) #uncomment for table 1b
    print(heat)

def question_2():
    '''
    Run this code to reproduce all results for question 2.

    Parameters: none

    Returns: none
    '''
    dt = 5/365 # timestep in years

    time,x,heat = solve_heat(xstop=100., tstop=75, dx=1, dt=dt, c2=0.25/1000/1000*60*60*24*365, initial=zeros_initial, 
                          lowerbound=temp_kanger, upperbound=5, suppressoutput=True)
        
    # Create a figure/axes object
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=[20,8])

    # Create a color map and add a color bar.
    map = ax1.pcolor(time, x, heat, cmap='seismic',vmin=-25, vmax=25)
    plt.colorbar(map, ax=ax1, label='Temperature ($^{o}C$)')

    # label axes, add title 
    ax1.invert_yaxis() 
    ax1.set_xlabel('Time $(years)$')
    ax1.set_ylabel('Depth $(meters)$')
    ax1.set_title(f'Permafrost Simulation for Kangerlussuaq, Greenland')


    # Set indexing for the final year of results:
    loc = int(-1/dt) # Final 365 days of the result.

    # Extract the extreme values over the final year:
    winter = heat[:, loc:].min(axis=1)
    summer = heat[:, loc:].max(axis=1)

    # Create a temp profile plot:
    ax2.plot(winter, x, 'b-', label='Winter')
    ax2.plot(summer, x, 'r--', label='Summer')

    # label axes, add title
    ax2.set_xlim([-8,6])
    ax2.set_ylim([-5,75])
    ax2.invert_yaxis()
    ax2.grid(True)
    ax2.set_xlabel('Temperature $(^{o}C)$')
    ax2.set_ylabel('Depth $(meters)$')
    ax2.set_title(f'Seasonal Temperature Profile for Kangerlussuaq, Greenland')
    ax2.legend(loc='lower left')

    fig.suptitle("No Warming",fontsize=30)

    print("With an initial condition of 0 ℃, we find that a steady state is reached after roughly 70 years of the model running. " \
    "At 75 years, we find by inspecting our plot that the active layer reaches from the surface to a depth of 2 meters, " \
    "while the permafrost layer reaches from the bottom of the active layer to a depth of 51 meters.")

def question_3():
    '''
    Run this code to reproduce all results for question 3.

    Parameters: none

    Returns: none
    ''' 
    for i in [0.5, 1, 3]: 

        dt = 5/365 # timestep in years

        time,x,heat = solve_heat(xstop=100., tstop=75, dx=1, dt=dt, c2=0.25/1000/1000*60*60*24*365, initial=zeros_initial, 
                            lowerbound=temp_kanger, upperbound=5, warming=i, suppressoutput=True)
            
        # Create a figure/axes object
        fig, [ax1,ax2] = plt.subplots(1, 2, figsize = [20,8])

        # Create a color map and add a color bar.
        map = ax1.pcolor(time, x, heat, cmap='seismic',vmin=-25, vmax=25)
        plt.colorbar(map, ax=ax1, label='Temperature ($^{o}C$)')

        # label axes, add title 
        ax1.invert_yaxis() 
        ax1.set_xlabel('Time $(years)$')
        ax1.set_ylabel('Depth $(meters)$')
        ax1.set_title(f'Permafrost Simulation for Kangerlussuaq, Greenland')

        # Set indexing for the final year of results:
        loc = int(-1/dt) # Final 365 days of the result.

        # Extract the extreme values over the final year:
        winter = heat[:, loc:].min(axis=1)
        summer = heat[:, loc:].max(axis=1)

        # Create a temp profile plot:
        ax2.plot(winter, x, 'b-', label='Winter')
        ax2.plot(summer, x, 'r--', label='Summer')

        # label axes, add title
        ax2.set_xlim([-8,6])
        ax2.set_ylim([-5,75])
        ax2.invert_yaxis()
        ax2.grid(True)
        ax2.set_xlabel('Temperature $(^{o}C)$')
        ax2.set_ylabel('Depth $(meters)$')
        ax2.set_title(f'Seasonal Temperature Profile for Kangerlussuaq, Greenland')
        ax2.legend(loc='lower left')

        fig.suptitle(f"Warming Scenario: +{i}$^{{o}}C$",fontsize=30)

    print("The permafrost layer depth decreases for each increase in temperature, from 51 m with no warming " \
    "to 50, 49, and 45 m for 0.5, 1, and 3 degrees Celsius, respectively. The active layer depth increases "
    "for each increase in temperature, but much less significantly at 3 ℃ of warming, the active layer " \
    "depth reaches roughly 4 m. ")

print('Question 1:')
question_1()

print('Question 2:')
question_2()
 
print('Question 3:') 
question_3()
