#!/usr/bin/env/python3
'''
Lab #5: Snowball Earth 
Julian Toogood

This file solves a system of equations modeling Earth's temperature 
to explore the Snowball Earth hypothesis. 

Hi Sarah :) 
To reproduce the values and plots in my report, do this: 
- For question 1, run the function question_1() with no inputs.
- For question 2, run the function question_2() with no inputs.
- For question 3, run the function question_3() with no inputs.
- For question 4, run the function question_4() with no inputs.
Running this script will also call these functions in order. 

'''

# import packages 
import numpy as np 
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

# turn on interactive plotting, set stylesheet
plt.ion()
plt.style.use('seaborn-v0_8-poster')

# set constants:
radearth = 6357000.  # Earth radius in meters.
mxdlyr = 50.         # depth of mixed layer (m)
sigma = 5.67e-8      # Stefan-Boltzmann constant
C = 4.2e6            # Heat capacity of water
rho = 1020           # Density of sea-water (kg/m^3)

### FROM LECTURE ### 
def gen_grid(npoints=18):
    '''
    Create a evenly spaced latitudinal grid with `npoints` cell centers.
    Grid will always run from zero to 180 as the edges of the grid. This
    means that the first grid point will be `dLat/2` from 0 degrees and the
    last point will be `180 - dLat/2`.

    Parameters
    ----------
    npoints : int, defaults to 18
        Number of grid points to create.

    Returns
    -------
    dLat : float
        Grid spacing in latitude (degrees)
    lats : numpy array
        Locations of all grid cell centers.
    '''

    dlat = 180 / npoints  # Latitude spacing.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints)  # Lat cell centers.

    return dlat, lats

### FROM LAB ASSIGNMENT ###
def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.
    Parameters
    ----------
    lats_in : Numpy array
    Array of latitudes in degrees where temperature is required.
    0 corresponds to the south pole, 180 to the north.
    Returns
    -------
    temp : Numpy array
    Temperature in Celsius.
    '''
    # Set initial temperature curve
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
    23, 19, 14, 9, 1, -11, -19, -47])

    # Get base grid:
    npoints = T_warm.size
    dlat = 180 / npoints # Latitude spacing.
    lats = np.linspace(dlat/2., 180-dlat/2., npoints) # Lat cell centers.

    # Fit a parabola to the above values
    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting sampled at "lats".
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp

### FROM LECTURE - MODIFIED### 
def temp_const(lats_in, const_temp):
    '''
    Create a temperature profile for an earth with a constant temperature at all locations.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required
    const_temp : float
        Temperature to set at all locations 

    Returns
    -------
    temp : Numpy array
        Temperature in Celsius.
    '''

    # Make an array with a constant hot temp
    T_hot = const_temp
    temp = np.ones(lats_in.size) * T_hot

    return temp

### FROM LAB ASSIGNMENT ###
def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.
    Parameters
    ----------
    S0 : float
    Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
    Latitudes to output insolation. Following the grid standards set in
    the diffusion program, polar angle is defined from the south pole.
    In other words, 0 is the south pole, 180 the north.
    Returns
    -------
    insolation : numpy array
    Insolation returned over the input latitudes.
    '''
    # Constants:
    max_tilt = 23.5 # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    # Daily rotation of earth reduces solar constant by distributing the sun
    # energy all along a zonal band
    dlong = 0.01 # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation

### FROM LECTURE - MODIFIED### 
def snowball_earth(nlat=18, tfinal=10000, dt=1.0, lam=100., emiss=1.0,
                   init_cond=temp_warm, apply_spherecorr=False, albice=.6,
                   albgnd=.3, apply_insol=False, solar=1370, gamma=1.0):
    '''
    Solve the snowball Earth problem.

    Parameters
    ----------
    nlat : int, defaults to 18
        Number of latitude cells.
    tfinal : int or float, defaults to 10,000
        Time length of simulation in years.
    dt : int or float, defaults to 1.0
        Size of timestep in years.
    lam : float, defaults to 100
        Set ocean diffusivity
    emiss : float, defaults to 1.0
        Set emissivity of Earth/ground.
    init_cond : function, float, or array
        Set the initial condition of the simulation. If a function is given,
        it must take latitudes as input and return temperature as a function
        of lat. Otherwise, the given values are used as-is.
    apply_spherecorr : bool, defaults to False
        Apply spherical correction term
    apply_insol : bool, defaults to False
        Apply insolation term.
    solar : float, defaults to 1370
        Set level of solar forcing in W/m2
    albice, albgnd : float, defaults to .6 and .3
        Set albedo values for ice and ground.
    gamma : float, defaults to 1.0
        Set solar multiplier 

    Returns
    --------
    lats : Numpy array
        Latitudes representing cell centers in degrees; 0 is south pole
        180 is north.
    Temp : Numpy array
        Temperature as a function of latitude.
    '''

    # Set up grid:
    dlat, lats = gen_grid(nlat)
    # Y-spacing for cells in physical units:
    dy = np.pi * radearth / nlat

    # Create our first derivative operator.
    B = np.zeros((nlat, nlat))
    B[np.arange(nlat-1)+1, np.arange(nlat-1)] = -1
    B[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    B[0, :] = B[-1, :] = 0

    # Create area array:
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    # Get derivative of Area:
    dAxz = np.matmul(B, Axz)

    # Set number of time steps:
    nsteps = int(tfinal / dt)

    # Set timestep to seconds:
    dt = dt * 365 * 24 * 3600

    # Create insolation:
    insol = gamma * insolation(solar, lats)

    # Create temp array; set our initial condition
    Temp = np.zeros(nlat)
    if callable(init_cond):
        Temp = init_cond(lats)
    else:
        Temp += init_cond

    # Create our K matrix:
    K = np.zeros((nlat, nlat))
    K[np.arange(nlat), np.arange(nlat)] = -2
    K[np.arange(nlat-1)+1, np.arange(nlat-1)] = 1
    K[np.arange(nlat-1), np.arange(nlat-1)+1] = 1
    # Boundary conditions:
    K[0, 1], K[-1, -2] = 2, 2
    # Units!
    K *= 1/dy**2

    # Create L matrix.
    Linv = np.linalg.inv(np.eye(nlat) - dt * lam * K)

    # Set initial albedo.
    albedo = np.zeros(nlat)
    loc_ice = Temp <= -10  # Sea water freezes at ten below.
    albedo[loc_ice] = albice
    albedo[~loc_ice] = albgnd

    # SOLVE!
    for istep in range(nsteps):
        # Update Albedo:
        loc_ice = Temp <= -10  # Sea water freezes at ten below.
        albedo[loc_ice] = albice
        albedo[~loc_ice] = albgnd

        # Create spherical coordinates correction term
        if apply_spherecorr:
            sphercorr = (lam*dt) / (4*Axz*dy**2) * np.matmul(B, Temp) * dAxz
        else:
            sphercorr = 0

        # Apply radiative/insolation term:
        if apply_insol:
            radiative = (1-albedo)*insol - emiss*sigma*(Temp+273)**4
            Temp += dt * radiative / (rho*C*mxdlyr)

        # Advance solution.
        Temp = np.matmul(Linv, Temp + sphercorr)

    return lats, Temp

### FROM LECTURE - UNUSED### 
def test_functions():
    '''Test our functions'''

    print('Test gen_grid')
    print('For npoints=5:')
    dlat_correct, lats_correct = 36.0, np.array([18., 54., 90., 126., 162.])
    result = gen_grid(5)
    if (result[0] == dlat_correct) and np.all(result[1] == lats_correct):
        print('\tPassed!')
    else:
        print('\tFAILED!')
        print(f"Expected: {dlat_correct}, {lats_correct}")
        print(f"Got: {gen_grid(5)}")

def save_this_plot(fig, filename, folder="Lab05_results/"):
    '''
    Given an already-made figure, save it as an image to the specified folder 
    
    Parameters
    --------
    fig : figure object 
        The already-made figure to save
    filename : string
        Name for the image 
    folder : string, defaults to "Lab05_results/"
        Defines path to place output images in
    
    Returns: None
    '''

    # Check to see if folder exists, if not, make it!
    if not os.path.exists(folder):
        os.mkdir(folder)

    # navigate to folder containing plots 
    os.chdir(folder)

    # Make a buncha plots.
    print(f"\tSaving plot: {filename}")
    fig.savefig(f"{filename}.png")
    plt.close()

    # return to original directory 
    os.chdir("..")

### FROM LECTURE - MODIFIED### 
def question_1():
    '''
    Run this code to reproduce all results for question 1.

    Parameters: none

    Returns: none
    '''

    # Set initial conditions 
    dlat, lats = gen_grid()
    temp_init = temp_warm(lats)

    # Get solution after 10K years for each combination of terms:
    lats, temp_diff = snowball_earth()
    lats, temp_sphe = snowball_earth(apply_spherecorr=True)
    lats, temp_alls = snowball_earth(apply_spherecorr=True, apply_insol=True,
                                     albice=.3)

    # Generate plot 
    fig, ax = plt.subplots(1, 1)
    ax.plot(lats-90, temp_init, color="dodgerblue", label='Initial Condition')
    ax.plot(lats-90, temp_diff, color="orangered", label='Basic Diffusion')
    ax.plot(lats-90, temp_sphe, color="gold", label='Diff. + Spherical Correction')
    ax.plot(lats-90, temp_alls, color="forestgreen", label='Diff. + SphCorr + Radiative')
    ax.set_title('Solution after 10,000 Years')
    ax.set_ylabel(r'Temp ($^{\circ}C$)')
    ax.set_xlabel('Latitude')
    ax.legend(loc='best')

    save_this_plot(fig,'Lab05_Q1_1')

def question_2():
    '''
    Run this code to reproduce all results for question 2.

    Parameters: none

    Returns: none
    '''

    # Set initial conditions 
    dlat, lats = gen_grid()
    temp_init = temp_warm(lats)

    # Attempt several model runs
    lats, temp_lowL = snowball_earth(apply_spherecorr=True, apply_insol=True, albice=.3, lam=30., emiss=0.72) 
    lats, temp_highL = snowball_earth(apply_spherecorr=True, apply_insol=True, albice=.3, lam=120., emiss=0.72) 
    lats, temp_lowE = snowball_earth(apply_spherecorr=True, apply_insol=True, albice=.3, lam=62., emiss=0.5) 
    lats, temp_highE = snowball_earth(apply_spherecorr=True, apply_insol=True, albice=.3, lam=62., emiss=0.9) 
    lats, temp_best = snowball_earth(apply_spherecorr=True, apply_insol=True, albice=.3, lam=62., emiss=0.72) # matches except at boundaries 

    # Generate plots for all model runs 
    fig, ax = plt.subplots(1, 1)
    ax.plot(lats-90, temp_lowL, color="darkgreen", linestyle='dashed', label='$\lambda=30, \epsilon=0.72$ | Low Diff.')
    ax.plot(lats-90, temp_highL, color="lightgreen", linestyle='dashed', label='$\lambda=120, \epsilon=0.72$ | High Diff.')
    ax.plot(lats-90, temp_lowE, color="darkviolet", linestyle='dashed', label='$\lambda=62, \epsilon=0.5$ | Low Emiss.')
    ax.plot(lats-90, temp_highE, color="violet", linestyle='dashed', label='$\lambda=62, \epsilon=0.9$ | High Emiss.')
    ax.plot(lats-90, temp_init, color="dodgerblue", label='Warm-Earth Equilibrium (given)')
    ax.plot(lats-90, temp_best, color="orangered", label='$\lambda=62, \epsilon=0.72$ | Best Result')
    ax.set_title('Solution after 10,000 Years: Constant Albedo')
    ax.set_ylabel(r'Temp ($^{\circ}C$)')
    ax.set_xlabel('Latitude')
    ax.legend(loc='best')

    save_this_plot(fig,'Lab05_Q2_1')

def question_3():
    '''
    Run this code to reproduce all results for question 3.

    Parameters: none

    Returns: none
    '''

    # Set initial conditions 
    dlat, lats = gen_grid()
    temp_init = temp_warm(lats)
    temp_hot = temp_const(lats,60.)
    temp_cold = temp_const(lats,-60.)

    # Run models 
    lats, hot_equil = snowball_earth(apply_spherecorr=True, apply_insol=True, init_cond=temp_hot, lam=62., emiss=0.72)
    lats, cold_equil = snowball_earth(apply_spherecorr=True, apply_insol=True, init_cond=temp_cold, lam=62., emiss=0.72)
    lats, flash_freeze = snowball_earth(apply_spherecorr=True, apply_insol=True, init_cond=temp_init, albgnd=.6, lam=62., emiss=0.72)

    # Generate plots for all model runs 
    fig, ax = plt.subplots(1, 1)
    ax.plot(lats-90, temp_init, color="dodgerblue", label='Warm-Earth Equilibrium (given)')
    ax.plot(lats-90, hot_equil, color="orangered", linestyle='dashed', label='Hot initial conditions (60$^{\circ}C$)')
    ax.plot(lats-90, cold_equil, color="lightskyblue", linestyle='dashed', label='Cold initial conditions (-60$^{\circ}C$)')
    ax.plot(lats-90, flash_freeze, color="grey", linestyle='dashed', label='Flash Freeze (albedo=0.6)')
    ax.set_title('Solution after 10,000 Years: Climate Extremes')
    ax.set_ylabel(r'Temp ($^{\circ}C$)')
    ax.set_xlabel('Latitude')
    ax.legend(loc='best')

    save_this_plot(fig,'Lab05_Q3_1')

def question_4(): 
    '''
    Run this code to reproduce all results for question 4.

    Parameters: none

    Returns: none
    '''
    # Set initial conditions 
    dlat, lats = gen_grid()
    temp_cold = temp_const(lats,-60.)

    # Determine variables and preallocate output array 
    gammas = np.concatenate((np.arange(0.4,1.45,0.05), np.arange(1.35,0.35,-0.05)),axis=0)
    avg_temps = np.zeros(gammas.size)

    # Run models 
    lats, current_temp = snowball_earth(apply_spherecorr=True, apply_insol=True, init_cond=temp_cold, lam=62., emiss=0.72)
    for i in range(gammas.size):
        lats, current_temp = snowball_earth(apply_spherecorr=True, apply_insol=True, init_cond=current_temp, gamma=gammas[i], lam=62., emiss=0.72)
        avg_temps[i] = np.mean(current_temp)

    # Generate plots for all model runs 
    fig, ax = plt.subplots(1, 1)
    ax.plot(gammas, avg_temps, color="dodgerblue")
    ax.set_title('Average Global Temperature vs. Solar Multiplier')
    ax.set_ylabel(r'Average Temp ($^{\circ}C$)')
    ax.set_xlabel('Solar Multiplier ($\gamma$)')

    save_this_plot(fig,'Lab05_Q4_1')

# clear workspace 
plt.close('all')

print('Question 1:')
question_1()

print('Question 2:')
question_2()
 
print('Question 3:') 
question_3()

print('Question 4:') 
question_4()