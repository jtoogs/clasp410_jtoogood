#!/usr/bin/env/python3
'''
Lab #4: Universality: Sick of Burning
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
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# turn on interactive plotting, set stylesheet
plt.ion()
plt.style.use('seaborn-v0_8-poster')

# declare global variables 
nx, ny = 3, 3 # Number of cells in X and Y direction.
prob_spread = 1.0 # Chance to spread to adjacent cells.
prob_bare = 0.0 # Chance of cell to start as bare patch.
prob_start = 0.0 # Chance of cell to start on fire.

# Create an initial grid, set all values to "2". dtype sets the value
# type in our array to integers only.
forest = np.zeros([ny, nx], dtype=int) + 2
# Set the center cell to "burning":
forest[1, 1] = 3

# Loop over every cell in the x and y directions.
for i in range(nx):
    for j in range(ny):
        # Roll our "dice" to see if we get a bare spot:
        if np.random.rand() < prob_bare:
            forest[j, i] = 1 # 1 is a bare spot.

# Create an array of randomly generated numbers of range [0, 1):
isbare = np.random.rand(ny, nx)
# Turn it into an array of True/False values:
isbare = isbare < prob_bare
# We can use this array of booleans to reference any existing array
# and change only the values corresponding to True:
forest[isbare] = 1

# Loop in the "x" direction:
for i in range(nx):
    # Loop in the "y" direction:
    for j in range(ny):
        ## Perform logic here:
        # forest[j, i] = #BE CAREFUL W EDGES 
        True


# Generate our custom segmented color map for this project.
# We can specify colors by names and then create a colormap that only uses
# those names. We have 3 funadmental states, so we want only 3 colors.
# Color info: https://matplotlib.org/stable/gallery/color/named_colors.html
forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
# Create figure and set of axes:
fig, ax = plt.subplots(1,1)

# Given our "forest" object, a 2D array that contains numbers 1, 2, or 3,
# Plot this using the "pcolor" method. Be sure to use our color map and
# set both *vmin* and *vmax*:
ax.pcolor(forest, cmap=forest_cmap, vmin=1, vmax=3)



def forest_fire(isize=3, jsize=3, nstep=4, pspread=1.0, pignite=0.0, pbare=0):
    '''
    Create a forest fire.

    Parameters
    ----------
    isize, jsize : int, defaults to 3
        Set size of forest in x and y direction, respectively.
    nstep : int, defaults to 4
        Set number of steps to advance solution.
    pspread : float, defaults to 1.0
        Set chance that fire can spread in any direction, from 0 to 1
        (i.e., 0% to 100% chance of spread.)
    pignite : float, defaults to 0.0
        Set the chance that a point starts the simulation on fire (or infected)
        from 0 to 1 (0% to 100%).
    pbare : float, defaults to 0.0
        Set the chance that a point starts the simulation on bare (or
        immune) from 0 to 1 (0% to 100%).
    '''

    # Creating a forest and making all spots have trees.
    forest = np.zeros((nstep, isize, jsize)) + 2

    # Set initial conditions for BURNING/INFECTED and BARE/IMMUNE
    # Start with BURNING/INFECTED:
    if pignite > 0:  # Scatter fire randomly:
        loc_ignite = np.zeros((isize, jsize), dtype=bool)
        while loc_ignite.sum() == 0:
            loc_ignite = rand(isize, jsize) <= pignite
        print(f"Starting with {loc_ignite.sum()} points on fire or infected.")
        forest[0, loc_ignite] = 3
    else:
        # Set initial fire to center:
        forest[0, isize//2, jsize//2] = 3

    # Set bare land/immune people:
    loc_bare = rand(isize, jsize) <= pbare
    forest[0, loc_bare] = 1

    # Loop through time to advance our fire.
    for k in range(nstep-1):
        # Assume the next time step is the same as the current:
        forest[k+1, :, :] = forest[k, :, :]
        # Search every spot that is on fire and spread fire as needed.
        for i in range(isize):
            for j in range(jsize):
                # Are we on fire?
                if forest[k, i, j] != 3:
                    continue
                # Ah! it burns. Spread fire in each direction.
                # Spread "up" (i to i-1)
                if (pspread > rand()) and (i > 0) and (forest[k, i-1, j] == 2):
                    forest[k+1, i-1, j] = 3
                # Spread "Down"
                # Spread "East"
                # Spread "West"

                # Change buring to burnt:
                forest[k+1, i, j] = 1

    return forest

def plot_progression(forest):
    '''Calculate the time dynamics of a forest fire and plot them.'''

    # Get total number of points:
    ksize, isize, jsize = forest.shape
    npoints = isize * jsize

    # Find all spots that have forests (or are healthy people)
    # ...and count them as a function of time.
    loc = forest == 2
    forested = 100 * loc.sum(axis=(1, 2))/npoints

    loc = forest == 1
    bare = 100 * loc.sum(axis=(1, 2))/npoints

    plt.plot(forested, label='Forested')
    plt.plot(bare, label='Bare/Burnt')
    plt.xlabel('Time (arbitrary units)')
    plt.ylabel('Percent Total Forest')


def question_1():
    '''
    Run this code to reproduce all results for question 1.

    Parameters: none

    Returns: none
    '''

def question_2():
    '''
    Run this code to reproduce all results for question 2.

    Parameters: none

    Returns: none
    '''

def question_3():
    '''
    Run this code to reproduce all results for question 3.

    Parameters: none

    Returns: none
    '''



print('Question 1:')
question_1()

print('Question 2:')
question_2()
 
print('Question 3:') 
question_3()



