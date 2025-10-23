#!/usr/bin/env/python3
'''
Lab #3: eat Diffusion & Permafrost
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

# turn on interactive plotting, set stylesheet
plt.ion()
plt.style.use('seaborn-v0_8-poster')

#####


# Get solution using your solver:
time, x, heat = your_fwddiff_solver(options)

# Create a figure/axes object
fig, axes = plt.subplots(1, 1)

# Create a color map and add a color bar.
map = axes.pcolor(time, x, heat, cmap='seismic', vmin=-25, vmax=25)
plt.colorbar(map, ax=axes, label='Temperature ($C$)')


# Set indexing for the final year of results:
loc = int(-365/dt) # Final 365 days of the result.
# Extract the min values over the final year:
winter = heat[:, loc:].min(axis=1)
# Create a temp profile plot:
fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
ax2.plot(winter, x, label='Winter')


# Kangerlussuaq average temperature:
t_kanger = np.array([-19.7, -21.0, -17., -8.4, 2.3, 8.4, 10.7, 8.5, 3.1, -6.0, -12.0, -16.9])


def temp_kanger(t):
    '''
    For an array of times in days, return timeseries of temperature for
    Kangerlussuaq, Greenland.
    '''
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(np.pi/180 * t - np.pi/2) + t_kanger.mean()

