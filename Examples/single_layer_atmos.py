#!/usr/bin/env/python3
'''
This script contains functions for exploring a single-layer atmosphere model.
'''

import numpy as np 
import matplotlib.pyplot as plt

# change stylesheet
plt.style.use('fivethirtyeight')
plt.ion()

# Declare constants 
sigma = 5.67E-8 # stefan boltzmann constant 

# other needed values 
year = np.array([1900, 1950, 2000])
s0 = np.array([1365.0, 1366.5, 1368.0]) # solar forcing in W/m2
t_anom = np.array([-0.4, 0.0, 0.4])


def temp_1layer(s0=1350.0, albedo=0.33, epsilon=1.0):
    ''' 
    Given solar forcing (s0) and albedo, determine the temperature of the Earth's surface 
    using a single layer perfectly absorbing energy balanced atmosphere model. 

    Parameters
    ----------
    s0 : float, defaults to 1350.0 
        Solar forcing in W/m2
    albedo : float, defaults to 0.33
        Albedo (reflection coefficient) for earth's surface 
    epsilon : float,m default to 1.0
        set absorptivity/emissivity of the atmosphere
    
    Return 
    --------
    te = float
        Temperature at the Earth's surface 
    '''

    te = (s0  * (1-albedo) / ((2*sigma)*(2-epsilon)))**(1/4.)

    return te

def compare_warming():
    '''
    Create a figure to test if changes in solar driving can account for climate change.
    '''
    
    t_model = temp_1layer(s0=s0)
    t_obs = t_model[1] + t_anom

    fig, ax = plt.subplots(1,1,figsize=(8,8))

    ax.plot(year,t_obs,label="Observed Temperature Change")
    ax.plot(year,t_model,label="Predicted Temperature Change")

    ax.legend(loc='best')
    ax.set_xlabel('Year')
    ax.set_ylabel('Surface Temperature ($K$)')

    fig.tight_layout()
