#!/usr/bin/env/python3 
'''
Solve the coffee problem to learn how to drink coffee effectively. 
'''

import numpy as np 
import matplotlib.pyplot as plt 
plt.ion() 

print("i DRINK a lot of COFFEE")
print("i SHIT a lot of POOP")

def solve_temp(t, T_init=90, T_env=20, k=1/300.):
    '''
    This function returns temperature as a function of time using Newton's Law of Cooling.

    Parameters 
    --------
    t: Numpy array 
        An array of time values in seconds.
    T_init: float, defaults to 90
        Initial temperature in Celsius. 
    T_env: float, defaults to 20
        Ambient air temperature in Celsius.
    k: float, defaults to 1/300
        Heat transfer coefficient in 1/s.

    Returns 
    --------
    T_coffee: Numpy array 
        Temperature corresponding to time t. 

    '''
    T_coffee = T_env + (T_init-T_env) * np.exp(-k*t)
    return T_coffee

# t = np.arange(0,600.0,0.5)
