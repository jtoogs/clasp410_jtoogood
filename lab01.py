#!/usr/bin/env/python3
'''
Lab #1: Energy Balance Atmosphere Model
Julian Toogood

This file solves the N-layer atmosphere problem for Lab 01 and all subparts. 

HI SARAH :) TO REPRODUCE THE VALUES AND PLOTS IN MY REPORT, DO THIS:
- TBD

'''

# import packages 
import numpy as np 
import matplotlib.pyplot as plt

# declare physical constants
sigma = 5.67E-8 # Stefan-Boltzmann constant (W/m2/K-4)

def n_layer_atmos(nlayers, epsilon=1, albedo=0.33, s0=1350.0, debug=False):
    '''
    String of doc 
    '''

    # Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    # Populate based on our model:
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i==j: 
                A[i, j] = -2+1 * (j==0) #diagonal 
            else:
                A[i,j] = epsilon**(i>0) * (1-epsilon)**(np.abs(j-i)-1)
    
    b[0] = 0.25 * s0 * (1-albedo) # longwave is only the first E_in

    # Invert matrix:
    Ainv = np.linalg.inv(A)
    # Get solution:
    fluxes = np.matmul(Ainv, b) # Note our use of matrix multiplication!

    # turn fluxes into temperatures
    temps = (fluxes * (1/sigma))**(1/4) # this is not right 

    # return temperatures to caller 
    return temps 

    # verify 
    print(temps)

    if debug:
        print(f'A[i={i},j={j}] = {A[i, j]}')
        print(f'{A}')





# known limitations for discussion section: 
# climate change, dynamic energy exchange within system (e.g. weather)