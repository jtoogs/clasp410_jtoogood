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
import matplot

# declare physical constants
sigma = 5.67E-8 # Stefan-Boltzmann constant (W/m2/K-4)

def n_layer_atmos(nlayers, epsilon=1, albedo=0.33, s0=1350.0, debug=false)
    '''
    String of doc 
    '''

    # Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    # Populate based on our model:
    for i in range(nlayers+1):
    for j in range(nlayers+1):
    A[i, j] = # What math should go here?
    b = # What should go here?

    # Invert matrix:
    Ainv = np.linalg.inv(A)
    # Get solution:
    fluxes = np.matmul(Ainv, b) # Note our use of matrix multiplication!

    if debug:
        print(f'A[i={i},j={j}] = {A[i, j]}')





# known limitations for discussion section: 
# climate change, dynamic energy exchange within system (e.g. weather)