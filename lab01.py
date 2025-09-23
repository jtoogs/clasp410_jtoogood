#!/usr/bin/env/python3
'''
Lab #1: Energy Balance Atmosphere Model
Julian Toogood

This file solves the N-layer atmosphere problem for Lab 01 and all subparts. 

Hi Sarah :) 
To reproduce the values and plots in my report, do this: 
- TBD 1,2 
- For question 3, run the function question_3() with no inputs.
- For question 4, run the function question_4() with no inputs.
- For question 5, run the function question_5() with no inputs.

'''

# import packages 
import numpy as np 
import matplotlib.pyplot as plt

# turn on interactive plotting, set stylesheet
plt.ion()
plt.style.use('fivethirtyeight')

# declare physical constants
sigma = 5.67E-8 # Stefan-Boltzmann constant (W/m2/K-4)

def n_layer_atmos(nlayers, epsilon=1.0, albedo=0.33, s0=1350.0, debug=False):
    '''
    Solve the N-layer atmosphere problem and return an array of temperatures
    with indices corresponding to each layer. 

    Parameters
    ----------
    nlayers : int
        Number of atmosphere layers 
    epsilon : float, default=1.0
        Emissivity of atmosphere layers
    albedo : float, default=0.33
        Planetary albedo
    s0 : float, default=1350.0
        Incoming solar shortwave flux in W/m^2
    debug : boolean, default=False
        Activates intermediate print steps to check for incorrect values

    Returns
    ---------
    temp : Numpy array, size nlayers+1
        Array of temperatures at the Earth's surface (layer 0) and each of nlayers
        atmosphere layers
    '''

    # Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    # confirm correct array size 
    if debug:
        print(f'Array size: {A.shape}')

    # Populate based on our model:
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i==j: 
                A[i, j] = -2 + 1*(j==0) #diagonal 
            else:
                # A[i,j] = epsilon**(i>0) * (1-epsilon)**(np.abs(j-i)-1)
                A[i,j] = epsilon * (1-epsilon)**(np.abs(j-i)-1)
                
    
    b[0] = -0.25 * s0 * (1-albedo) # longwave is only the first E_in
    A[0,1:] /= epsilon # correct first row 

    # check matrix A 
    if debug:
        print(A)

    # Invert matrix:
    Ainv = np.linalg.inv(A)

    # Get solution:
    fluxes = np.matmul(Ainv, b) # Note our use of matrix multiplication!

    # turn fluxes into temperatures
    temps = (fluxes * (1/sigma) * (1/epsilon))**(1/4) 
    temps[0] = (fluxes[0] * 1/sigma)**(1/4) # surface is a blackbody 
    
    # verify 
    if debug:
        print(fluxes)
        print(temps)

    # return temperatures to caller 
    return temps 

def question_3(debug=False):
    '''
    Run this code to reproduce all results for question 3.

    Parameters
    ----------
    debug : boolean, default=False
        Activates intermediate print steps to check for incorrect values

    Returns: none
    '''

    # calculate temps for varying emissivity 
    e = np.arange(0.1,1.1,0.1)
    exp1Temps = np.zeros(e.size)
    for i in np.arange(0,e.size):
        exp1Temps[i] = n_layer_atmos(1,epsilon=e[i])[0]

    # calculate temps for varying nlayers 
    n = np.arange(1,6)
    exp2Temps = np.zeros(n.size)
    for j in np.arange(0,n.size):
        exp2Temps[j] = n_layer_atmos(n[j], epsilon=0.255)[0]

    # confirm correct values 
    if debug:
        print(f'Emissivity array: {e}')
        print(f'nlayers array: {n}')
        print(f'Exp1 temp results: {exp1Temps}')
        print(f'Exp2 temp results: {exp2Temps}')
    
    # print exp2 results for presentation
    if not debug:
        print('How many layers of atmosphere are required to produce a surface temperature of ~288K?')
        print(f'1 layer: {round(exp2Temps[0],2)} Kelvin')
        print(f'2 layers: {round(exp2Temps[1],2)} Kelvin')
        print(f'3 layers: {round(exp2Temps[2],2)} Kelvin')
        print(f'4 layers: {round(exp2Temps[3],2)} Kelvin')
        print(f'5 layers: {round(exp2Temps[4],2)} Kelvin')
        print('5 layers of atmosphere are required to produce a surface temperature of ~288 Kelvin.')

    # plot 
    fig, [ax1, ax2] = plt.subplots(2,1)
    ax1.plot(e,exp1Temps)
    ax2.plot(n,exp2Temps)

    plt.axhline(y=288, color='r', linestyle='--', label='288 K')

    fig.suptitle('Earth Surface Temperature Dependence')
    ax1.set_xlabel('Emissivity ($\epsilon$)')
    ax1.set_ylabel('Surface Temperature ($K$)')
    ax2.set_xlabel('Number of Layers ($n$)')
    ax2.set_ylabel('Surface Temperature ($K$)')
    ax2.set_xticks([1,2,3,4,5])
    ax2.legend(loc='best')

    fig.tight_layout()

def question_4(debug=False):
    '''
    Run this code to reproduce all results for question 4.

    Parameters
    ----------
    debug : boolean, default=False
        Activates intermediate print steps to check for incorrect values

    Returns: none
    '''
    temps = np.zeros(2)
    idx = 0

    # for increasing index from 2 to an arbitrarily large non infinite number
    for i in np.arange(2,100):
        # calculate surface T for #index layers 
        tempAtIdx = n_layer_atmos(i, epsilon=1.0, albedo=0.7, s0=2600.0)[0]
        if debug:
            print(f'Surface temperature at index {i}: {tempAtIdx}')
        # test whether surface T is greater than 700K 
        if tempAtIdx > 700:
            temps[0] = n_layer_atmos(i-1, epsilon=1.0, albedo=0.7, s0=2600.0)[0]
            temps[1] = tempAtIdx 
            idx = i
            break

    # print this #index and the previous 
    print(f'Surface temperature exceeds 700 Kelvin at {idx} layers with temperature {round(temps[1],2)} Kelvin.')
    print(f'With one fewer layers ({idx-1}), surface temperature is {round(temps[0],2)} Kelvin.')

def n_layer_atmos_nuked(nlayers, epsilon=1.0, albedo=0.33, s0=1350.0, debug=False):

    '''
    Solve the N-layer atmosphere problem for a special case where 
    the top layer of the atmosphere absorbs all incoming solar flux 
    and return an array of temperatures with indices corresponding 
    to each layer. 

    Parameters
    ----------
    nlayers : int
        Number of atmosphere layers 
    epsilon : float, default=1.0
        Emissivity of atmosphere layers
    albedo : float, default=0.33
        Planetary albedo - unused in calculation 
    s0 : float, default=1350.0
        Incoming solar shortwave flux in W/m^2
    debug : boolean, default=False
        Activates intermediate print steps to check for incorrect values

    Returns
    ---------
    temp : Numpy array, size nlayers+1
        Array of temperatures at the Earth's surface (layer 0) and each of nlayers
        atmosphere layers
    '''

    # Create array of coefficients, an N+1xN+1 array:
    A = np.zeros([nlayers+1, nlayers+1])
    b = np.zeros(nlayers+1)

    # confirm correct array size 
    # if debug:
        # print(f'Array size: {A.shape}')

    # Populate based on our model:
    for i in range(nlayers+1):
        for j in range(nlayers+1):
            if i==j: 
                A[i, j] = -2 + 1*(j==0) #diagonal 
            else:
                # A[i,j] = epsilon**(i>0) * (1-epsilon)**(np.abs(j-i)-1)
                A[i,j] = epsilon * (1-epsilon)**(np.abs(j-i)-1)
                
    
    b[0] = 0.0 # no shortwave reaches surface 
    b[nlayers] = -(1/4)*s0 # all shortwave absorbed/re-emitted by top layer 
    A[0,1:] /= epsilon # correct first row 

    # check matrix A, b
    if debug:
        # print(A)
        print(b)

    # Invert matrix:
    Ainv = np.linalg.inv(A)

    # Get solution:
    fluxes = np.matmul(Ainv, b) # Note our use of matrix multiplication!

    # turn fluxes into temperatures
    temps = (fluxes * (1/sigma) * (1/epsilon))**(1/4) 
    temps[0] = (fluxes[0] * 1/sigma)**(1/4) # surface is a blackbody 
    
    # verify 
    if debug:
        print(fluxes)
        print(temps)

    # return temperatures to caller 
    return temps 

def question_5(debug=False):
    '''
    Run this code to reproduce all results for question 5.

    Parameters
    ----------
    debug : boolean, default=False
        Activates intermediate print steps to check for incorrect values

    Returns: none
    '''

    temps = n_layer_atmos_nuked(5,epsilon=0.5, albedo=0.33,s0=1350)
    print(f'Earth\'s surface temperature under a nuclear winter scenario: {round(temps[0],2)} Kelvin')

    if debug:
        print(temps)
    
    fig, ax = plt.subplots(1,1)
    ax.plot(temps,np.arange(0,6))
    fig.suptitle('Earth vertical temperature profile, nuclear winter scenario')
    ax.set_ylabel('atmosphere level')
    ax.set_yticks([0,1,2,3,4,5])
    ax.set_yticklabels(['surface',1,2,3,4,5])
    ax.set_xlabel('$K$')


# known limitations for discussion section: 
# climate change, dynamic energy exchange within system (e.g. weather)