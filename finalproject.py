#!/usr/bin/env/python3
'''
Lab #6: Acoustic Waves 
Julian Toogood

This file solves the wave equation to model sound propagation under 
varying conditions. 

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
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate 
import os

# turn on interactive plotting, set stylesheet
plt.ion()
plt.style.use('seaborn-v0_8-poster')

def f_q1(x):
    '''
    Boundary condition function for wave position at x_1
    '''
    y = np.sin(np.pi*x) + np.sin(2*np.pi*x)
    return y

def g_zero(x):
    '''
    Boundary condition function for wave velocity at x_1
    '''
    y = 0
    return y 

def finedif(f,g,a,b,c,n,m,debug=False):
    '''
    FINEDIF Finite-Difference Solution for the Wave Equation
    (Mathews & Fink, 4th ed., pg.554, translated from MATLAB) 
    Input  - f=u(x,0) as a string 'f' - initial position at time zero
           - g=ut(x,0) as a string 'g'- initial velocity at time zero
           - a and b right endpoints of [0,a] and [0,b] given 0<=x<=a, 0<=t<=b
           - c the constant in the wave equation 
           - n and m number of grid points over [0,a] and [0,b]
           - debug : boolean, defaults to False - activate print statements to check variable values 
    Output - U solution matrix; analagous to Table 10.1 
    '''

    # Initialize parameters and U
    h=a/(n-1) #delta x
    k=b/(m-1) #delta t
    r=c*k/h #must be <=1 for stable solution 
    r2=r**2
    r22=r**2/2
    s1=1-r**2 # formula coefficients 
    s2=2-2*r**2
    U=np.zeros([n,m]) # first and last rows will remain zero given boundary conditions 

    if r>1:
        raise ValueError(f'Unstable solution detected: r={r} must be less than or equal to 1.')

    if debug:
        print(f'h={h}')
        print(f'k={k}')
        print(f'r={r}')

    # Compute first and second rows of actual wave (skip fixed endpoints)
    for i in range(1,n-1): 
        U[i,0] = f(h*(i))
        U[i,1] = s1*f(h*(i)) + k*g(h*(i)) + r22*( f(h*(i+1)) + f(h*(i-1)) ) # eq. 13: 2nd order taylor approx. for higher accuracy 

    # Compute remaining rows of U
    for j in range(2,m):
        for i in range(1,n-1):
            U[i,j] = s2*U[i,j-1] + r2*U[i-1,j-1] + U[i+1,j-1] - U[i,j-2] # eq. 7

    U=U.T # vertical time 

    return U

def question_2():
    '''
    Run this code to reproduce all results for question 4.

    Parameters: none

    Returns: none
    '''

    # define variables
    a = 1
    b = 0.5
    c = 2
    n = 11
    m = 11

    # run wave equation solver, print result 
    U = finedif(f_q1,g_zero,a,b,c,n,m)
    print(tabulate(U)); 

    # create field to plot U over 
    x = np.linspace(0,a,n)
    t = np.linspace(0,b,m)
    X,T = np.meshgrid(x,t)

    # plot 
    fig, ax = plt.subplots(1,1,subplot_kw={"projection":"3d"})
    ax.plot_surface(X,T,U,cmap="coolwarm")
    ax.set_ylim(b,0)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("U")
    fig.tight_layout()


# clear workspace 
plt.close('all')

print('Question 1:')
#question_1()

print('Question 2:')
question_2()
 
print('Question 3:') 
#question_3()