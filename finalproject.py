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

def f_q3(x):
    '''
    Boundary condition function for wave position at x_1
    '''
    if 0.0 <= x <= 0.1:
        y = np.sin(np.pi*x/0.1)
    else:
        y=0
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

    # Test solution stability
    if r>1:
        raise ValueError(f'Unstable solution detected: r={r} must be less than or equal to 1.')

    # Debugging clause 
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

    # Transpose array
    U=U.T 

    return U

def wave_analytical(a,b,n,m):
    # determine array dimensions for output values 
    x = np.linspace(0,a,n)
    t = np.linspace(0,b,m)

    # preallocate
    U = np.zeros([n,m])

    # solve equation for each gridpoint
    for i in range(0,x.size):
        for j in range(0,t.size):
            U[i,j] = np.sin(np.pi*x[i])*np.cos(2*np.pi*t[j]) + np.sin(2*np.pi*x[i])*np.cos(4*np.pi*t[j])

    # transpose array 
    U = U.T

    return U

def save_this_plot(fig, filename, folder="Lab06_results/"):
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

def question_1():
    '''
    Run this code to reproduce all results for question 1.

    Parameters: none

    Returns: none
    '''

    # define variables
    a = 1
    b = 0.5
    # c = 2
    n = 11
    m = 11

    # generate analytical solution grid 
    U = wave_analytical(a,b,n,m)

    # generate table
    print(tabulate(U))

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
    ax.set_title("Analytic Wave Equation Solution, c=2")
    fig.tight_layout()

    save_this_plot(fig,'Lab06_Q1_1')

def question_2():
    '''
    Run this code to reproduce all results for question 2.

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
    print(tabulate(U))

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
    ax.set_title("PDE Solver Wave Equation Solution, c=2")
    fig.tight_layout()

    save_this_plot(fig,'Lab06_Q2_1')

def question_3():
    '''
    Run this code to reproduce all results for question 3.

    Parameters: none

    Returns: none
    '''

    # define variables
    a = 1 #m
    b = 0.01 #s
    c = 320 #m/s
    n = 201
    m = 641

    # run wave equation solver, print result 
    U = finedif(f_q3,g_zero,a,b,c,n,m)
    # print(tabulate(U))

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
    ax.set_title("PDE Solver Wave Equation Solution, c=320 m/s")
    fig.tight_layout()

    save_this_plot(fig,'Lab06_Q3_1')

    # define variables
    a = 1 #m
    b = 0.01 #s
    c = 350 #m/s
    n = 201
    m = 701

    # run wave equation solver, print result 
    U = finedif(f_q3,g_zero,a,b,c,n,m)
    # print(tabulate(U))

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
    ax.set_title("PDE Solver Wave Equation Solution, c=350 m/s")
    fig.tight_layout()

    save_this_plot(fig,'Lab06_Q3_2')

# clear workspace 
# plt.close('all')

print('Question 1:')
question_1()

print('Question 2:')
question_2()
 
print('Question 3:') 
question_3()