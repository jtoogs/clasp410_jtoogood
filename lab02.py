#!/usr/bin/env/python3
'''
Lab #2: Population Control
Julian Toogood

This file solves the population control problem for Lab 02 and all subparts. 

Hi Sarah :) 
To reproduce the values and plots in my report, do this: 
- For question 1, run the function question_1() with no inputs.
- For question 2, run the function question_2() with no inputs.
- For question 3, run the function question_3() with no inputs.

'''

# import packages 
import numpy as np 
import matplotlib.pyplot as plt

# turn on interactive plotting, set stylesheet
plt.ion()
plt.style.use('seaborn-v0_8-poster')

def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra competition equations for
    two species. Given normalized populations, `N1` and `N2`, as well as the
    four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.

    Parameters
    ----------
    t : float
        The current time (not used here).
    N : two-element list
        The current values of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
        The values of the Lotka-Volterra coefficients.

    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of `N1` and `N2`.
    '''

    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dN2dt = c*N[1]*(1-N[1]) - d*N[1]*N[0]

    return dN1dt, dN2dt

def euler_solve(func, N1_init=.5, N2_init=.5, dT=.1, t_final=100.0):
    '''
    <Your good docstring here>

    Parameters
    ----------
    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init : float
        DESCRIPTION
    
    Returns
    -------
    <more good docstring here>
    '''

    ## from lecture example 
    # configure the problem
    time = np.arange(0, t_final, dT)
    N1 = np.zeros(time.size)
    N2 = np.zeros(time.size)
    dN1 = np.zeros(time.size)
    dN2 = np.zeros(time.size)
    N1[0] = N1_init
    N2[0] = N2_init 

    # solve for the function values 
    for i in range(time.size-1):
        dN1[i], dN2[i] = func(i, [N1[i], N2[i]] ) #retrieve derivatives at index i 
        N1[i+1] = N1[i] + (dT * dN1[i]) #euler method to find next function value 
        N2[i+1] = N2[i] + (dT * dN2[i])

    return time, N1, N2 

def solve_rk8(func, N1_init=.5, N2_init=.5, dT=10, t_final=100.0,
a=1, b=2, c=1, d=3):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    Scipy's ODE class and the adaptive step 8th order solver.

    Parameters
    ----------
    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init, N2_init : float
        Initial conditions for `N1` and `N2`, ranging from (0,1]
    dT : float, default=10
        Largest time step allowed in years.
    t_final : float, default=100
        Integrate until this value is reached, in years.
    a, b, c, d : float, default=1, 2, 1, 3
        Lotka-Volterra coefficient values

    Returns
    -------
    time : Numpy array
        Time elapsed in years.
    N1, N2 : Numpy arrays
        Normalized population density solutions.
    '''
    from scipy.integrate import solve_ivp
    
    # Configure the initial value problem solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init],
    args=[a, b, c, d], method='DOP853', max_step=dT)
    
    # Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :]
    
    # Return values to caller.
    return time, N1, N2

def check_solver(dt=1.0):
    '''
    Quickly see if the function is working properly. 
    Also serves as scaffolding/sandbox for preparing question functions  

    Parameters
    ---------
    dt : float, defaults to 1.0
        Set the time step for the solver.
    '''

    t = np.arange(0, 300., 0.5)

    # Obtain Euler solver numerical solution.
    etime, eN1, eN2 = euler_solve(dNdt_comp)
    rk8time, rk8N1, rk8N2 = solve_rk8(dNdt_comp)

    # Make a beautiful plot to illustrate how the numerical solution
    # performs.
    fig, ax = plt.subplots(1, 1, figsize=[10.24,  5.91])
    # Plot lines we want to show:
    ax.plot(etime, eN1, 'o--', label=f'N1 Euler Solution for $\Delta t={dt}s$')
    ax.plot(etime, eN2, 'o--', label=f'N2 Euler Solution for $\Delta t={dt}s$')
    ax.plot(rk8time, rk8N1, 'o--', label=f'N1 RK8 Solution')
    ax.plot(rk8time, rk8N2, 'o--', label=f'N2 RK8 Solution')

    ax.legend(loc='lower right')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Population (%)')
    ax.set_title('TEST FIGURE')
    fig.tight_layout()

    return fig

def question_1(debug=False):
    '''
    Run this code to reproduce all results for question 1.

    Parameters
    ----------
    debug : boolean, default=False
        Activates intermediate print steps to check for incorrect values

    Returns: none
    '''

def question_2(debug=False):
    '''
    Run this code to reproduce all results for question 2.

    Parameters
    ----------
    debug : boolean, default=False
        Activates intermediate print steps to check for incorrect values

    Returns: none
    '''

def question_3(debug=False):
    '''
    Run this code to reproduce all results for question 3.

    Parameters
    ----------
    debug : boolean, default=False
        Activates intermediate print steps to check for incorrect values

    Returns: none
    '''


print('Question 1:')
question_1()

print('Question 2:')
question_2()

print('Question 3:')
question_3()