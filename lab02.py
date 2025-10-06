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

def dNdt_pp(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra predator/prey equations for
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
    dN1dt = a*N[0] - b*N[0]*N[1]
    dN2dt = -c*N[1] + d*N[1]*N[0]

    return dN1dt, dN2dt

def euler_solve(func, N1_init=.5, N2_init=.5, dT=.1, t_final=100.0,
                a=1, b=2, c=1, d=3):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    a first-order-accurate Euler method with a constant forward time step.

    Parameters
    ----------
    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init, N2_init : float, default=0.5
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

    ## from lecture example 
    # configure the problem / preallocate variables 
    time = np.arange(0, t_final, dT)
    N1 = np.zeros(time.size)
    N2 = np.zeros(time.size)
    dN1 = np.zeros(time.size)
    dN2 = np.zeros(time.size)
    N1[0] = N1_init
    N2[0] = N2_init 

    # solve for the function values 
    for i in range(time.size-1):
        dN1[i], dN2[i] = func(i, [N1[i], N2[i]], a=a, b=b, c=c, d=d) #retrieve derivatives at index i 
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
    N1_init, N2_init : float, default=0.5
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

def check_solver(func):
    '''
    Quickly see if the function is working properly. 
    Also serves as scaffolding/sandbox for preparing question functions  

    Parameters: 
    ----------
    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    Returns: none 
    '''

    # Obtain solver solutions.
    etime, eN1, eN2 = euler_solve(func)
    rk8time, rk8N1, rk8N2 = solve_rk8(func)

    # Make a beautiful plot to illustrate how the numerical solution
    # performs.
    fig, ax = plt.subplots(1, 1, figsize=[10.24,  5.91])
    # Plot lines we want to show:
    ax.plot(etime, eN1, 'o--', label=f'N1 Euler Solution for $\Delta t=(dt)s$')
    ax.plot(etime, eN2, 'o--', label=f'N2 Euler Solution for $\Delta t=(dt)s$')
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

    # Obtain solver solutions.
    etime_comp, eN1_comp, eN2_comp = euler_solve(dNdt_comp,N1_init=0.3,N2_init=0.6,dT=1.0)
    rk8time_comp, rk8N1_comp, rk8N2_comp = solve_rk8(dNdt_comp,N1_init=0.3,N2_init=0.6,dT=1.0)
    etime_pp, eN1_pp, eN2_pp = euler_solve(dNdt_pp,N1_init=0.3,N2_init=0.6,dT=0.05)
    rk8time_pp, rk8N1_pp, rk8N2_pp = solve_rk8(dNdt_pp,N1_init=0.3,N2_init=0.6,dT=0.05)

    # Make a beautiful plot to illustrate how the numerical solution
    # performs.
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=[12,6])
    # Plot lines we want to show:
    ax1.plot(etime_comp, eN1_comp, '-', color='b', label=f'N1 Euler')
    ax1.plot(etime_comp, eN2_comp, '-', color='orangered', label=f'N2 Euler')
    ax1.plot(rk8time_comp, rk8N1_comp, ':', color='b', label=f'N1 RK8')
    ax1.plot(rk8time_comp, rk8N2_comp, ':', color='orangered', label=f'N2 RK8')

    ax1.legend(loc='upper left')
    ax1.set_xlabel('Time $(years)$')
    ax1.set_ylabel('Population/Carrying Cap.')
    ax1.set_title('Lokta-Volterra Competition Model')
    ax1.grid(True)

    ax2.plot(etime_pp, eN1_pp, '-', color='b', label=f'N1 (Prey) Euler')
    ax2.plot(etime_pp, eN2_pp, '-', color='orangered', label=f'N2 (Predator) Euler')
    ax2.plot(rk8time_pp, rk8N1_pp, ':', color='b', label=f'N1 (Prey) RK8')
    ax2.plot(rk8time_pp, rk8N2_pp, ':', color='orangered', label=f'N2 (Predator) RK8')

    ax2.legend(loc='upper left')
    ax2.set_xlabel('Time $(years)$')
    ax2.set_ylabel('Population/Carrying Cap.')
    ax2.set_title('Lokta-Volterra Predator-Prey Model')
    ax2.grid(True)

    plt.figtext(0.5, 0.01, 'Coefficients: a=1, b=2, c=1, d=3', 
            ha='center', va='bottom', fontsize=12)
    
    fig.tight_layout()

    print("We verify our code is working by reproducing the figure from the lab instructions.")
    print("Our figure matches the one from the lab instructions, so our code appears to be " \
    "working correctly.")
    

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

#check solver
# check_solver(dNdt_comp)
# check_solver(dNdt_pp)

print('Question 1:')
question_1()

print('Question 2:')
question_2()

print('Question 3:')
question_3()