#!/usr/bin/env python3
'''
Solve the coffee problem to learn how to drink coffee effectively.
'''

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-poster')
plt.ion() 

def solve_temp(t, T_init=90., T_env=20.0, k=1/300.):
    '''
    This function returns temperature as a function of time using Newton's
    law of cooling.

    Parameters
    ----------
    t: Numpy array
        An array of time values in seconds.
    T_init: floating point, defaults to 90.
        Initial temperature in Celsius.
    T_env: floating point, defaults to 20
        Ambient air temperature in Celsius
    k: floating point, defaults to 1/300.
        Heat transfer coefficient in 1/s.

    Returns
    -------
    t_coffee: Numpy array
        Temperature corresponding to time t

    '''

    t_coffee = T_env + (T_init - T_env) * np.exp(-k*t)

    return t_coffee


def newtcool(t, Tnow, k=1/300., T_env=20.0):
    '''
    Newton's law of cooling: given time t, Temperature now (Tnow), a cooling
    coefficient (k), and an environmental temp (T_env), return the rate of
    cooling (i.e., dT/dt)
    '''

    return -k * (Tnow - T_env)


def solve_euler(dfx, dt=.25, f0=90., t_start=0., t_final=300., **kwargs):
    '''
    Solve an ordinary diffyQ using Euler's method.
    Extra kwargs are passed to the dfx function.


    Parameters
    ----------
    dfx : function
        A function representing the time derivative of our diffyQ. It should
        take 2 arguments: the current time and current function value
        and return 1 value: the time derivative at time `t`.
    f0 : float
        Initial condition for our differential equation
    t_start, t_final : float, 0 and 300. respectively
        The start and final times for our solver in seconds
    dt : float, defaults to 0.25
        Time step in seconds.

    Returns
    -------
    t : Numpy array
        Time in seconds over the entire solution.
    fx : Numpy array
        The solution as a function of time.

    '''

    # configure the problem
    time = np.arange(t_start, t_final, dt)
    fx = np.zeros(time.size)
    fx[0] = f0

    # solve 
    for i in range(time.size-1):
        fx[i+1] = fx[i] + dt * dfx(time[i], fx[i], **kwargs)

    return time, fx

def solve_rk8(dfx, dt=0.25, f0=90., t_start=0., t_final=300., **kwargs):
    '''
    Solve an ODE using the DOP853 method 
    Extra kwargs are passed to the dfx function.

    Parameters 
    ----------
    dfx : function
        A function representing the time derivative of our diffyQ. It should
        take 2 arguments: the current time and current function value
        and return 1 value: the time derivative at time `t`.
    f0 : float
        Initial condition for our differential equation
    t_start, t_final : float, 0 and 300. respectively
        The start and final times for our solver in seconds
    dt : float, defaults to 0.25
        Time step in seconds.

    Returns (NEEDS UPDATING)
    -------
    t : Numpy array
        Time in seconds over the entire solution.
    fx : Numpy array
        The solution as a function of time.
    '''
    
    from scipy.integrate import solve_ivp

    result = solve_ivp(dfx, [t_start, t_final], [f0], method='DOP853',max_step=dt)

    return result.t, result.y[0,:]


def explore_numerical_solve(dt=1.0):
    '''
    Compare numerical vs. analytical solution for Newton's law of cooling.

    Parameters
    ---------
    dt : float, defaults to 1.0
        Set the time step for the Euler solver.
    '''

    # Create ANALYTICAL time series of temperatures for cooling coffee.
    t = np.arange(0, 300., 0.5)
    temp1 = solve_temp(t)  # also the same as control case.

    # Obtain Euler solver numerical solution.
    etime, etemp = solve_euler(newtcool, t_final=300., dt=dt)
    etime2, etemp2 = solve_rk8(newtcool, t_final=300., dt=dt)

    # Make a beautiful plot to illustrate how the numerical solution
    # performs.
    fig, ax = plt.subplots(1, 1, figsize=[10.24,  5.91])
    # Plot lines we want to show:
    ax.plot(t, temp1, label='Analytical Solution')
    ax.plot(etime, etemp, 'o--', label=f'Euler Solution for $\Delta t={dt}s$')
    ax.plot(etime2, etemp2, 'o--', label=f'RK8 Solution')

    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (C)')
    ax.set_title('Analytical vs. Numerical: The Greatest Battle of Our Time')
    fig.tight_layout()

    return fig

