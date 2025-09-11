#!/usr/bin/env/python3 
'''
Solve the coffee problem to learn how to drink coffee effectively. 
'''

import numpy as np 
import matplotlib.pyplot as plt 
plt.ion() 

plt.style.use('fivethirtyeight')

print("i DRINK a lot of COFFEE")
print("i SHIT a lot of POOP\n")

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

def time_to_temp(T_final, T_init=90, T_env=20, k=1/300.):
    '''
    Given a target temperature, determine how long it takes to reach the 
    target temp using Newton's law of cooling 
    
    Parameters 
    --------
    T_final: float
        Final temperature in Celsius. 
    T_init: float, defaults to 90
        Initial temperature in Celsius. 
    T_env: float, defaults to 20
        Ambient air temperature in Celsius.
    k: float, defaults to 1/300
        Heat transfer coefficient in 1/s.

    Returns 
    --------
    t: float 
        Time in seconds to cool to target T_final. 
    '''
    
    t = (-1/k) * np.log( (T_final - T_env) / (T_init - T_env) )
    return t 

def verify_code():
    '''
    Verify that our implementation is correct using example problem from
    {URL}
    '''

    t_real = 60.*10.76
    k = np.log(95/110) / -120
    t_code = time_to_temp(120, T_init=180, T_env=70, k=k)
    print("Target solution is: ", t_real)
    print("Numerical solution is: ", t_code)
    print("Difference is: ", t_real-t_code)

# solve the actual problem using the functions declared above
# quantitative solution:
t_1 = time_to_temp(65) # add cream at T=65 to get to 60
t_2 = time_to_temp(60, T_init=85) # add cream immediately 
t_c = time_to_temp(60) # control: no cream 

print("TIME TO DRINKABLE COFFEE:")
print(f"\tControl case = {t_c:.2f}s")
print(f"\tDelayed cream = {t_1:.2f}s")
print(f"\tImmediate cream = {t_2:.2f}s")

# create time series of temperatures 
t = np.arange(0,600.0,0.5)
temp1 = solve_temp(t) #control
temp2 = solve_temp(t, T_init=85)

# plot 
fig, ax = plt.subplots(1,1)
ax.plot(t,temp1,label=f"Delayed cream (T={t_1:.1f}s")
ax.plot(t,temp2,label=f"Immediate cream (T={t_2:.1f}s")
ax.legend()
ax.set_xlabel("Time (s)")
ax.set_ylabel("Temperature (C)")
ax.set_title("To cream or not to cream (and when)",fontsize=14)
fig.tight_layout() 

