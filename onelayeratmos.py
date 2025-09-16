#!/usr/bin/env/python3 
'''
Use a one-layer atmosphere model and compare with temperature anomaly data to test the hypothesis
that global warming can be solely attributed to variations in solar irradiance. 
'''

import numpy as np 
import matplotlib.pyplot as plt 
plt.ion() 

plt.style.use('fivethirtyeight')

def get_temp(S_0, a=0.33):
    '''
    This function returns the Earth's temperature as a function of solar irradiance
    and albedo using a one-layer atmosphere model.

    Parameters 
    --------
    S_0: float
        Solar irradiance in W m^-2
    a: float, defaults to 0.33
        Planetary albedo

    Returns 
    --------
    T_earth: Numpy array 
        Earth's temperature for the given solar irradiance and albedo. 

    '''

    s = 5.67*(10**-8) # stefan-boltzmann constant 
    T_earth = ( ((1-a)*S_0) / (2*s) )**(1/4)
    return T_earth

# confirm function works with a known test case 
T_test = get_temp(1350)
print(f'Test value {T_test} should equal 298.8')

# define variables
year = np.array([1900, 1950, 2000])
s_0 = np.array([1365, 1366.5, 1368]) #THIS MIGHT MESS UP THE EQ BC OF VARIABLE TYPE 
t_anom =np.array([-0.4, 0, 0.4])

# calculate an array from these variables using the function defined above
t_series = np.array([0.,0.,0.])
t_series[0] = get_temp(s_0[0])
t_series[1] = get_temp(s_0[1])
t_series[2] = get_temp(s_0[2])
print(f'Time series of calculated temperatures: {t_series}'); 

# plot 
fig, ax1 = plt.subplots(1,1)

color = 'tab:red'
ax1.plot(year,t_series,color=color) #Temperature from one-layer atmosphere model
ax1.set_ylim([299,301])
ax1.set_xlabel("Year")
ax1.set_ylabel("Temperature (degrees C)")
ax1.tick_params(axis='y', labelcolor=color)
#ax1.grid(False)

ax2 = ax1.twinx() 
color = 'tab:blue'
ax2.plot(year,t_anom,color=color) #Observed temperature anomaly
ax2.set_ylim([-1.5,0.5])
ax2.set_ylabel("Temperature anomaly (degrees C)")
ax2.tick_params(axis='y', labelcolor=color)
#ax2.grid(False)

ax1.set_title("One-layer atmosphere model compared with observed temperature anomaly",fontsize=10)

fig.tight_layout() 
plt.show()