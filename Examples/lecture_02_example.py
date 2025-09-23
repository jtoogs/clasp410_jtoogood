#!/usr/bin/env python3

'''
Series of simple examples for Lecture 2 about turkeys
'''

# import packages 
import numpy as np
import matplotlib.pyplot as plt

# interactive plotting 
plt.ion() 
plt.style.use('fivethirtyeight')

# define variables 
dx = 0.1
x = np.arange(0, 6*np.pi, dx)

# calculate values 
sinx = np.sin(x)
cosx = np.cos(x) # analytical solution

# fwd and bck are the same calculation but they index to different parts of the x series 
# python indexing is [start:stop) - ie upper bound is exclusive
fwd_diff = (sinx[1:] - sinx[:-1]) / dx
bkd_diff = (sinx[1:] - sinx[:-1]) / dx
cnt_diff = (sinx[2:] - sinx[:-2]) / (2*dx)

# plot 
plt.plot(x,cosx,label=r"analytical derivative of $\sin{x}$")
plt.plot(x[:-1],fwd_diff,label='forward diff approx')
plt.plot(x[1:],bkd_diff,label='backward diff approx')
plt.plot(x[1:-1],cnt_diff,label='central diff approx')
plt.legend(loc='best')
plt.show()

# how much error for each delta x
dxs = np.array([2**-n for n in range(20)])
err_fwd, err_cnt = [],[]

for dx in dxs:
    x = np.arange(0, 2.5*np.pi, dx)
    sinx = np.sin(x)

    fwd_diff = (sinx[1:] - sinx[:-1]) / dx
    cnt_diff = (sinx[2:] - sinx[:-2]) / (2*dx)

    err_fwd.append(np.abs(fwd_diff[-1]-np.cos(x[-1])))
    err_cnt.append(np.abs(cnt_diff[-1]-np.cos(x[-2])))

# plot 
fig, ax = plt.subplots(1,1)
ax.loglog(dxs,err_fwd,'.',label='forward diff')
ax.loglog(dxs,err_cnt,'.',label='central diff')
ax.set_xlabel('$\Delta$x')
ax.set_ylabel('Error')
ax.legend(loc='best')
fig.tight_layout()