# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:54:57 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Since I don't have data points for TS1 below 470nm, interpolate and extract a couple
of values to extend the range (otherwise extrapolation might give strange results)
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

data = np.genfromtxt('TS1_extended.txt',dtype='float').T

wv = data[0,:]
ua = data[1,:]
us = data[2,:]
n = data[3,:]

WV = np.arange(380,721,1,dtype='float')

fa = interp1d(wv,ua,kind='cubic',fill_value='extrapolate')
UA = fa(WV)
fs = interp1d(wv,us,kind='cubic',fill_value='extrapolate')
US = fs(WV)
fn = interp1d(wv,n,kind='cubic',fill_value='extrapolate')
N = fn(WV)


# Plot
labels=['$\mu_A$','$\mu_s$','n']
plt.figure(1,figsize=(8,5))
for (i,line) in enumerate([UA,US,N]):
    plt.plot(WV,line,'-',label=labels[i])
for (i,points) in enumerate([ua,us,n]):
    plt.plot(wv,points,'o')
plt.legend(loc=1)
plt.xlim([380,720])