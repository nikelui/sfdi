# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 23:57:44 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

import sys, os
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

sys.path.append('../common')
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work

from sfdi.readParams3 import readParams
from sfdi.crop import crop
from stackPlot import stackPlot
from rawDataLoad import rawDataLoad

par = readParams('../processing/parameters.ini')

if len(par['freq_used']) == 0: # use all frequencies if empty
    par['freq_used'] = list(np.arange(len(par['freqs'])))

if len(par['wv_used']) == 0: # use all wavelengths if empty
    par['wv_used'] = list(np.arange(len(par['wv'])))

## Load tissue data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
AC,name,DC = rawDataLoad(par, 'Select tissue data folder')
stackPlot(AC, cmap='magma', num=100)
stackPlot(DC, cmap='magma', num=200)

fx = par['freqs']
s = []  # sinusoidal signals
plt.figure(1, figsize=(15,6))
for _i in range(len(fx)):
    temp = cv.imread('{}/im_0{}0.bmp'.format(name,_i), cv.IMREAD_GRAYSCALE)
    h = temp.shape[0] // 2
    s.append(np.mean(temp[h-10:h+10,:], axis=0))
    plt.plot(s[-1], label=r'{} mm$^{{-1}}$'.format(fx[_i]))
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
