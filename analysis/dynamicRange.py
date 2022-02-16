# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 23:57:44 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

from sfdi.common.readParams import readParams
#from sfdi.processing.crop import crop
from sfdi.processing.stackPlot import stackPlot
from sfdi.processing.rawDataLoad import rawDataLoad
from sfdi.processing import __path__ as par_path

par = readParams('{}/parameters.ini'.format(par_path[0]))

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
    temp = cv.imread('{}/im_0-{}-0.bmp'.format(name,_i), cv.IMREAD_GRAYSCALE)
    h = temp.shape[0] // 2
    s.append(np.mean(temp[h-10:h+10,:], axis=0))
    plt.plot(s[-1], label=r'{} mm$^{{-1}}$'.format(fx[_i]))
    if _i == len(fx) - 1:
        plt.plot(s[-1], '-r', linewidth=2.5)
plt.legend(loc=2)
plt.grid(True, linestyle=':')
plt.tight_layout()
