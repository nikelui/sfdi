# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:16:24 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import sys
import numpy as np
#import cv2 as cv
from matplotlib import pyplot as plt

sys.path.append('../common')
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work

from sfdi.readParams2 import readParams
from sfdsDataLoad import sfdsDataLoad
from calibrate import calibrate

from fitOps_sfds import fitOps_sfds
from chromFit import chromFit


par = readParams('parameters.cfg')

if len(par['freq_used']) == 0: # use all frequencies if empty
    par['freq_used'] = list(np.arange(len(par['freqs'])))

# Load tissue data. Note: a 30-samples moving average is applied to smooth the data
AC,wv,names = sfdsDataLoad(par,'Select tissue data file')
par['wv'] = wv # for SFDS

# Load calibration phantom data. Note: a 30-samples moving average is applied to smooth the data
ACph,wv,_ = sfdsDataLoad(par,'Select calibration phantom data file')

# Calibration step (in a loop)
cal_R = []
for ac in AC:
    cal_R.append(np.squeeze(calibrate(ac,ACph[0],par)))

# Fitting for optical properties (in a loop)
# TODO: this part is pretty computationally intensive, might be worth to optimize
op_fit_sfds = []
for cal in cal_R:
    op_fit_sfds.append(fitOps_sfds(cal,par))

chrom_map = []
for op in op_fit_sfds:
    chrom_map.append(chromFit(op,par)) # linear fitting for chromofores

## Saving results
#print('Saving data...')
#np.savez(par['savefile']+'sfds',op_fit_sfds=op_fit_sfds,cal_R=cal_R,chrom_map=chrom_map) # save important results
#print('Done!')

## Plotting (Maybe put this in a function?)
fig = plt.figure(1,figsize=(9,4))
plt.subplot(1,2,1)
for i in range(len(op_fit_sfds)):
    plt.plot(wv,op_fit_sfds[i][:,0],label=names[i].split('/')[-1])
plt.title(r'Absorption coefficient ($\mu_A$)')
plt.xlabel('wavelength (nm)')
plt.grid(True,linestyle=':')
plt.xlim([450,650])
plt.ylim([0.12,0.24])


plt.subplot(1,2,2)
for i in range(len(op_fit_sfds)):
    plt.plot(wv,op_fit_sfds[i][:,1],label=names[i].split('/')[-1])
plt.title(r'Scattering coefficient ($\mu_S$)')
plt.xlabel('wavelength (nm)')
plt.grid(True,linestyle=':')
plt.xlim([450,650])
plt.ylim([0,6])
plt.legend()

plt.tight_layout()
plt.show(block=False)

# Since the sfds only probes in on point, the chromophores map is actually a single value
titles = ['','HbO2','Hb','H2O','lipid','melanin'] # chromophores names. the first is empty to
                                                              # respect the naming convention
titles = [titles[i] for i in par['chrom_used']] # Only keep used chromophores
if len(chrom_map) > 0:
    for i,cm in enumerate(chrom_map):
        print('\n%s'%names[i].split('/')[-1])
        try:
            for j,x in enumerate(cm):
                print('%10s: %f' % (titles[j],x))
        except TypeError:
            print('%10s: %f' % (titles[0],cm))
