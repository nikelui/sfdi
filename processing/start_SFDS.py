# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:16:24 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

from sfdi.common.readParams import readParams
from sfdi.processing.sfdsDataLoad import sfdsDataLoad
from sfds.processing.calibrate import calibrate
from sfdi.processing.fitOps_sfds import fitOps_sfds
from sfdi.processing.chromFit import chromFit

from sfdi.processing import __path__ as par_path  # processing parameters path
par = readParams('{}/parameters.ini'.format(par_path))

if len(par['freq_used']) == 0: # use all frequencies if empty
    par['freq_used'] = list(np.arange(len(par['freqs'])))
    
# Load tissue data. Note: a 30-samples moving average is applied to smooth the data
AC,wv,names = sfdsDataLoad(par,'Select tissue data file')
par['wv'] = wv # for SFDS

# Load calibration phantom data. Note: a 30-samples moving average is applied to smooth the data
ACph,wv,_ = sfdsDataLoad(par,'Select calibration phantom data file')


## TODO: process one dataset at a time (even if SFDS data does not take much memory)
# Calibration step (in a loop)
cal_R = []
for ac in AC:
    cal_R.append(np.squeeze(calibrate(ac, ACph[0], par, old=True)))

# Fitting for optical properties (in a loop)
# TODO: this part is pretty computationally intensive, might be worth to optimize
op_fit_sfds = []
for cal in cal_R:
    op_fit_sfds.append(fitOps_sfds(cal[:,par['freq_used']], par))

chrom_map = []
for op in op_fit_sfds:
    chrom_map.append(chromFit(op,par)) # linear fitting for chromofores

nn = []

for name in names:
    nn.append(name.split('/')[-1].split('_')[-2])
## Saving results
np.save('{}{}_SFDS_{}fx'.format(par['savefile'], nn[0], len(par['freq_used'])),
        np.concatenate((wv, op_fit_sfds[0]), axis=1))

## TODO: Plotting (Maybe put this in a function?)
fig = plt.figure(1,figsize=(9,4))
plt.subplot(1,2,1)
for i in range(len(op_fit_sfds)):
    plt.plot(wv,op_fit_sfds[i][:,0],label=names[i].split('/')[-1])
plt.title(r'Absorption coefficient ($\mu_A$)')
plt.xlabel('wavelength (nm)')
plt.grid(True,linestyle=':')
plt.xlim([450,650])
plt.ylim([0,0.5])

plt.subplot(1,2,2)
for i in range(len(op_fit_sfds)):
    plt.plot(wv,op_fit_sfds[i][:,1],label=names[i].split('/')[-1])
plt.title(r'Scattering coefficient ($\mu_S$)')
plt.xlabel('wavelength (nm)')
plt.grid(True,linestyle=':')
plt.xlim([450,650])
plt.ylim([0,5])
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
