# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:16:24 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Script for processing SFDS data

Steps before starting:
    - Check the parameters.ini file
    - Turn multi-frequencies to True or False
    - Select if dataset is homogeneous (fix mua at f0)

Steps:
    - Select calibration phantom data
    - Select tissue data
    - Select chromophore reference file [if processed]
"""
import os
import numpy as np
from matplotlib import pyplot as plt
# from scipy.io import loadmat
from scipy.io import savemat

from sfdi.common.readParams import readParams
from sfdi.processing.sfdsDataLoad import sfdsDataLoad
from sfdi.processing.calibrate import calibrate
from sfdi.processing.fitOps_sfds import fitOps_sfds
from sfdi.processing.chromFit import chromFit

from sfdi.processing import __path__ as par_path  # processing parameters path
par = readParams('{}/parameters.ini'.format(par_path[0]))

if len(par['freq_used']) == 0: # use all frequencies if empty
    par['freq_used'] = list(np.arange(len(par['freqs'])))

# Load tissue data. Note: a 30-samples moving average is applied to smooth the data
AC,wv,names = sfdsDataLoad(par, 'Select tissue data file')
par['wv'] = wv # for SFDS

# Load calibration phantom data. Note: a 30-samples moving average is applied to smooth the data
ACph,wv,_ = sfdsDataLoad(par, 'Select calibration phantom data file')
  
## TODO: process one dataset at a time (even if SFDS data does not take much memory)
# Calibration step (in a loop)
cal_R = []
for ac in AC:
    cal_R.append(np.squeeze(calibrate(ac, ACph[0], par, old=True)))

# Fitting for optical properties (in a loop)
if True:  # multi-frequencies approach
    FX = list(list(range(_i, _i+4)) for _i in range(len(par['freqs']) - 3))
else:
    FX = [par['freq_used']]

op_fit_sfds = []

for _c, cal in enumerate(cal_R):
    print('\nData set {} of {}'.format(_c+1, len(cal_R)))
    # Initialize. Dimensions: (wv, fx, op)
    temp = np.zeros((len(par['wv']), len(FX), 2), dtype=float)
    for _f, fx in enumerate(FX):
        print('Frequency set {} of {}'.format(_f+1, len(FX)))
        par['freq_used'] = fx
        if _f == 0:
            temp[:, _f, :] = fitOps_sfds(cal[:, par['freq_used']], par, homogeneous=False)
            op_guess = temp[:, _f, :]  # save [mua, mus] at f0 as initial guess
        else:
            temp[:, _f, :] = fitOps_sfds(cal[:, par['freq_used']], par, guess= op_guess, homogeneous=True)
    op_fit_sfds.append(temp)

# TODO: fix this fitting
# chrom_map = []
# for op in op_fit_sfds:
#     chrom_map.append(chromFit(op,par)) # linear fitting for chromofores

nn = []

for name in names:
    nn.append(name.split('/')[-1][:-4])

## Saving results
## check if save directories exist and create them otherwise
if not os.path.exists(par['savefile']):
    os.makedirs(par['savefile'])
   
to_save = {'wv': wv}
for _n, name in enumerate(nn):
    to_save[name] = op_fit_sfds[_n]
savemat('{}/SFDS_{}fx.mat'.format(par['savefile'], len(FX)), to_save)

#%%
## TODO: Plotting (Maybe put this in a function?)
n = 13
fig = plt.figure(1,figsize=(9,4))
plt.subplot(1,2,1)
plt.suptitle('{}'.format(nn[n]))
for i in range(1):
    plt.plot(wv,op_fit_sfds[n][:,:5,0])#,label=names[i].split('/')[-1])
plt.title(r'Absorption coefficient ($\mu_A$)')
plt.xlabel('wavelength (nm)')
plt.grid(True,linestyle=':')
plt.xlim([450,750])
plt.ylim([-0.01,0.05])

plt.subplot(1,2,2)
for i in range(1):
    plt.plot(wv,op_fit_sfds[n][:,:5,1])#,label=names[i].split('/')[-1])
plt.title(r'Scattering coefficient ($\mu_S$)')
plt.xlabel('wavelength (nm)')
plt.grid(True,linestyle=':')
plt.xlim([450,750])
plt.ylim([1, 3])
plt.legend(['f0','f1','f2','f3','f4'])

plt.tight_layout()
plt.show(block=False)

# Since the sfds only probes in on point, the chromophores map is actually a single value
# titles = ['','HbO2','Hb','H2O','lipid','melanin'] # chromophores names. the first is empty to
#                                                               # respect the naming convention
# titles = [titles[i] for i in par['chrom_used']] # Only keep used chromophores
# if len(chrom_map) > 0:
#     for i,cm in enumerate(chrom_map):
#         print('\n%s'%names[i].split('/')[-1])
#         try:
#             for j,x in enumerate(cm):
#                 print('%10s: %f' % (titles[j],x))
#         except TypeError:
#             print('%10s: %f' % (titles[0],cm))
