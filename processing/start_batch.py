# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:16:24 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import sys
import numpy as np
import numpy.ma as mask
import cv2 as cv
from matplotlib import pyplot as plt

sys.path.append('../common')
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work

from sfdi.readParams3 import readParams
from sfdi.crop import crop
from rawDataLoadBatch import rawDataLoadBatch
from rawDataLoad import rawDataLoad
from calibrate import calibrate
from stackPlot import stackPlot
from fitOps import fitOps
from chromFit import chromFit
from chromPlot import chromPlot
from opticalSpectra_batch import opticalSpectra

par = readParams('parameters.ini')

if len(par['freq_used']) == 0: # use all frequencies if empty
    par['freq_used'] = list(np.arange(len(par['freqs'])))

if len(par['wv_used']) == 0: # use all wavelengths if empty
    par['wv_used'] = list(np.arange(len(par['wv'])))

# Load tissue data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
AC,names,tstamps = rawDataLoadBatch(par, 'Select tissue') # This approach is a bit rough, but there are no simple solutions

## Load calibration phantom data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
ACph,_ = rawDataLoad(par,'Select calibration phantom data folder')

# Calibration step
cal_R = []
for ac in AC:
    cal_R.append(calibrate(ac,ACph,par))

stackPlot(cal_R[0],'magma') # Maybe show all? Or none

#sys.exit()

## Select only one ROI on calibration image
ROI = cv.selectROI('Select ROI',cal_R[0][:,:,0,0]) # press Enter to confirm selection
cv.destroyWindow('Select ROI')

## Fitting for optical properties
# TODO: this part is pretty computationally intensive, might be worth to optimize
op_fit_maps = []
for cal in cal_R:
    op_fit_maps.append(fitOps(crop(cal,ROI),par))

if (len(par['chrom_used'])>0):
    chrom_map = []
    for op in op_fit_maps:
        chrom_map.append(chromFit(op,par)) # linear fitting for chromofores

op_ave,op_std = opticalSpectra(op_fit_maps,par,names,outliers=True)

print('Saving data...')
#np.savez(par['savefile'],op_fit_maps=op_fit_maps,cal_R=cal_R,ROI=ROI) # save important results
np.savez(par['savefile'],op_ave=op_ave,op_std=op_std)
print('Done!')


if (len(par['chrom_used'])>0):
    for i in range(len(chrom_map)):
        chrom_map[i] = chromPlot(chrom_map[i],names[i],par)
    
    ## Plot average of chromophores in time
    titles = ['',r'HbO$_2$','Hb',r'H$_2$O','lipid','melanin'] # chromophores names. the first is empty to
                                                                  # respect the naming convention
    titles = [titles[i] for i in par['chrom_used']] # Only keep used chromophores
    
    chroms_ave = mask.masked_array(chrom_map).mean(axis=(1,2)) # collapse dimensions and average
    chroms_std = mask.masked_array(chrom_map).std(axis=(1,2)) # collapse dimensions and std
    plt.figure(300)
    for i in range(np.shape(chroms_ave)[1]):
        plt.errorbar(np.arange(9),chroms_ave[:,i],fmt='D',yerr=chroms_std[:,i].data,linestyle='solid',
                     capsize=5,markersize=3)
    plt.legend(titles)
    plt.grid(True)
    plt.show(block=False)