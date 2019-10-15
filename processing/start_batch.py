# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:16:24 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import sys
import numpy as np
import cv2 as cv

sys.path.append('../common')
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work

from sfdi.readParams2 import readParams
from sfdi.crop import crop
from rawDataLoadBatch import rawDataLoadBatch
from rawDataLoad import rawDataLoad
from calibrate import calibrate
from stackPlot import stackPlot
from fitOps import fitOps
from chromFit import chromFit
from chromPlot import chromPlot
from opticalSpectra_batch import opticalSpectra


par = readParams('parameters.cfg')

if len(par['freq_used']) == 0: # use all frequencies if empty
    par['freq_used'] = list(np.arange(len(par['freqs'])))

# Load tissue data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
AC,names,tstamps = rawDataLoadBatch(par,'Select tissue ') # This approach is a bit rough, but there are no simple solutions

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

chrom_map = []
for op in op_fit_maps:
    chrom_map.append(chromFit(op,par)) # linear fitting for chromofores

#print('Saving data...')
#np.savez(par['savefile'],op_fit_maps=op_fit_maps,cal_R=cal_R,ROI=ROI) # save important results
#print('Done!')

op_ave,op_std = opticalSpectra(op_fit_maps,par,names)

for cm,name in zip(chrom_map,names):
    chromPlot(cm,name,par)