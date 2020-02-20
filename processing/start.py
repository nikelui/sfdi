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

from sfdi.readParams3 import readParams
from sfdi.crop import crop
from rawDataLoad import rawDataLoad
from calibrate import calibrate
from stackPlot import stackPlot
from fitOps import fitOps
from chromFit import chromFit
from chromPlot import chromPlot
from opticalSpectra import opticalSpectra


par = readParams('parameters.ini')

if len(par['freq_used']) == 0: # use all frequencies if empty
    par['freq_used'] = list(np.arange(len(par['freqs'])))

# Load tissue data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
AC,name = rawDataLoad(par,'Select tissue data folder')

## Load calibration phantom data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
ACph,_ = rawDataLoad(par,'Select calibration phantom data folder')

# Calibration step
cal_R = calibrate(AC,ACph,par)

stackPlot(cal_R,'magma')


## Select ROI on calibration image
ROI = cv.selectROI('Select ROI',cal_R[:,:,0,0]) # press Enter to confirm selection
cv.destroyWindow('Select ROI')

## Fitting for optical properties
# TODO: this part is pretty computationally intensive, might be worth to optimize
op_fit_maps = fitOps(crop(cal_R,ROI),par)

chrom_map = chromFit(op_fit_maps,par) # linear fitting for chromofores

#print('Saving data...')
#np.savez(par['savefile'],op_fit_maps=op_fit_maps,cal_R=cal_R,ROI=ROI,chrom_map=chrom_map) # save important results
#print('Done!')

op_fit_maps,op_ave,op_std,radio = opticalSpectra(crop(cal_R[:,:,0,0],ROI),op_fit_maps,par,outliers=True)
chrom_map = chromPlot(chrom_map,name.split('/')[-1],par)
