# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:16:24 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.io import savemat

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
AC,name = rawDataLoad(par, 'Select tissue data folder')

## Load calibration phantom data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
ACph,_ = rawDataLoad(par, 'Select calibration phantom data folder')

# Calibration step
cal_R = calibrate(AC, ACph, par)

## Select ROI on calibration image
ROI = cv.selectROI('Select ROI', cal_R[:,:,0,0]) # press Enter to confirm selection

nn = name.split('/')[-1].split('_')[1]  # base file name

# save ROI to file
#fname = open(par['savefile']+'ROI.csv','w')
#np.array(ROI).tofile(fname,format='%d',sep=',')
#fname.close()
#cv.destroyWindow('Select ROI')

# New: plot this after selecting ROI (smaller image)
stackPlot(crop(cal_R, ROI), 'magma')
#plt.savefig(par['savefile'] + '/' + name.split('/')[-1].split('_')[1] + '_CalR.png')

## save calibrated reflectance data (matlab and npy format)

savemat('{}calR/{}'.format(par['savefile'], nn), {'calibrated_R':crop(cal_R, ROI)})  # matlab format
np.save('{}calR/{}'.format(par['savefile'], nn), crop(cal_R, ROI))  # numpy format

## DEBUG: stop here if you only want calibrated reflectance
#sys.exit()

## Fitting for optical properties
# TODO: this part is pretty computationally intensive, might be worth to optimize
## TODO: putting in a loop  to fit for all combination of fx. Remove afterwards
# First, fit at all fx
op_fit_maps = fitOps(crop(cal_R[:,:,:,par['freq_used']],ROI),par)

# save optical properties to file
print('Saving data...')
fullpath = '{}{}_OPmap_{}wv'.format(par['savefile'], nn, len(par['freq_used']))
savemat(fullpath, {'op_map'=op_fit_maps.data})
np.save(fullpath,op_fit_maps.data)  # numpy format
print('Done!')

chrom_map = chromFit(op_fit_maps,par) # linear fitting for chromophores. This is fast, no need to save

op_fit_maps,opt_ave,opt_std,radio = opticalSpectra(crop(cal_R[:,:,0,0],ROI),op_fit_maps,par)

# Now looop throug different fx combinations
#for i in range(5):
#    par['freq_used'] = [i,i+1,i+2,i+3] # select spatial frequencies
#    op_fit_maps = fitOps(crop(cal_R[:,:,:,par['freq_used']],ROI),par)
#    
#    chrom_map = chromFit(op_fit_maps,par) # linear fitting for chromofores
#    
#    print('Saving data...')
#    fullpath = (par['savefile'] + name.split('/')[-1].split('_')[1] +
#                '_OPmap_3wv_%d' % i)
#    np.save(fullpath,op_fit_maps.data)
#    #np.savez(par['savefile']+,op_fit_maps=op_fit_maps,cal_R=cal_R,ROI=ROI,chrom_map=chrom_map) # save important results
#    print('Done!')

#op_fit_maps,op_ave,op_std,radio = opticalSpectra(crop(cal_R[:,:,0,0],ROI),op_fit_maps,par,outliers=True)
chrom_map = chromPlot(chrom_map,name.split('/')[-1],par)
