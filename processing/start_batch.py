# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:16:24 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import sys, os
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
from opticalSpectra import opticalSpectra as oss  # DEBUG

par = readParams('parameters.ini')

if len(par['freq_used']) == 0: # use all frequencies if empty
    par['freq_used'] = list(np.arange(len(par['freqs'])))

if len(par['wv_used']) == 0: # use all wavelengths if empty
    par['wv_used'] = list(np.arange(len(par['wv'])))

## Load calibration phantom data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
ACph,_ = rawDataLoad(par,'Select calibration phantom data folder')

# Load tissue data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
AC,names,tstamps = rawDataLoadBatch(par, 'Select tissue') # This approach is a bit rough, but there are no simple solutions

# Calibration step
cal_R = []
for ac in AC:
    c_R = calibrate(ac,ACph,par)

    ### True here to mask background
    if False:
        th = 0.1  # threshold value (calculated on RED wavelength at fx=0)
        mask = c_R[:,:,-1,0] < th
        mask = mask.reshape((mask.shape[0], mask.shape[1], 1, 1))  # otherwise np.tile does not work correctly
        mask = np.tile(mask, (1, 1, c_R.shape[2], c_R.shape[3]))
        c_R = np.ma.array(c_R, mask=mask)
    
    cal_R.append(c_R)

stackPlot(cal_R[0],'magma') # Maybe show all? Or none

#sys.exit()

## Select only one ROI on calibration image
ROI = cv.selectROI('Select ROI',cal_R[0][:,:,0,0]) # press Enter to confirm selection
cv.destroyWindow('Select ROI')

## Fitting for optical properties
# TODO: this part is pretty computationally intensive, might be worth to optimize
# Loop through different spatial frequencies
FX = ([0,1,2,3], [3,4,5,6], [5,6,7,8])
for _f,fx in enumerate(FX):
    par['freq_used'] = fx
    op_fit_maps = []
    for cal in cal_R:
        op_fit_maps.append(fitOps(crop(cal,ROI),par))
    
    if (len(par['chrom_used'])>0):
        chrom_map = []
        for op in op_fit_maps:
            chrom_map.append(chromFit(op,par)) # linear fitting for chromofores

#    op_ave,op_std = opticalSpectra(op_fit_maps,par,names,outliers=True,roi=False)
#    op_fit_maps[0],opt_ave,opt_std,radio = oss(crop(cal_R[0][:,:,0,0],ROI), op_fit_maps[0],par,outliers=False)

    print('Saving data...')
    # check if path exists and create it
    if not os.path.exists(par['savefile']):
        os.mkdir(par['savefile'])
        
    for _i,name in enumerate(names):  # save individual files
        if _f == 0:  # need to save only once
            np.savez('{}{}_calR'.format(par['savefile'], name), cal_R=cal_R[_i], ROI=ROI)
        np.savez('{}{}_f{}'.format(par['savefile'], name, _f), op_fit_maps=op_fit_maps[_i].data)
        print('{} saved'.format(name))
        #np.savez('{}processed'.format(par['savefile']),op_fit_maps=op_fit_maps,cal_R=cal_R,ROI=ROI)
    #np.savez(par['savefile'],op_ave=op_ave,op_std=op_std)
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