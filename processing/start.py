# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:16:24 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import sys, os
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
from utilities.motionCorrect import motionCorrect


par = readParams('parameters.ini')

if len(par['freq_used']) == 0: # use all frequencies if empty
    par['freq_used'] = list(np.arange(len(par['freqs'])))

if len(par['wv_used']) == 0: # use all wavelengths if empty
    par['wv_used'] = list(np.arange(len(par['wv'])))

## Load tissue data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
AC,name,_ = rawDataLoad(par, 'Select tissue data folder')
AC = motionCorrect(AC, par, edge='sobel', con=2, gauss=(7,5), debug=True)

## Load calibration phantom data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
ACph,_,_ = rawDataLoad(par, 'Select calibration phantom data folder')

## Calibration step
cal_R = calibrate(AC, ACph, par)

nn = name.split('/')[-1].split('_')[1]  # base file name

## Select ROI on calibration image
ROI = cv.selectROI('Select ROI', cal_R[:,:,0,0]) # press Enter to confirm selection
cv.destroyWindow('Select ROI')

## save ROI to file
with open('{}{}_ROI.csv'.format(par['savefile'], nn), 'w') as fname:
    np.array(ROI).tofile(fname, format='%d', sep=',')

## New: plot this after selecting ROI (smaller image)
stackPlot(crop(cal_R, ROI), 'magma')

## check if save directories exist and create them otherwise
if not os.path.exists(par['savefile']):
    os.makedirs(par['savefile'])
#if not os.path.exists('{}calR/'.format(par['savefile'])):
#    os.makedirs('{}calR/'.format(par['savefile']))

if True:  # multi-frequencies approach
    FX = list(list(range(_i, _i+4)) for _i in range(len(par['freqs']) - 3))
else:
    FX = [par['freq_used']]

## Save processing prameters to file
params = {'wv': np.array(par['wv'])[par['wv_used']],  # processed wavelengths
          'binsize': par['binsize'],  # pixel binning
          'ROI': ROI,  # processed ROI
          'fx': par['freqs'],  # all spatial frequencies
    }
to_write = ['# Parameters\nbinsize = {}\nROI = {}\nwv = {}nm\nfx = {}mm^-1'.format(
             params['binsize'], params['ROI'], params['wv'], params['fx'])]
for _f, fx in enumerate(FX):
    params['f{}'.format(_f)] = np.array(par['freqs'])[fx]  # partial fx
    to_write.append('f{} -> {}mm^-1\n'.format(_f, params['f{}'.format(_f)]))
with open('{}processing_parameters.txt'.format(par['savefile']), 'w') as par_file:
    print('\n'.join(to_write), file=par_file)
print('Parameters saved to file {}processing_parameters.txt'.format(par['savefile']))

## save calibrated reflectance data (matlab and npy format)
if 'numpy' in par['savefmt']:
    np.save('{}calR/{}'.format(par['savefile'], nn), crop(cal_R, ROI))  # numpy format
if 'matlab' in par['savefmt']:
    savemat('{}calR/{}'.format(par['savefile'], nn), {'calibrated_R':crop(cal_R, ROI)})  # matlab format

## DEBUG: stop here if you only want calibrated reflectance
# plt.savefig('{}calR/{}.png'.format(par['savefile'], nn))  # stack plot
# sys.exit()

## Fitting for optical properties
## TODO: this part is pretty computationally intensive, might be worth to optimize
op_fit_maps = fitOps(crop(cal_R[:,:,:,par['freq_used']],ROI),par)  # fit for all fx

## save optical properties to file. Remember to adjust the filename
if len(par['savefmt']) > 0:
    print('Saving data...')
    fullpath = '{}/{}'.format(par['savefile'], nn)
    if 'numpy' in par['savefmt']:
        np.savez('{}_OPmap_f0'.format(fullpath), op_fit_maps=op_fit_maps.data)  # numpy format
    if 'matlab' in par['savefmt']:
        savemat(fullpath, {'op_map':op_fit_maps.data})  # matlab format
    print('Done!')

#sys.exit()

chrom_map = chromFit(op_fit_maps, par) # linear fitting for chromophores. This is fast, no need to save
op_fit_maps,opt_ave,opt_std,radio = opticalSpectra(crop(cal_R[:,:,0,0], ROI), op_fit_maps, par, outliers=True)

#if not os.path.exists(par['savefile']):
#        os.mkdir(par['savefile'])
#np.save('{}{}_ave_{}fx.npy'.format(par['savefile'], nn, len(par['freq_used'])), opt_ave)
#np.save('{}{}_std_{}fx.npy'.format(par['savefile'], nn, len(par['freq_used'])), opt_std)

# TODO: loop through FX earlier and save
## Now looop through different fx combinations
#for i in range(5):
#    par['freq_used'] = [i,i+1,i+2,i+3] # select spatial frequencies
#    
#    op_fit_maps = fitOps(crop(cal_R[:,:,:,par['freq_used']], ROI), par)  # fit optical properties
#    chrom_map = chromFit(op_fit_maps,par)  # linear fitting for chromofores
#    
#    ## save optical properties to file. Remember to adjust the filename
#    print('Saving data...')
#    suffix = '{}'.format(i)  # suffix = loop iteration
#    fullpath = '{}{}_OPmap_{}wv{}'.format(par['savefile'], nn, len(par['freq_used']), suffix)
#    savemat(fullpath, {'op_map':op_fit_maps.data})  # matlab format
#    np.save(fullpath, op_fit_maps.data)  # numpy format
#    print('Done!')

## Plots
#op_fit_maps,op_ave,op_std,radio = opticalSpectra(crop(cal_R[:,:,0,0], ROI), op_fit_maps, par, outliers=True)
#chrom_map = chromPlot(chrom_map, name.split('/')[-1], par)