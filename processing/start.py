# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:16:24 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Script for processing SFDI data

Steps before starting:
    - Check the parameters.ini file
    - Turn motion correction to True or False
    - Turn background masking to True or False
    - Turn multi-frequencies to True or False

Steps:
    - Select calibration phantom data folder
    - Select tissue data folder
    - Select chromophore reference file
    - Select a ROI on calibrated reflectance data
"""
import os
import numpy as np
import cv2 as cv
from scipy.io import savemat

from sfdi.common.readParams import readParams
from sfdi.common.getFile import getFile
from sfdi.processing.crop import crop
from sfdi.processing.rawDataLoad import rawDataLoad
from sfdi.processing.calibrate import calibrate
from sfdi.processing.stackPlot import stackPlot
from sfdi.processing.fitOps import fitOps
from sfdi.processing.chromFit import chromFit
from sfdi.processing.chromPlot import chromPlot
from sfdi.processing.opticalSpectra import opticalSpectra
from sfdi.processing.motionCorrect import motionCorrect


from sfdi.processing import __path__ as par_path  # processing parameters path
par = readParams('{}/parameters.ini'.format(par_path))

if len(par['freq_used']) == 0: # use all frequencies if empty
    par['freq_used'] = list(np.arange(len(par['freqs'])))

if len(par['wv_used']) == 0: # use all wavelengths if empty
    par['wv_used'] = list(np.arange(len(par['wv'])))

## Load calibration phantom data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
ACph,_,_ = rawDataLoad(par, 'Select calibration phantom data folder')

## Load tissue data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
AC,name,_ = rawDataLoad(par, 'Select tissue data folder')
if True:
    AC = motionCorrect(AC, par, edge='sobel', con=2, gauss=(7,5), debug=True)

cfile = getFile('Select chromophores reference file')

## Calibration step
cal_R = calibrate(AC, ACph, par)

## True to mask background (e.g to remove black background that will return very high absorption)
if False:
    th = 0.1  # threshold value (calculated on RED wavelength at fx=0)
    MASK = cal_R[:,:,-1,0] < th
    MASK = MASK.reshape((MASK.shape[0], MASK.shape[1], 1, 1))  # otherwise np.tile does not work correctly
    MASK = np.tile(MASK, (1, 1, cal_R.shape[2], cal_R.shape[3]))
    cal_R = np.ma.array(cal_R, mask=MASK)

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

# loop through frequencies sub-sets and fit
for _f, fx in enumerate(FX):
    print('\nFrequency set {} of {}'.format(_f+1, len(FX)))
    par['freq_used'] = fx
    op_fit_maps = fitOps(crop(cal_R, ROI), par)
    
    if (len(par['chrom_used']) > 0):
        chrom_map = chromFit(op_fit_maps, par, cfile) # linear fitting for chromofores
        
    ## Save data to file
    if 'numpy' in par['savefmt']:
        np.savez('{}/{}_f{}'.format(par['savefile'], nn, _f), op_fit_maps=op_fit_maps.data)
        if (len(par['chrom_used']) > 0):
            np.savez('{}/{}_f{}_chrom'.format(par['savefile'], nn, _f), chrom_map=chrom_map.data)
    if 'matlab' in par['savefmt']:
        savemat('{}/{}_f{}'.format(par['savefile'], nn, _f), {'op_fit_maps':op_fit_maps.data})
        if (len(par['chrom_used']) > 0):
            savemat('{}/{}_f{}_chrom'.format(par['savefile'], nn, _f), {'chrom_map':chrom_map.data})
    if len(par['savefmt']) > 0:
        print('{} saved'.format(nn))
print('Done!')

# Interactive plot
# TODO: save all datasets and allow to choose one?
op_fit_maps,opt_ave,opt_std,radio = opticalSpectra(crop(cal_R[:,:,0,0], ROI), op_fit_maps, par, outliers=True)
# chrom_map = chromPlot(chrom_map, name.split('/')[-1], par)

## Save average optical properties to file
# if not os.path.exists(par['savefile']):
#         os.mkdir(par['savefile'])
# np.save('{}{}_ave_{}fx.npy'.format(par['savefile'], nn, len(par['freq_used'])), opt_ave)
# np.save('{}{}_std_{}fx.npy'.format(par['savefile'], nn, len(par['freq_used'])), opt_std)


