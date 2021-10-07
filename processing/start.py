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
from sfdi.common.getPath import getPath
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
from sfdi.common.phantoms import __path__ as ph_path  # phantoms reference data path

par = readParams('{}/parameters.ini'.format(par_path[0]))

if len(par['freq_used']) == 0: # use all frequencies if empty
    par['freq_used'] = list(np.arange(len(par['freqs'])))

if len(par['wv_used']) == 0: # use all wavelengths if empty
    par['wv_used'] = list(np.arange(len(par['wv'])))

## Load calibration phantom data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
phantom_path = getPath('Select calibration phantom data folder')
ACph,_ = rawDataLoad(par, phantom_path, batch=True)

## Load tissue data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
data_path = getPath('Select tissue data folder')
AC,_ = rawDataLoad(par, data_path, batch=True)

if False:
    AC = motionCorrect(AC, par, edge='sobel', con=2, gauss=(7,5), debug=True)

if (len(par['chrom_used']) > 0):
    cfile = getFile('Select chromophores reference file')

# Update parameter if acquisition_parameters is found
a_path = '{}/acquisition_parameters.ini'.format('/'.join(phantom_path.split('/')[:-1]))
if os.path.exists(a_path):
    apar = readParams(a_path)
    par['fx'] = apar['fx']
    par['nphase'] = apar['nphase']
    par['wv'] = apar['wv']

## Calibration step
ph_name = phantom_path.split('/')[-1].split('_')[-2]
cphantom = ['{}/{}.txt'.format(ph_path._path[0], ph_name)]

if not os.path.exists(cphantom[0]):  # if phantom reference file is not found
    cphantom = []  # pass an empty list to get interactive prompt
cal_R = calibrate(AC, ACph, par, path=cphantom)

## True to mask background (e.g to remove black background that will return very high absorption)
if False:
    th = 0.1  # threshold value (calculated on RED wavelength at fx=0)
    MASK = cal_R[:,:,-1,0] < th
    MASK = MASK.reshape((MASK.shape[0], MASK.shape[1], 1, 1))  # otherwise np.tile does not work correctly
    MASK = np.tile(MASK, (1, 1, cal_R.shape[2], cal_R.shape[3]))
    cal_R = np.ma.array(cal_R, mask=MASK)

nn = data_path.split('/')[-1].split('_')[-2]  # base file name

## Select ROI on calibration image
ROI = cv.selectROI('Select ROI', cal_R[:,:,0,0])  # press Enter to confirm selection
cv.destroyWindow('Select ROI')

## New: plot this after selecting ROI (smaller image)
stackPlot(crop(cal_R, ROI), 'magma')
#%%
## check if save directories exist and create them otherwise
if not os.path.exists(par['savefile']):
    os.makedirs(par['savefile'])

if False:  # multi-frequencies approach
    FX = list(list(range(_i, _i+4)) for _i in range(len(par['freqs']) - 3))
else:
    FX = [par['freq_used']]

## Save processing prameters to file
params = {'wv': np.array(par['wv'])[par['wv_used']],  # processed wavelengths
          'binsize': par['binsize'],  # pixel binning
          'ROI': list(ROI),  # processed ROI
          'fx': par['freqs'],  # all spatial frequencies
    }
to_write = ['# Parameters\nbinsize = {}\nROI = {}\nwv = {}nm\nfx = {}mm^-1'.format(
             params['binsize'], params['ROI'], params['wv'], params['fx'])]
for _f, fx in enumerate(FX):
    params['f{}'.format(_f)] = np.array(par['freqs'])[fx]  # partial fx
    to_write.append('f{} = {}mm^-1\n'.format(_f, params['f{}'.format(_f)]))
with open('{}/processing_parameters.txt'.format(par['savefile']), 'w') as par_file:
    print('\n'.join(to_write), file=par_file)
print('Parameters saved to file {}/processing_parameters.txt'.format(par['savefile']))

## save calibrated reflectance data (matlab and npy format)
# if 'numpy' in par['savefmt']:
#     np.save('{}/calR/{}'.format(par['savefile'], nn), crop(cal_R, ROI))  # numpy format
# if 'matlab' in par['savefmt']:
#     savemat('{}/calR/{}'.format(par['savefile'], nn), {'calibrated_R':crop(cal_R, ROI)})  # matlab format

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
        np.savez('{}/{}_calR'.format(par['savefile'], nn), cal_R=cal_R, ROI=ROI)
        if (len(par['chrom_used']) > 0):
            np.savez('{}/{}_f{}_chrom'.format(par['savefile'], nn, _f), chrom_map=chrom_map.data)
    if 'matlab' in par['savefmt']:
        savemat('{}/{}_f{}'.format(par['savefile'], nn, _f), {'op_fit_maps':op_fit_maps.data})
        savemat('{}/{}_calR'.format(par['savefile'], nn), {'cal_R':cal_R, 'ROI':ROI})
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


