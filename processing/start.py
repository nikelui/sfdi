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
    - Select if dataset is homogeneous (fix mua at f0)

Steps:
    - Select calibration phantom data folder
    - Select tissue data folder
    - Select chromophore reference file [if processed]
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

## Load calibration phantom data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
phantom_path = getPath('Select calibration phantom data folder')
# Update parameter if acquisition_parameters is found
a_path = '{}/acquisition_parameters.ini'.format('/'.join(phantom_path.split('/')[:-1]))
if os.path.exists(a_path):
    apar = readParams(a_path)
    par['fx'] = apar['fx']
    par['nphase'] = apar['nphase']
    par['wv'] = apar['wv']

if len(par['freq_used']) == 0: # use all frequencies if empty
    par['freq_used'] = list(np.arange(len(par['freqs'])))

if len(par['wv_used']) == 0: # use all wavelengths if empty
    par['wv_used'] = list(np.arange(len(par['wv'])))

ACph,_ = rawDataLoad(par, phantom_path, batch=True)

## Load tissue data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
data_path = getPath('Select tissue data folder')
AC,_ = rawDataLoad(par, data_path, batch=True)

if False:
    AC = motionCorrect(AC, par, edge='sobel', con=2, gauss=(7,5), debug=True)

if (len(par['chrom_used']) > 0):
    cfile = getFile('Select chromophores reference file')

# get processing parameters, if exist from previous datasets
p_path = '{}/processing_parameters.ini'.format(par['savefile'])
if os.path.exists(p_path):
    ppar = readParams(p_path)

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

nn = data_path.split('/')[-1]  # base file name
#%%
## Select ROI on calibration image
# New: if a ROI already exists from previous datasets, you can choose to keep it
if True and os.path.exists(p_path) and 'roi' in ppar.keys():
    ROI = ppar['roi']
else:
    ROI = cv.selectROI('Select ROI', cal_R[:,:,0,0])  # press Enter to confirm selection
    cv.destroyWindow('Select ROI')

## New: plot this after selecting ROI (smaller image)
stackPlot(crop(cal_R, ROI), 'magma')

## check if save directories exist and create them otherwise
if not os.path.exists(par['savefile']):
    os.makedirs(par['savefile'])

if True:  # multi-frequencies approach
    FX = list(list(range(_i, _i+4)) for _i in range(len(par['freqs']) - 3))
    # FX = [[0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,6]]  # DEBUG
else:
    FX = [par['freq_used']]

## Save processing parameters to file
params = {'wv': np.array(par['wv'])[par['wv_used']],  # processed wavelengths
          'binsize': par['binsize'],  # pixel binning
          'ROI': list(ROI),  # processed ROI
          'fx': par['freqs'],  # all spatial frequencies
    }
to_write = ['[DEFAULT]\n# Parameters\nbinsize = {}\nROI = {}\nwv = {}\nfx = {}'.format(
             params['binsize'], params['ROI'], list(params['wv']), params['fx'])]
for _f, fx in enumerate(FX):
    params['f{}'.format(_f)] = np.array(par['freqs'])[fx]  # partial fx
    to_write.append('f{} = {}'.format(_f, list(params['f{}'.format(_f)])))
with open('{}/processing_parameters.ini'.format(par['savefile']), 'w') as par_file:
    print('\n'.join(to_write), file=par_file)
print('Parameters saved to file {}/processing_parameters.ini'.format(par['savefile']))

## DEBUG: stop here if you only want calibrated reflectance
# plt.savefig('{}calR/{}.png'.format(par['savefile'], nn))  # stack plot
# sys.exit()

## Fitting for optical properties
## TODO: this part is pretty computationally intensive, might be worth to optimize

# loop through frequencies sub-sets and fit
for _f, fx in enumerate(FX):
    print('\nFrequency set {} of {}'.format(_f+1, len(FX)))
    par['freq_used'] = fx
    
    if _f == 0:  # in case of multi-fx, save initial mua
        op_fit_maps = fitOps(crop(cal_R, ROI), par, homogeneous=False)
        # Initial guess is based on mua, mus median value calculated in a ROI at the center
        X0, Y0 = np.array(op_fit_maps.shape[:2])//2  # coordinates of the center
        W = 5  # ROI half-width
        op_guess = np.squeeze(np.nanmedian(op_fit_maps[X0-W:X0+W,Y0-W:Y0+W,:,:], axis=(0,1)))       
        # op_guess = np.random.rand(5,2)  # DEBUG
        WV = np.array(par['wv'])[par['wv_used'], np.newaxis]
        op_guess = np.append(WV, op_guess, axis=1)  # use mua, mus at f0 as initial guess
    else:
        op_fit_maps = fitOps(crop(cal_R, ROI), par, guess=op_guess, homogeneous=True)
    
    if (len(par['chrom_used']) > 0):
        chrom_map = chromFit(op_fit_maps, par, cfile) # linear fitting for chromofores
        
    ## Save data to file
    if 'numpy' in par['savefmt']:
        np.savez('{}/{}_f{}'.format(par['savefile'], nn, _f), op_fit_maps=op_fit_maps.data)
        np.savez('{}/{}_calR'.format(par['savefile'], nn), cal_R=crop(cal_R,ROI), ROI=ROI)
        if (len(par['chrom_used']) > 0):
            np.savez('{}/{}_f{}_chrom'.format(par['savefile'], nn, _f), chrom_map=chrom_map.data)
    if 'matlab' in par['savefmt']:
        savemat('{}/{}_f{}.mat'.format(par['savefile'], nn, _f), {'op_fit_maps':op_fit_maps.data})
        savemat('{}/{}_calR.mat'.format(par['savefile'], nn), {'cal_R':crop(cal_R,ROI), 'ROI':ROI})
        if (len(par['chrom_used']) > 0):
            savemat('{}/{}_f{}_chrom.mat'.format(par['savefile'], nn, _f), {'chrom_map':chrom_map.data})
    if len(par['savefmt']) > 0:
        print('{} saved'.format(nn))
print('Done!')

# Interactive plot
# TODO: save all datasets and allow to choose one?
# op_fit_maps,opt_ave,opt_std,radio = opticalSpectra(crop(cal_R[:,:,0,0], ROI), op_fit_maps, par, outliers=True)
# chrom_map = chromPlot(chrom_map, name.split('/')[-1], par)

## Save average optical properties to file
# if not os.path.exists(par['savefile']):
#         os.mkdir(par['savefile'])
# np.save('{}{}_ave_{}fx.npy'.format(par['savefile'], nn, len(par['freq_used'])), opt_ave)
# np.save('{}{}_std_{}fx.npy'.format(par['savefile'], nn, len(par['freq_used'])), opt_std)


