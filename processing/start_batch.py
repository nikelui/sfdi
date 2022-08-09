# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:16:24 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Script for batch processing the data.

Steps before starting:
    - Check the parameters.ini file
    - Fill the "toProcess" array (or leave blank for interactive mode)
    - Turn motion correction to True or False
    - Turn background masking to True or False
    - Turn multi-frequencies to True or False
    - Select if data sample is homogeneous or not in fitOps

Steps:
    - Select calibration phantom data folder
    - Select datasets base folder
    - Select chromophore reference file
    - Select a ROI on calibrated reflectance data
"""
import os, re
import numpy as np
import numpy.ma as mask
import cv2 as cv
from scipy.io import savemat

from sfdi.common.readParams import readParams
from sfdi.common.getPath import getPath
from sfdi.common.getFile import getFile
from sfdi.processing.crop import crop
from sfdi.processing.rawDataLoad import rawDataLoad
from sfdi.processing.calibrate import calibrate
from sfdi.processing.stackPlot import stackPlot
from sfdi.processing.fitOps import fitOps
from sfdi.processing.chromFit import chromFit
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
    par['freqs'] = apar['fx']
    par['nphase'] = apar['nphase']
    par['wv'] = apar['wv']

if len(par['freq_used']) == 0:  # use all frequencies if empty
    par['freq_used'] = list(np.arange(len(par['freqs'])))

if len(par['wv_used']) == 0:  # use all wavelengths if empty
    par['wv_used'] = list(np.arange(len(par['wv'])))
ACph,_ = rawDataLoad(par, phantom_path, batch=True)

path = getPath('Select base folder')
if path:  # check for empty path
    dirs = [x for x in os.listdir(path) if os.path.isdir(path+'/'+x)]  # list of directories
    ##############################################################################
    ##  Define the folders to process here. Leave empty for interactive prompt  ##
    ##############################################################################
    toProcess = ['AlObase', 'TiObase', 'TiO05ml', 'TiO10ml', 'TiO15ml', 'TiO20ml', 'TiO30ml']
    if(not toProcess):  # In case you define by hand
        regex = input('Input base name to match (end with empty line): ').lower()  # only process matching directories
        while (regex != ''):  # End with an empty name
            toProcess.append(regex)
            regex = input('Input base name to match: ').lower()  # only process matching directories
    
    ## Some regex magic
    pattern = "|".join(re.escape(s) for s in toProcess)
    rexp = re.compile(pattern)
    dirs = [x for x in dirs if rexp.search(x)]  # filter only matching names

if (len(par['chrom_used']) > 0):
    cfile = getFile('Select chromophores reference file')  # Get chromophores reference file

ph_name = phantom_path.split('/')[-1].split('_')[-2]
cphantom = ['{}/{}.txt'.format(ph_path._path[0], ph_name)]
if not os.path.exists(cphantom[0]):
    cphantom = []  # pass an empty list to get interactive prompt
#%%
## Begin loop (one folder at a time)
for _d, dataset in enumerate(dirs):
    # Load tissue data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
    print('Dataset {} of {}'.format(_d+1, len(dirs)))
    AC,_ = rawDataLoad(par, '{}/{}'.format(path, dataset), batch=True)

    ## Calibration step
    print('Calibrating {}...'.format(dataset))
    if False:  # True to perform motion correction (slower)
        AC = motionCorrect(AC, par, edge='sobel', con=2, gauss=(7,5), debug=False)  # correct motion artifacts in raw data
        
    cal_R = calibrate(AC, ACph, par, path=cphantom)

    ## True to mask background (e.g to remove black background that will return very high absorption)
    if False:
        th = 0.1  # threshold value (calculated on RED wavelength at fx=0)
        MASK = cal_R[:,:,-1,0] < th
        MASK = MASK.reshape((MASK.shape[0], MASK.shape[1], 1, 1))  # otherwise np.tile does not work correctly
        MASK = np.tile(MASK, (1, 1, cal_R.shape[2], cal_R.shape[3]))
        cal_R = np.ma.array(cal_R, mask=MASK)
    print('Calibration done.')
    
    if False:  # plot for DEBUG purposes
        stackPlot(cal_R,'magma')
        # sys.exit()
    
    ## Select only one ROI on the first calibration image
    if _d == 0:
        ROI = cv.selectROI('Select ROI', cal_R[:,:,0,0])  # press Enter to confirm selection
        cv.destroyWindow('Select ROI')
    
    ## Fitting for optical properties
    # TODO: this part is pretty computationally intensive, might be worth to optimize
    # Loop through different spatial frequencies
    if True:  # multi-frequencies approach
        FX = list(list(range(_i, _i+4)) for _i in range(len(par['freqs']) - 3))
    else:
        FX = [par['freq_used']]
    
    ## Save processing parameters on file
    if _d == 0:
        if not os.path.exists(par['savefile']):  # check if path exists and create it
            os.mkdir(par['savefile'])
    
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
    
    # loop through frequencies sub-sets and fit
    for _f, fx in enumerate(FX):
        print('\nFrequency set {} of {}'.format(_f+1, len(FX)))
        par['freq_used'] = fx
        if _f == 0:  # in case of multi-fx, save initial mua
            op_fit_maps = fitOps(crop(cal_R, ROI), par)
            # Initial guess is based on mua, mus median value calculated in a ROI at the center
            X0, Y0 = np.array(op_fit_maps.shape[:2])//2  # coordinates of the center
            W = 5  # ROI half-width
            op_guess = np.squeeze(np.nanmedian(op_fit_maps[X0-W:X0+W,Y0-W:Y0+W,:,:], axis=(0,1)))       
            # op_guess = np.random.rand(5,2)  # DEBUG
            WV = np.array(par['wv'])[par['wv_used'], np.newaxis]
            op_guess = np.append(WV, op_guess, axis=1)  # use mua, mus at f0 as initial guess
        else:
            op_fit_maps = fitOps(crop(cal_R, ROI), par, guess=op_guess, homogeneous=False)
        
        if (len(par['chrom_used']) > 0):
            chrom_map = chromFit(op_fit_maps, par, cfile) # linear fitting for chromofores
            
        ## Save data to file
        if _f == 0:  # need to save cal_R only once
            if 'numpy' in par['savefmt']:
                np.savez('{}/{}_calR'.format(par['savefile'], dataset), cal_R=cal_R, ROI=ROI)
            if 'matlab' in par['savefmt']:
                savemat('{}/{}_calR.mat'.format(par['savefile'], dataset), {'cal_R':cal_R, 'ROI':ROI})
        if 'numpy' in par['savefmt']:
            np.savez('{}/{}_f{}'.format(par['savefile'], dataset, _f), op_fit_maps=op_fit_maps.data)
            if (len(par['chrom_used']) > 0):
                np.savez('{}/{}_f{}_chrom'.format(par['savefile'], dataset, _f), chrom_map=chrom_map.data)
        if 'matlab' in par['savefmt']:
            savemat('{}/{}_f{}.mat'.format(par['savefile'], dataset, _f), {'op_fit_maps':op_fit_maps.data})
            if (len(par['chrom_used']) > 0):
                savemat('{}/{}_f{}_chrom.mat'.format(par['savefile'], dataset, _f), {'chrom_map':chrom_map.data})
        if len(par['savefmt']) > 0:
            print('{} saved'.format(dataset))
    print('Done!')