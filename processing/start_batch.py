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

Steps:
    - Select calibration phantom data folder
    - Select datasets base folder
    - Select chromophore reference file
    - Select a ROI on calibrated reflectance data
"""
import sys, os, re
import numpy as np
import numpy.ma as mask
import cv2 as cv
from scipy.io import savemat

sys.path.append('../common')
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work

from sfdi.readParams import readParams
from sfdi.crop import crop
from sfdi.getPath import getPath
from sfdi.getFile import getFile

from rawDataLoad import rawDataLoad
from calibrate import calibrate
from stackPlot import stackPlot
from fitOps import fitOps
from chromFit import chromFit
from utilities.motionCorrect import motionCorrect

par = readParams('parameters.ini')

if len(par['freq_used']) == 0:  # use all frequencies if empty
    par['freq_used'] = list(np.arange(len(par['freqs'])))

if len(par['wv_used']) == 0:  # use all wavelengths if empty
    par['wv_used'] = list(np.arange(len(par['wv'])))

## Load calibration phantom data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
ACph,_ = rawDataLoad(par,'Select calibration phantom data folder')
path = getPath('Select base folder')
if path:  # check for empty path
    dirs = [x for x in os.listdir(path) if os.path.isdir(path+'/'+x)]  # list of directories
    ##############################################################################
    ##  Define the folders to process here. Leave empty for interactive prompt  ##
    ##############################################################################
    toProcess = ['1601804882', '1601805054', '1601805170', '1601805243']
    
    if(not toProcess):  # In case you define by hand
        regex = input('Input base name to match (end with empty line): ').lower()  # only process matching directories
        while (regex != ''):  # End with an empty name
            toProcess.append(regex)
            regex = input('Input base name to match: ').lower()  # only process matching directories
    ## Some regex magic
    pattern = "|".join(re.escape(s) for s in toProcess)
    rexp = re.compile(pattern)
    dirs = [x for x in dirs if rexp.search(x.lower())]  # filter only matching names

cfile = getFile('Select chromophores reference file')

## Begin loop (one folder at a time)
for _d, dataset in enumerate(dirs):
    # Load tissue data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
    AC,_ = rawDataLoad(par, '{}/{}'.format(path, dataset), batch=True)

    ## Calibration step
    print('Dataset {} of {}'.format(_d+1, len(dataset)))
    print('Calibrating {}...'.format(dataset))
    if True:  # True to perform motion correction (slower)
        AC = motionCorrect(AC, par, edge='sobel', con=2, gauss=(7,5), debug=False)  # correct motion artifacts in raw data
    cal_R = calibrate(AC, ACph, par)

    ## True to mask background (e.g to remove black background that will return very high absorption)
    if False:
        th = 0.1  # threshold value (calculated on RED wavelength at fx=0)
        MASK = cal_R[:,:,-1,0] < th
        MASK = MASK.reshape((MASK.shape[0], MASK.shape[1], 1, 1))  # otherwise np.tile does not work correctly
        MASK = np.tile(MASK, (1, 1, cal_R.shape[2], cal_R.shape[3]))
        cal_R = np.ma.array(cal_R, mask=MASK)
    print('Calibration done.')
    
    if False:  # plot for DEBUG purposes
        stackPlot(cal_R[0],'magma')
        # sys.exit()
    
    ## Select only one ROI on the first calibration image
    if _d == 0:
        ROI = cv.selectROI('Select ROI', cal_R[0][:,:,0,0])  # press Enter to confirm selection
        cv.destroyWindow('Select ROI')
    
    ## Fitting for optical properties
    # TODO: this part is pretty computationally intensive, might be worth to optimize
    # Loop through different spatial frequencies
    if True:  # multi-frequencies approach
        FX = list(list(range(_i, _i+4)) for _i in range(len(par['freqs']) - 3))
    else:
        FX = [par['freq_used']]
    
    ## Save processing parameters on file
    if not os.path.exists(par['savefile']):  # check if path exists and create it
        os.mkdir(par['savefile'])
    
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
    with open('{}/processing_parameters.txt'.format(par['savefile']), 'w') as par_file:
        print('\n'.join(to_write), file=par_file)
    print('Parameters saved to file {}/processing_parameters.txt'.format(par['savefile']))
    
    # loop through frequencies sub-sets and fit
    for _f, fx in enumerate(FX):
        print('\nFrequency set {} of {}'.format(_f+1, len(FX)))
        par['freq_used'] = fx
        op_fit_maps = fitOps(crop(cal_R, ROI), par)
        
        if (len(par['chrom_used']) > 0):
            chrom_map = chromFit(op_fit_maps, par, cfile) # linear fitting for chromofores
            
        ## Save data to file
        if _f == 0:  # need to save cal_R only once
            if 'numpy' in par['savefmt']:
                np.savez('{}/{}_calR'.format(par['savefile'], dataset), cal_R=cal_R, ROI=ROI)
            if 'matlab' in par['savefmt']:
                savemat('{}/{}_calR'.format(par['savefile'], dataset), {'cal_R':cal_R, 'ROI':ROI})
        if 'numpy' in par['savefmt']:
            np.savez('{}/{}_f{}'.format(par['savefile'], dataset, _f), op_fit_maps=op_fit_maps.data)
            if (len(par['chrom_used']) > 0):
                np.savez('{}/{}_f{}_chrom'.format(par['savefile'], dataset, _f), chrom_map=chrom_map.data)
        if 'matlab' in par['savefmt']:
            savemat('{}/{}_f{}'.format(par['savefile'], dataset, _f), {'op_fit_maps':op_fit_maps.data})
            if (len(par['chrom_used']) > 0):
                savemat('{}/{}_f{}_chrom'.format(par['savefile'], dataset, _f), {'chrom_map':chrom_map.data})
        if len(par['savefmt']) > 0:
            print('{} saved'.format(dataset))
    print('Done!')