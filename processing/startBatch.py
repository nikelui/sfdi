# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:00:42 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

## NOTE: old batch processing, not developed anymore. Please use start_batch.py instead

import sys, os
import numpy as np
import json
from scipy.io import savemat

sys.path.append('../common')
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work

from sfdi.readParams3 import readParams
from sfdi.getFile import getFile
from fitOps import fitOps
from chromFit import chromFit

par = readParams('parameters.ini')

if len(par['freq_used']) == 0: # use all frequencies if empty
    par['freq_used'] = list(np.arange(len(par['freqs'])))

if len(par['wv_used']) == 0: # use all wavelengths if empty
    par['wv_used'] = list(np.arange(len(par['wv'])))

batchpath = getFile('Select batch file list')

with open(batchpath, 'r') as bfile:
    batch_list = json.loads(bfile.read())

## loop over batch files
for item in batch_list:
    par['freq_used'] = [0,1,2,3,4,5,6,7]  # reset frequencies
    nn = item.split('/')[-1].split('.')[0]  # base file name
    #tstamp = int(item.split('/')[-1].split('_')[0])  # timestamp
    
    ## load calibrated reflectance data
    cal_R = np.load(item)
      
    ## check if save directories exist and create them otherwise
    if not os.path.exists(par['savefile']):
        os.makedirs(par['savefile'])

    ## Fitting for optical properties
    ## TODO: this part is pretty computationally intensive, might be worth to optimize
    op_fit_maps = fitOps(cal_R[:,:,:,par['freq_used']], par)  # fit for all fx
    
    ## save optical properties to file. Remember to adjust the filename
    print('Saving data...')
    suffix = ''  # no suffix if all wv
    fullpath = '{}{}_OPmap_{}wv{}'.format(par['savefile'], nn, len(par['freq_used']), suffix)
#    savemat(fullpath, {'op_map':op_fit_maps.data})  # matlab format
    np.save(fullpath, op_fit_maps.data)  # numpy format
    print('Done!')
    
    #chrom_map = chromFit(op_fit_maps, par) # linear fitting for chromophores. This is fast, no need to save
    
    ## Now looop through different fx combinations
#    for i in range(5):
#        par['freq_used'] = [i,i+1,i+2,i+3] # select spatial frequencies
#        
#        op_fit_maps = fitOps(cal_R[:,:,:,par['freq_used']], par)  # fit optical properties
#        #chrom_map = chromFit(op_fit_maps,par)  # linear fitting for chromofores
#        
#        ## save optical properties to file. Remember to adjust the filename
#        print('Saving data...')
#        suffix = '{}'.format(i)  # suffix = loop iteration
#        fullpath = '{}{}_OPmap_{}wv{}'.format(par['savefile'], nn, len(par['freq_used']), suffix)
##        savemat(fullpath, {'op_map':op_fit_maps.data})  # matlab format
#        np.save(fullpath, op_fit_maps.data)  # numpy format
#        print('Done!')