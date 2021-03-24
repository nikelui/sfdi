# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 08:28:24 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import os,sys
import numpy as np
import cv2 as cv

sys.path.append('./common')
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work

from sfdi.readParams3 import readParams
from sfdi.getPath import getPath

def rawDataLoad(par, prompt='Select folder', batch=False):
    """Select a folder to load the images contained inside.
par: Dictionary containing all the processing parameters
prompt: optional string for file dialog. If run in batch mode it is the file path instead
batch: flag to pass if run in batch mode (e.g. from startBatch)
"""
    if batch:
        path = prompt  # TODO: find a less ambiguous variable name
    else:
        path = getPath(prompt)
    
    if len(path) > 0: # check for empty path
        intT = float(path.split('/')[-1].split('_')[-1][:-2]) # exposure time
    else:
        sys.exit()
    
    files = [x for x in os.listdir(path) if '.bmp' in x]
    try:  # Old naming convention -> xxx_000.bmp
        _ = int(files[0].split('_')[-1][:-4])  
        files.sort()
    except ValueError:  # New naming convention -> xxx_0-0-0.bmp
        files.sort(key=lambda x: list(map(int, x.split('_')[-1][:-4].split('-')))) 
    
    # initialize 3 phase AC data structure
    temp = np.zeros((par['ylength'], par['xlength'], par['nphase']), dtype='float')  # To read each n-phase image
    AC = np.zeros((int(par['ylength']), int(par['xlength']), len(par['wv']),
                   len(par['freqs'])), dtype='float')  #try to adopt this as standard data format
    DC = np.zeros((int(par['ylength']), int(par['xlength']), len(par['wv']),
                   len(par['freqs'])), dtype='float')  #try to adopt this as standard data format
    
    for i in range(len(par['wv'])):
        print('loading all frequencies for wavelength: {} nm'.format(par['wv'][i]))
        for j in range(len(par['freqs'])):
            for p in range(par['nphase']):
                fname = files[p + j * par['nphase'] + i * len(par['freqs']) * par['nphase']]
                temp[:,:,p] = cv.imread('{}/{}'.format(path, fname), cv.IMREAD_GRAYSCALE)

#            AC[:,:,i,j] = np.sqrt(2)/3 * np.sqrt( (temp[:,:,0]-temp[:,:,1])**2 +
#                    (temp[:,:,1]-temp[:,:,2])**2 +
#                    (temp[:,:,2]-temp[:,:,0])**2 ) / intT # normalize by exposure time
            
            ## New AC demodulation, with vectorialization. Allows to use n-phase instead of 3
            temp = np.dstack((temp,temp[:,:,0])) # append the first element again at the end
            AC[:,:,i,j] = 1 * np.sqrt(np.sum(np.diff(temp,axis=2)**2,axis=2)) / intT
            ##TODO: XX is the correct normalization term (depends on nPhase?)
            DC[:,:,i,j] = np.mean(temp[:,:,:-1], axis=2)
            temp = np.zeros((par['ylength'], par['xlength'], par['nphase']), dtype='float') # Reset temp
            # DC(:,:,i,j) = np.mean(temp,2)
            if par['ker'] > 1: # apply gaussian smoothing
                AC[:,:,i,j] = cv.GaussianBlur(AC[:,:,i,j], (par['ker'], par['ker']), par['sig'])
    return AC, path, DC


if __name__ == '__main__':
    par = readParams('../processing/parameters.ini')
    AC = rawDataLoad(par,'Select dummy folder (TEST)')
    ACph = rawDataLoad(par,'Select phantom folder (TEST)')