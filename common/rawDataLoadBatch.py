# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 08:28:24 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import os,sys,re
import numpy as np
import cv2 as cv

sys.path.append('./common')
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work

from sfdi.readParams2 import readParams
from sfdi.getPath import getPath

def rawDataLoadBatch(par, prompt='Select base folder'):
    """Select a folder to load the images contained inside.
par: Dictionary containing all the processing parameters
prompt: optional string for file dialog
"""
    ## New approach: select base path, then input the name(s) that you want to process
    ## in the terminal (process with regex?)
    ## TODO: when reworking the interface in Qt, this will be trivial
    path = getPath(prompt)
    if len(path) > 0: # check for empty path
        dirs = [x for x in os.listdir(path) if os.path.isdir(path+'/'+x)] # list of directories
        #toProcess = []
        # you can also define the names here manually
        #toProcess = ['_testhannabase2_','_testhannaocclusion_','_testhannarelease_']
        toProcess = ['1601804851', '1601805014', '1601805130', '1601805243']
        
        if(len(toProcess)==0): # In case you define by hand
            regex = input('Input base name to match: ').lower() # only process matching directories
            while (regex != ''): # End with an empty name
                toProcess.append(regex)
                regex = input('Input base name to match: ').lower() # only process matching directories
        
        # Some regex magic
        pattern = "|".join(re.escape(s) for s in toProcess)
        rexp = re.compile(pattern)
        dirs = [x for x in dirs if rexp.search(x.lower())] # filter only matching names
        
        # Return vales are lists containing, data, file name and timestamp
        ACv = []
        names = []
        tstamps = []        
        
        for name in dirs:
            intT = float(name.split('_')[-1][:-2]) # exposure time
            
            files = [x for x in os.listdir(path+'/'+name) if '.bmp' in x]
            files.sort() # This assumes the correct naming convention is used
            
            # initialize 3 phase AC data structure
            temp = np.zeros((par['ylength'],par['xlength'],par['nphase']),dtype='float') # To read each 3-phase image
            AC = np.zeros((int(par['ylength']),int(par['xlength']),len(par['wv']),
                           len(par['freqs'])),dtype='float')  #try to adopt this as standard data format
            
            for i in range(len(par['wv'])):
                print('loading all frequencies for wavelength: {} nm'.format(par['wv'][i]))
                for j in range(len(par['freqs'])):
                    for p in range(par['nphase']):
                        fname = files[p + j*par['nphase'] + i*len(par['freqs']) * par['nphase']]
                        #print(fname) # debug
                        #continue # debug
                        temp[:,:,p] = cv.imread('{}/{}/{}'.format(path, name, fname),cv.IMREAD_GRAYSCALE); # all three channels should be equal, anyways
                    
                    #AC[:,:,i,j] = np.sqrt(2)/3 * np.sqrt( (temp[:,:,0]-temp[:,:,1])**2 +
                    #        (temp[:,:,1]-temp[:,:,2])**2 +
                    #        (temp[:,:,2]-temp[:,:,0])**2 ) / intT # normalize by exposure time
                    
                    ## New AC demodulation
                    temp = np.dstack((temp,temp[:,:,0])) # append the first element again at the end
                    AC[:,:,i,j] = 1 * np.sqrt(np.sum(np.diff(temp,axis=-1)**2,axis=-1)) / intT
                    ##TODO: XX is the correct normalization term (depends on nPhase?)
                    #DC(:,:,i,j) = np.mean(temp,3);
                    temp = np.zeros((par['ylength'], par['xlength'], par['nphase']), dtype='float') # Reset temp
                    
                    if par['ker'] > 1: # apply gaussian smoothing
                        AC[:,:,i,j] = cv.GaussianBlur(AC[:,:,i,j], (par['ker'], par['ker']), par['sig']) # Gaussian smoothing, radius 3px, sigma=1.5
            
            ACv.append(AC)
            names.append(name)
            tstamps.append(int(name.split('_')[0]))
    else:
        sys.exit()
            
    return ACv,names,tstamps


if __name__ == '__main__':
    par = readParams('../processing/parameters.cfg')
    AC,names,tstamps = rawDataLoadBatch(par,'Select dummy folder (TEST)')
    #ACph = rawDataLoadBatch(par,'Select phantom folder (TEST)')