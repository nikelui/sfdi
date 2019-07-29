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

from sfdi.readParams2 import readParams
from sfdi.getPath import getPath

def rawDataLoad(par,prompt='Select folder'):
    """Select a folder to load the images contained inside.
par: Dictionary containing all the processing parameters
prompt: optional string for file dialog"""
    path = getPath(prompt)
    
    if len(path) > 0: # check for empty path
        intT = float(path.split('/')[-1].split('_')[-1][:-2]) # exposure time
    
    files = [x for x in os.listdir(path) if '.bmp' in x]
    files.sort() # This assumes the correct naming convention is used
    
    # initialize 3 phase AC data structure
    temp = np.zeros((par['ylength'],par['xlength'],3),dtype='float') # To read each 3-phase image
    AC = np.zeros((int(par['ylength']),int(par['xlength']),len(par['wv']),
                   len(par['freqs'])),dtype='float')  #try to adopt this as standard data format
    
    for i in range(len(par['wv'])):
        print('loading all frequencies for wavelength: %d nm' % par['wv'][i])
        for j in range(len(par['freqs'])):
            for p in range(3):
                fname = files[p + j*3 + i*len(par['freqs'])*3]
                #print(fname) # debug
                #continue # debug
                temp[:,:,p] = cv.imread(path+'/'+fname,cv.IMREAD_GRAYSCALE); # all three channels should be equal, anyways
            
            AC[:,:,i,j] = np.sqrt(2)/3 * np.sqrt( (temp[:,:,0]-temp[:,:,1])**2 +
                    (temp[:,:,1]-temp[:,:,2])**2 +
                    (temp[:,:,2]-temp[:,:,0])**2 ) / intT # normalize by exposure time
            #DC(:,:,i,j) = np.mean(temp,3);
            if par['ker'] > 1: # apply gaussian smoothing
                AC[:,:,i,j] = cv.GaussianBlur(AC[:,:,i,j],(9,9),1.5) # Gaussian smoothing, radius 3px, sigma=1.5
    return AC,path


if __name__ == '__main__':
    par = readParams('../parameters.cfg')
    AC = rawDataLoad(par,'Select dummy folder (TEST)')
    ACph = rawDataLoad(par,'Select phantom folder (TEST)')