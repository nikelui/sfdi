# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:30:59 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

import sys
import numpy as np

sys.path.append('./common')
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work

from sfdi.getFile import getFiles
from scipy.io import loadmat

def smooth(interval, window_size):
    """Perform moving average to smooth dataset"""
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def sfdsDataLoad(par,prompt='Select file'):
    """Select a folder to load the images contained inside.
par: Dictionary containing all the processing parameters
prompt: optional string for file dialog"""

    names = list(getFiles(prompt)) # slect one or more files
    names.sort() # sort them
    AC_list = [] # empty container
    
    for fname in names:
        temp = loadmat(fname) # load the .mat file
        
        spec = temp['s'] # Spectrum
        wv = temp['w'] # wavelengths
        intT = float(fname.split('/')[-1].split('_')[-1][:-6]) # exposure time in ms, assuming the name
                                                          # convention is correct
        
        idx = np.where(np.all([wv >= 450,wv <= 750],axis=0))[0] # Limit the spatial range
        spec = spec[idx,:]
        wv = wv[idx]
    
        # initialize 3 phase AC data structure. The "extra" dimensions are inserted for compatibility with SFDI
        AC = np.zeros((1,1,wv.size,len(par['freqs'])),dtype='float')  #try to adopt this as standard data format
        
        for i in range(0,spec.shape[1],3): # read data 3 phases at a time
            AC[0,0,:,i//3] = np.sqrt(2)/3 *np.sqrt((spec[:,i]-spec[:,i+1])**2 + # demodulate
                        (spec[:,i+1]-spec[:,i+2])**2 +
                        (spec[:,i+2]-spec[:,i])**2) / intT # Normalize by exposure time
        # perform smoothing with a moving average
        for i in range(AC.shape[1]):
            AC[0,0,:,i] = smooth(AC[0,0,:,i],30)
        AC_list.append(AC)
        
    return AC_list,wv,names # should return names to identify the data


if __name__ == '__main__':
    
    from matplotlib import pyplot as plt
    from sfdi.readParams2 import readParams
    
    par = readParams('../parameters.cfg')
    temp,wv = sfdsDataLoad(par,'Select SDFD data file')

    fig,ax = plt.subplots(1,1)
    for i in range(temp.shape[1]):
        ax.plot(wv,temp[:,i],label='line %d'%i)
    ax.legend()
    plt.grid(True,which='both')
    plt.show(block=False)