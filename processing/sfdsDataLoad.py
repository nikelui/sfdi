# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:30:59 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

import numpy as np
from scipy.io import loadmat
from sfdi.common.getFile import getFiles

def smooth(interval, window_size):
    """Perform moving average to smooth dataset"""
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def sfdsDataLoad(par,prompt='Select file',lim=[450, 750]):
    """Select a folder to load the images contained inside.
par: Dictionary containing all the processing parameters
prompt: optional string for file dialog
lim: wavelength boundaries [low, high]"""

    names = list(getFiles(prompt)) # select one or more files
    names.sort() # sort them
    AC_list = [] # empty container
    
    for fname in names:
        temp = loadmat(fname) # load the .mat file
        
        spec = temp['s'] # Spectrum
        wv = temp['w'] # wavelengths
        # intT = float(fname.split('/')[-1].split('_')[-1][:-6]) # exposure time in ms, assuming the name
                                                                 # convention is correct
        intT = temp['intTime']
        idx = np.where(np.all([wv >= lim[0],wv <= lim[1]], axis=0))[0] # Limit the spatial range
        spec = spec[idx, :]
        wv = wv[idx]
    
        # initialize 3 phase AC data structure. The "extra" dimensions are inserted for compatibility with SFDI
        AC = np.zeros((1, 1, wv.size, len(par['freqs'])), dtype='float')  #try to adopt this as standard data format
        
        for i in range(0,spec.shape[1], par['nphase']): # read data n phases at a time
            ## New AC demodulation, with vectorialization. Allows to use n-phase instead of 3
            temp = np.concatenate((spec[:, i:i+par['nphase']], spec[:, i, np.newaxis]), axis=1) # append the first element again at the end
            AC[0,0,:,i//par['nphase']] = np.sqrt(np.sum(np.diff(temp, axis=1)**2, axis=1)) / intT
        # perform smoothing with a moving average
        for i in range(AC.shape[-1]):
            AC[0,0,:,i] = smooth(AC[0,0,:,i],30)
        AC_list.append(AC)
        
    return AC_list,wv,names # should return names to identify the data

# DEBUG
if __name__ == '__main__':
    
    from matplotlib import pyplot as plt
    from sfdi.readParams3 import readParams
    
    par = readParams('../acquisition/parameters.ini')
    temp,wv = sfdsDataLoad(par,'Select SFDS data file')

    fig,ax = plt.subplots(1,1)
    for i in range(temp.shape[1]):
        ax.plot(wv, temp[:,i], label='line %d'%i)
    ax.legend()
    plt.grid(True, which='both')
    plt.show(block=False)