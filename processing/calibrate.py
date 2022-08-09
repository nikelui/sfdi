# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:53:22 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
from scipy.interpolate import interp1d
import numpy as np
from sfdi.common.getFile import getFile
from sfdi.processing.models.reflecMCSFD import reflecMCSFD
from sfdi.processing.models.larsSFD import larsSFD
from sfdi.processing import __path__ as over_path

def calibrate(AC, ACph, par, path=[], old=False):
    """Take the AC measure of the tissue and calibrates against the AC measure on the calibration phantom.
Need a .txt file with the phantom known optical properties."""
    
    if len(path) == 0:
        path.append(getFile('Select phantom reference file'))
    test = np.genfromtxt(path[0], delimiter='\t')
    
    # Interpolate optical properties at the measured wavelengths
    ## OLD interpolation (only central wavelength)
    if old:
        f = interp1d(test[:,0],test[:,1],kind='cubic',fill_value='extrapolate') # absorption coefficient
        mua = f(par['wv'])
        f = interp1d(test[:,0],test[:,2],kind='cubic',fill_value='extrapolate') # scattering coefficient
        mus = f(par['wv'])
        f = interp1d(test[:,0],test[:,3],kind='cubic',fill_value='extrapolate') # refraction index
        n = f(par['wv'])
    
    ## NEW interpolation (with weighted average over bands)
    else:
        data = np.genfromtxt('{}/{}'.format(over_path[0], par['channels']),
                               delimiter=',') # load overlaps spectrum
        wv = data[0,:] # wavelength axis
        spec = data[1:,:] # cross-channels
        
        # Interpolate phantom to whole spectrum
        f = interp1d(test[:,0],test[:,1],kind='quadratic',fill_value='extrapolate') # absorption coefficient
        MUA = f(wv)
        f = interp1d(test[:,0],test[:,2],kind='quadratic',fill_value='extrapolate') # scattering coefficient
        MUS = f(wv)
        f = interp1d(test[:,0],test[:,3],kind='quadratic',fill_value='extrapolate') # refraction index
        N = f(wv)
        
        mua = np.zeros(len(par['wv'])) # initialize
        mus = np.zeros(len(par['wv'])) # initialize
        n = np.zeros(len(par['wv'])) # initialize
        
        for i,band in enumerate(spec):
            mua[i] = np.sum(MUA*band) / np.sum(band)
            mus[i] = np.sum(MUS*band) / np.sum(band)
            n[i] = np.sum(N*band) / np.sum(band)
        
        
    # Generate reflectance response vs. spatial frequency based on the model
    if (par['process_method'] == 'diff'): # diffusion model
        Rd_refl_model = larsSFD(mua,mus,n[0],np.array([par['freqs']]))
    elif (par['process_method'] == 'mc'): # Monte Carlo
        Rd_refl_model = reflecMCSFD(mua,mus,n[0],np.array([par['freqs']])) # to ensure that freqs is a row array
    
    # Calibrate measures
    cal_reflectance = np.zeros(AC.shape,dtype='float') # initialize
    for i in range(len(par['wv'])):
        for j in range(len(par['freqs'])):
            cal_reflectance[:,:,i,j] = AC[:,:,i,j] / ACph[:,:,i,j] * Rd_refl_model[j,i]
    
    # put non-valid numbers to zero
    cal_reflectance = np.where((cal_reflectance == np.inf),0,cal_reflectance)
    
    return cal_reflectance

if __name__ == '__main__':
#    AC,ACph = (1,2)
#    par = {'wv':[458,520,536,556,626]}
    cal_R = calibrate(AC,ACph,par)