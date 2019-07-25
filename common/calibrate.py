# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:53:22 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import sys
sys.path.append('./common')
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work

from sfdi.getFile import getFile
from mycsv import csvread
from scipy.interpolate import interp1d
import numpy as np
from reflecMCSFD import reflecMCSFD
from larsSFD import larsSFD

def calibrate(AC,ACph,par,path=[]):
    """Take the AC measure of the tissue and calibrates against the AC measure on the calibration phantom.
Need a .txt file with the phantom known optical properties."""
    
    if len(path) == 0:
        path.append(getFile('Select phantom reference file'))
    test,_ = csvread(path[0],arr=True,delimiter='\t')
    
    # Interpolate optical properties at the measured wavelengths
    f = interp1d(test[:,0],test[:,1],kind='cubic',fill_value='extrapolate') # absorption coefficient
    mua = f(par['wv'])
    f = interp1d(test[:,0],test[:,2],kind='cubic',fill_value='extrapolate') # scattering coefficient
    mus = f(par['wv'])
    f = interp1d(test[:,0],test[:,3],kind='cubic',fill_value='extrapolate') # refraction index
    n = f(par['wv'])
    
    # Generate reflectance response vs. spatial frequency based on the model
    if (par['process_method'] == 'diff'): # diffusion model
        Rd_refl_model = larsSFD(mua,mus,n[0],np.array([par['freqs']]))
        pass # for now skip this part
    elif (par['process_method'] == 'mc'): # Monte Carlo
        ## TODO: finish to write reflecMCSFD function
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
    cal_R = calibrate(AC,ACph,par)