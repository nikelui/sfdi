# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:59:17 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

from sfdi.getFile import getFile
from mycsv import csvread

import numpy as np
from scipy.interpolate import interp1d

def chromFit(op_fit_maps,par,cfile=[]):
    """A function to do linear fitting of absorption to known chromophores.
    - op_fit_maps: map of (mua,mus) at all wavelenghs. Should have shape = (x,y,w,2)
            (but only mua will be fitted)
    - par: dictionary containing the parameters (need the wavelengths and used chromophores)
    """
    if len(par['chrom_used']) > 0: # Only process if the chromophore list is not empty
        
        if len(cfile) == 0:
            # Select chromophores file
            cfile.append(getFile('Select chromophores file'))
        chromophores,_ = csvread(cfile[0],arr=True,delimiter='\t') # remember that cfile is a list
        
        # Interpolate at the used wavelengths
        f = interp1d(chromophores[:,0],chromophores[:,par['chrom_used']],kind='cubic',axis=0)
        E = np.matrix(f(par['wv']))
        
        # some linear algebra: inv_E = (E' * E)^-1 * E'
        #inv_E = np.matmul(np.matmul(E.T,E).I,E.T)
        inv_E = np.array((E.T @ E).I @ E.T) # try with @ operator (matrix multiplication)
        
        # Here use reshape() to multiply properly across the last two dimensions
        chrom_map = np.reshape(inv_E,(1,1,inv_E.shape[0],inv_E.shape[1])) @ op_fit_maps[:,:,:,1:]
        
        return np.squeeze(chrom_map) # remove extra dimension at the end
    
if __name__ == '__main__':
    #chrom_map = chromFit(op_fit_maps,par)
    chrom_map = []
    for op in op_fit_maps:
        chrom_map.append(chromFit(op_fit_maps[0],par)) # linear fitting for chromofores