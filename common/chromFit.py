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

def chromFit(op_fit_maps,par,cfile=[],old=False):
    """A function to do linear fitting of absorption to known chromophores.
    - op_fit_maps: map of (mua,mus) at all wavelenghs. Should have shape = (x,y,w,2)
            (but only mua will be fitted)
    - par: dictionary containing the parameters (need the wavelengths and used chromophores)
    """
    chrom_map = []
    if len(par['chrom_used']) > 0: # Only process if the chromophore list is not empty
        
        if len(cfile) == 0:
            # Select chromophores file
            cfile.append(getFile('Select chromophores file'))
        chromophores,_ = csvread(cfile[0],arr=True,delimiter='\t') # remember that cfile is a list
        
        # Interpolate at the used wavelengths
        ## OLD method: central wavelengths
        if old:
            f = interp1d(chromophores[:,0],chromophores[:,par['chrom_used']],kind='linear',axis=0,fill_value='extrapolate')
            E = np.matrix(f(par['wv']))
        
        ## NEW method: weighted average
        else:
            data,_ = csvread('common/overlaps_calibrated.csv',arr=True) # load overlaps spectrum
            wv = data[0,:] # wavelength axis
            spec = data[(9,6,5,4,1),:] # 5 channels [380-720]nm
            f = interp1d(chromophores[:,0],chromophores[:,par['chrom_used']],kind='linear',axis=0,fill_value='extrapolate')
            chrom = f(wv).T # chromophores used [380-720]nm
            
            E = np.zeros((len(spec),len(chrom))) # initialize
            
            # TODO: Double for loop, very inefficient. Try to optimize
            for i,band in enumerate(spec):
                for j,cr in enumerate(chrom):
                    E[i,j] = np.sum(cr*band) / np.sum(band)
            
        if (len(par['chrom_used']) == 1):
            E = E.T # This way is forced to be a column matrix
        
        # some linear algebra: inv_E = (E' * E)^-1 * E'
        #inv_E = np.matmul(np.matmul(E.T,E).I,E.T)
        inv_E = np.array((E.T @ E).I @ E.T) # try with @ operator (matrix multiplication)
        
        if op_fit_maps.ndim == 4: # in SFDI is a 4D array
            # Here use reshape() to multiply properly across the last two dimensions
            chrom_map = np.reshape(inv_E,(1,1,inv_E.shape[0],inv_E.shape[1])) @ op_fit_maps[:,:,:,1:]
        else:
            chrom_map = inv_E @ op_fit_maps[:,0,None] # assuming they are in 2D
        
    return np.squeeze(chrom_map) # remove extra dimension at the end
    
if __name__ == '__main__':
#    op_fit_maps = [1]
#    par = {'wv':[458,520,536,556,626],'chrom_used':[1,2,5]}
    chrom_map = chromFit(op_fit_maps,par)
#    chrom_map = []
#    for op in op_fit_sfds:
#        chrom_map.append(chromFit(op,par)) # linear fitting for chromofores