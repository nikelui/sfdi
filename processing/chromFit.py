# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:59:17 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import lsq_linear

from sfdi.common.getFile import getFile
from sfdi.processing import __path__ as over_path

def chromFit(op_fit_maps, par, cfile='', old=False, linear=False):
    """A function to do linear fitting of absorption to known chromophores.
    - op_fit_maps: map of (mua,mus) at all wavelenghs. Should have shape = (x,y,w,2)
            (but only mua will be fitted)
    - par: dictionary containing the parameters (need the wavelengths and used chromophores)
    - cfile: list containing a string with the path to the chromophores reference file (for batch mode)
             NOTE: if the path is in a list, it 
    - old: flag. If true, it uses old approach (average of +-5nm instead of weighted sum)
    - linear: flag, if True it uses linear algebra to fit, otherwise uses scipy lsq_linear() 
              function, with bounds on the solution (0, +inf)
    """
    chrom_map = []
    if len(par['chrom_used']) > 0: # Only process if the chromophore list is not empty
        if not cfile:
            # Select chromophores file
            cfile = getFile('Select chromophores file')
        chromophores = np.genfromtxt(cfile, delimiter='\t')
        
        # Interpolate at the used wavelengths
        ## OLD method: central wavelengths
        if old:
            f = interp1d(chromophores[:,0],chromophores[:,par['chrom_used']],kind='linear',axis=0,fill_value='extrapolate')
            waves = np.array(par['wv'])[par['wv_used']]
            E = np.array(f(waves))
        
        ## NEW method: weighted average
        else:
            data = np.genfromtxt('{}/overlaps_calibrated.csv'.format(over_path), delimiter=',') # load overlaps spectrum
            
            spec = data[tuple(x+1 for x in par['wv_used'][::-1]),:]  # Need to invert the order to go from RGB to BGR
            wv = data[0,:]  # wavelength axis
            
            f = interp1d(chromophores[:,0],chromophores[:,par['chrom_used']],kind='linear',
                         axis=0,fill_value='extrapolate')
            chrom = f(wv).T  # chromophores used, full spectrum [380-720]nm
            
            E = np.zeros((len(par['wv_used']),len(chrom))) # initialize
            ## changed: matrix data type is obsolete, try to use only arrays
            
            # TODO: Double for loop, very inefficient. Try to optimize
            for i,band in enumerate(spec):
                for j,cr in enumerate(chrom):
                    E[i,j] = np.sum(cr*band) / np.sum(band)
            #E = np.matrix(E)
            
        if (len(par['chrom_used']) == 1):
            E = E[:,np.newaxis] # This way is forced to be a column matrix
        
        if linear:
            # some linear algebra: inv_E = (E' * E)^-1 * E'
            #inv_E = np.array((E.T @ E).I @ E.T) # try with @ operator (matrix multiplication)
            inv_E = np.linalg.pinv(E) # pseudo inverse of E. It is the same as the old code 
            
            if op_fit_maps.ndim == 4: # in SFDI is a 4D array
                # Here use reshape() to multiply properly across the last two dimensions
                chrom_map = np.reshape(inv_E,(1,1,inv_E.shape[0],inv_E.shape[1])) @ op_fit_maps[:,:,par['wv_used'],0:1]
            else:
                chrom_map = inv_E @ op_fit_maps[:,0,None] # assuming they are in 2D
        ## new fitting algorithm (non-linear)
        else:
            chrom_map = np.zeros((op_fit_maps.shape[0], op_fit_maps.shape[1], len(par['chrom_used'])+1), dtype=float)
            for _i in range(op_fit_maps.shape[0]):
                for _j in range(op_fit_maps.shape[1]):
                    temp = lsq_linear(E, op_fit_maps[_i,_j,:,0], bounds=(0,np.inf), max_iter=100, method='bvls')
                    chrom_map[_i, _j, :-1] = temp.x  # solution
                    res = np.sum(temp.fun) / np.sum(op_fit_maps[_i,_j,:,0]) * 100  # residuals (in %)
                    chrom_map[_i, _j, -1] = res
                    # print('{} ({}, {}): niter={}'.format(o,_i,_j,temp.nit))  # DEBUG
                    # print('### END ###')
    return np.squeeze(chrom_map) # remove extra dimension at the end
    

if __name__ == '__main__':
#    op_fit_maps = [1]
#    par = {'wv':[458,520,536,556,626],'chrom_used':[1,2,5]}
    chrom_map = chromFit(op_fit_maps, par)
#    chrom_map = []
#    for op in op_fit_sfds:
#        chrom_map.append(chromFit(op,par)) # linear fitting for chromofores