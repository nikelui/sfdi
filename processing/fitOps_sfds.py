# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:15:51 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import interp1d

from sfdi.processing.models.reflecMCSFD import reflecMCSFD
from sfdi.processing.models.larsSFD import larsSFD

def target_fun(opt,n,model,freqs,R_meas):
    """Function to minimize. It calculates the squared error between the model and the data.
    - model: light transport model (Monte Carlo or diffusion)
    - opt,n: optical properties [opt = tuple(mua,mus)]
    - freqs: spatial frequencies used in the computation
    - values: measured Reflectance values
"""
    mua,mus = opt # unpack values
    R_model = np.squeeze(model([mua],[mus],n,freqs)) # calculate R from light transport model
    return np.sum((R_model - R_meas)**2) # return sum of square diff.
    

def fitOps_sfds(cal_R, par, model='mc'):
    """Optimization routine to fit for optical properties.
- cal_R: calibrated reflectance values (measured)
- par: dictionary with the parameter used
- model: light transport model. Either 'mc' or 'diff'
"""

    if model == 'diff':
        forward_model = larsSFD  # Diffusion model
    else:
        forward_model = reflecMCSFD  # default to Monte Carlo
           
    ## Initial guess at lowest wavelength
    guess = np.array([0.0223, 2.818]) # [mua, mus]
    
    freqs = np.array([par['freqs']])[:,np.array(par['freq_used'])] # only freq_used
    
    ## Very heavy optimization algorithm here
    # Initialize optical properties map. The last dimension is 0:mua, 1:mus
    op_fit_maps = np.zeros((len(par['wv']), 2), dtype=float)
    
    start = time.time()
    for i, w in enumerate(par['wv']):
        #print('processing wavelength: %dnm' % w)
        temp = minimize(target_fun,
                        x0=[guess[0], guess[1]], # initial guess from previous step
                        args=(par['n_sample'], forward_model, freqs, cal_R[i]),
                        method = 'Nelder-Mead', # TODO: check other methods
                        options = {'maxiter':200, 'xatol':0.001, 'fatol':0.001})
                
        op_fit_maps[i,:] = temp.x
    end = time.time() # total time
    print('Elapsed time: %.1fs' % (end-start))
    
    return op_fit_maps

if __name__ == '__main__':
    op_fit_maps = fitOps(crop(cal_R,ROI),par)