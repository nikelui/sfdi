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
    
def target_fun2(mus,mua,n,model,freqs,R_meas):
    """Function to minimize. It calculates the squared error between the model and the data.
    - model: light transport model (Monte Carlo or diffusion)
    - mus,mua,n: optical properties
    - freqs: spatial frequencies used in the computation
    - R_meas: measured Reflectance values
"""
    R_model = np.squeeze(model([mua],[mus],n,freqs)) # calculate R from light transport model
    return np.sum((R_model - R_meas)**2) # return sum of square diff.

def fitOps_sfds(cal_R, par, model='mc', guess=np.array([]), homogeneous=False):
    """Optimization routine to fit for optical properties.
- cal_R: calibrated reflectance values (measured)
- par: dictionary with the parameter used
- model: light transport model. Either 'mc' or 'diff'
"""

    if model == 'diff':
        forward_model = larsSFD  # Diffusion model
    else:
        forward_model = reflecMCSFD  # default to Monte Carlo
           
    ## Initial guess, if not passed
    if len(guess) == 0:
        first_run = True  # assume this is called only on the first loop (fx = f0)
        guess = np.tile(np.array([0.01, 1]), (len(par['wv']), 1))  # [mua, mus]
    else:
        first_run = False
    
    freqs = np.array([par['freqs']])[:,np.array(par['freq_used'])] # only freq_used
    op_fit_maps = np.zeros((len(par['wv']), 2), dtype=float)
    start = time.time()
    
    if homogeneous and not first_run:
        for w in range(len(par['wv'])):
            temp = minimize(target_fun2,  # only fit mus
                            x0=guess[w, 1], # initial guess from previous step
                            args=(guess[w,0], par['n_sample'], forward_model, freqs, cal_R[w]),
                            method = 'Nelder-Mead', # TODO: check other methods
                            options = {'maxiter':300, 'xatol':0.001, 'fatol':0.001})
            op_fit_maps[w,0] = guess[w,0]
            op_fit_maps[w,1] = temp.x
    else:
        for w in range(len(par['wv'])):
            temp = minimize(target_fun,
                            x0=guess[w, :], # initial guess from previous step
                            args=(par['n_sample'], forward_model, freqs, cal_R[w]),
                            method = 'Nelder-Mead', # TODO: check other methods
                            options = {'maxiter':300, 'xatol':0.001, 'fatol':0.001})
            op_fit_maps[w,:] = temp.x
        
    end = time.time() # total time
    print('Elapsed time: %.1fs' % (end-start))
    
    return op_fit_maps

if __name__ == '__main__':
    op_fit_maps = fitOps(crop(cal_R,ROI),par)