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

from reflecMCSFD import reflecMCSFD
from larsSFD import larsSFD
from sfdi.rebin import rebin

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
    

def fitOps_sfds(cal_R,par,model='mc'):
    """Optimization routine to fit for optical properties.
- cal_R: calibrated reflectance values (measured)
- par: dictionary with the parameter used
- model: light transport model. Either 'mc' or 'diff'
"""

    if model == 'diff':
        forward_model = larsSFD # Diffusion model
    else:
        forward_model = reflecMCSFD # default to Monte Carlo
       
    # Initial guess of optical properties:
    # wavelength, mua, mus
#    guess = np.array( [[400.0000,0.0223,2.8186]]
#                       [450.0000,0.0203,2.8186],
#                       [670.0000,0.0172,0.9732],
#                       [690.0000,0.0179,0.8592],
#                       [710.0000,0.0177,0.8045],
#                       [730.0000,0.0178,0.7742],
#                       [750.0000,0.0177,0.7519],
#                       [770.0000,0.0177,0.7185],
#                       [790.0000,0.0171,0.6992],
#                       [790.0000,0.0171,0.6992],
#                       [810.0000,0.0173,0.6848],
#                       [830.0000,0.0171,0.6942],
#                       [850.0000,0.0166,0.6750],
#                       [870.0000,0.0188,0.6375],
#                       [890.0000,0.0211,0.6538],
#                       [910.0000,0.0249,0.6651],
#                       [930.0000,0.0187,0.7915],
#                       [950.0000,0.0189,0.7068],
#                       [970.0000,0.0192,0.7235]])
#    # interpolate at used wavelengths
#    f = interp1d(guess[:,0],guess[:,1],kind='linear',bounds_error=False,fill_value='extrapolate')
#    mua_guess = f(np.array(par['wv']))
#    f = interp1d(guess[:,0],guess[:,2],kind='linear',bounds_error=False,fill_value='extrapolate')
#    mus_guess = f(np.array(par['wv']))
    
    ## Initial guess at lowest frequency
    guess = np.array([0.0223, 2.818]) # [mua, mus]
    
    
    #res = [] # This list will contain the optimization results. Might be redundant
    freqs = np.array([par['freqs']])[:,np.array(par['freq_used'])] # only freq_used
    
    # initial guess for fitting: optimize the average value
    #ave_R = np.mean(cal_R,axis=(0,1)) # average reflectance
    #print('Initial guess...')
#    for w in range(len(mua_guess)): # loop over wavelengths
#        temp = minimize(target_fun,
#                       x0=(mua_guess[w],mus_guess[w]), # initial guess
#                       args = (par['n_sample'],forward_model,freqs,ave_R[w,:]), # function arguments
#                       method = 'Nelder-Mead', # TODO: check other methods
#                       options = {'maxiter':200, 'xatol':0.001, 'fatol':0.001})
#        #res.append(temp) # this is not really used, but contains extra informations (eg. n. of iterations)
#        mua_guess[w] = temp.x[0] # update initial guess
#        mus_guess[w] = temp.x[1]
#    
#    cal_Rbin = rebin(cal_R,par['binsize'])   # data binning
#    print('Done!')
    ## Very heavy optimization algorithm here
    
    # Initialize optical properties map. The last dimension is 0:mua, 1:mus
    op_fit_maps = np.zeros((len(par['wv']),2),dtype=float)
    #res = [] # store results
    
    start = time.time()
    for i,w in enumerate(par['wv']):
        #print('processing wavelength: %dnm' % w)
        temp = minimize(target_fun,
                        x0=[guess[0],guess[1]], # initial guess from previous step
                        args=(par['n_sample'],forward_model,freqs,cal_R[i]),
                        method = 'Nelder-Mead', # TODO: check other methods
                        options = {'maxiter':200, 'xatol':0.001, 'fatol':0.001})
                
        op_fit_maps[i,:] = temp.x
        #guess = temp.x # update guess for next wavelength
        #res.append(temp)
        
        #end = time.time()
        #print('Elapsed time: %.1fs' % (end-start))
        #start = end
    
    end = time.time() # total time
    print('Elapsed time: %.1fs' % (end-start))
    
    return op_fit_maps#,res

if __name__ == '__main__':
    op_fit_maps = fitOps(crop(cal_R,ROI),par)