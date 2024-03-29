# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:15:51 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

import numpy as np
from datetime import datetime
from scipy.optimize import minimize
from scipy.interpolate import interp1d

from sfdi.processing.models.reflecMCSFD import reflecMCSFD
from sfdi.processing.models.larsSFD import larsSFD
from sfdi.processing.rebin import rebin

def target_fun(opt,n,model,freqs,R_meas):
    """Function to minimize. It calculates the squared error between the model and the data.
    - model: light transport model (Monte Carlo or diffusion)
    - opt,n: optical properties [opt = tuple(mua,mus)]
    - freqs: spatial frequencies used in the computation
    - R_meas: measured Reflectance values
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

def fitOps(cal_R, par, model='mc', guess=np.array([]), homogeneous=False):
    """Optimization routine to fit for optical properties.
- cal_R: calibrated reflectance values (measured)
- par: dictionary with the parameter used
- model: light transport model. Either 'mc' or 'diff'
- guess: initial guess for optical properties, [wv x 3] array: [wv; mua; mus]
- homogeneous: if True, it will fix the fitted value of mua(f0) and only fit for
               mus(fn) at higher frequencies.
               NOTE: you need to pass your mua(f0), mus(f0) as initial guess.
"""
    if model == 'diff':
        forward_model = larsSFD # Diffusion model
    else:
        forward_model = reflecMCSFD # default to Monte Carlo
       
    # Initial guess of optical properties:
    # wavelength, mua, mus
    print('Initial guess...')
    if len(guess) == 0:
        first_run = True  # assume this is called only on the first loop (fx = f0)
        guess = np.array([[455.0, 0.0203,1.4186],
                          [525.0, 0.0203, 1.3732],
                          [650.0, 0.0203, 1.2186],
                          [670.0, 0.0172, 0.9732],
                          [690.0, 0.0179, 0.8592]])
        # interpolate at used wavelengths
        f = interp1d(guess[:,0],guess[:,1],kind='linear',bounds_error=False,fill_value='extrapolate')
        mua_guess = f(np.array(par['wv'])[par['wv_used']])
        f = interp1d(guess[:,0],guess[:,2],kind='linear',bounds_error=False,fill_value='extrapolate')
        mus_guess = f(np.array(par['wv'])[par['wv_used']])
        
        #res = [] # This list will contain the optimization results. Might be redundant
        freqs = np.array([par['freqs']])[:,np.array(par['freq_used'])] # only freq_used
        
        # initial guess for fitting: optimize the average value
        ave_R = np.mean(cal_R[:,:,par['wv_used'],:],axis=(0,1)) # average reflectance
        # DEBUG
        #    import pdb; pdb.set_trace()
        for w in range(len(mua_guess)): # loop over wavelengths
            temp = minimize(target_fun,
                           x0=(mua_guess[w],mus_guess[w]), # initial guess
                           args = (par['n_sample'],forward_model,freqs,ave_R[w,np.array(par['freq_used'])]), # function arguments
                           method = 'Nelder-Mead', # TODO: check other methods
                           options = {'maxiter':300, 'xatol':0.001, 'fatol':0.001})
            #res.append(temp) # this is not really used, but contains extra informations (eg. n. of iterations)
            mua_guess[w] = temp.x[0] # update initial guess
            mus_guess[w] = temp.x[1]
    else:
        first_run = False  # assume this is called on successive loops (fx > f0)
        f = interp1d(guess[:,0],guess[:,1],kind='linear',bounds_error=False,fill_value='extrapolate')
        mua_guess = f(np.array(par['wv'])[par['wv_used']])
        f = interp1d(guess[:,0],guess[:,2],kind='linear',bounds_error=False,fill_value='extrapolate')
        mus_guess = f(np.array(par['wv'])[par['wv_used']])
        freqs = np.array([par['freqs']])[:,np.array(par['freq_used'])] # only freq_used
        
    cal_Rbin = rebin(cal_R[:,:,par['wv_used'],:], par['binsize'])   # data binning
    print('Done!')
    ## Very heavy optimization algorithm here
    
    # Initialize optical properties map. The last dimension is 0:mua, 1:mus
    op_fit_maps = np.zeros((cal_Rbin.shape[0],cal_Rbin.shape[1],cal_Rbin.shape[2],2),dtype=float)
    
    start = datetime.now()
    for i,w in enumerate(np.array(par['wv'])[par['wv_used']]):
        print('processing wavelength: {}nm'.format(w))
        
        for j in range(cal_Rbin.shape[0]): # loop over rows
            for k in range(cal_Rbin.shape[1]): # loop over columns
                if homogeneous and not first_run:
                    temp = minimize(target_fun2,  # only fits mus
                               x0=[mus_guess[i]], # initial guess from previous step
                               args=(mua_guess[i], par['n_sample'],forward_model,freqs,cal_Rbin[j,k,i,np.array(par['freq_used'])]),
                               method = 'Nelder-Mead', # TODO: check other methods
                               options = {'maxiter':300, 'xatol':0.001, 'fatol':0.001})
                    op_fit_maps[j,k,i,0] = mua_guess[i]
                    op_fit_maps[j,k,i,1] = temp.x
                else:
                    temp = minimize(target_fun,
                               x0=[mua_guess[i],mus_guess[i]], # initial guess from previous step
                               args=(par['n_sample'],forward_model,freqs,cal_Rbin[j,k,i,np.array(par['freq_used'])]),
                               method = 'Nelder-Mead', # TODO: check other methods
                               options = {'maxiter':300, 'xatol':0.001, 'fatol':0.001})
                    op_fit_maps[j,k,i,:] = temp.x
        
        end = datetime.now()
        print('Elapsed time: {}'.format(str(end-start)))
        start = end
    return op_fit_maps#,res

if __name__ == '__main__':
    op_fit_maps = fitOps(crop(cal_R,ROI),par)