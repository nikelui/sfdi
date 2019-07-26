# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:41:14 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import numpy as np
import os
from scipy.io import loadmat
from scipy.special import j0 # Bessel function of the first kind, order zero
from scipy.constants import c # speed of light in vacuum

def fresnel(n_in,n_out,theta):
    """Function to calculate reflection and transmission coefficients (Fresnel equation)."""
    theta_p = (n_in/n_out) * np.sin(theta)
    
    R = 0.5*((n_in*np.cos(theta_p) - n_out*np.cos(theta)) /
             (n_in*np.cos(theta_p) + n_out*np.cos(theta)) )**2 +\
        0.5*((n_in*np.cos(theta) - n_out*np.cos(theta_p)) /
             (n_in*np.cos(theta) + n_out*np.cos(theta_p)) )**2
    return R


def reflecMCSFD(mua,mus,n,freqs,MC={}):
    """Function to get frequency response of reflectance."""
    ## TODO: fix doc
    if (n < 1.4):
        mc_sim_type = 'motherMC_n1p33_g0p71'
    else:
        mc_sim_type = 'motherMC_n1p4_g0p9'
    
    # just in case
    freqs = np.array(freqs).reshape(1,-1)
#    load white MC simulation -> motherR(rho_MC,t_MC)
#    rho_MC - source-detector separation
#    rho_interval - thickness of detection ring for corresponding rho_MC
#    t_MC - time of photon detection
#    t_interval - detection time interval for corresponding t_MC
#    mother R - reflectance array (rho_MC down vs t_MC across)
    if len(MC) == 0: # Trying to implement persistence
        path = os.path.abspath('../common/models/')
        #TODO: this will only work if the working directory is the "start" one
        temp = loadmat(os.path.join(path,mc_sim_type)) # if the Monte Carlo has not been loaded
        MC.update(temp) # Append values to the dictionary containing MC variables
    
    v = c/n/1e6 # speed of light in medium [mm/ns]
    R_at_fx = np.zeros((len(freqs[0]),len(mua)),dtype='float') # Refl. as function of fx
    
    for i in range(len(mua)): # loop over each pair of optical properties        
        ## Apply absorption and pathlength correction to white Monte Carlo model 
        ## NOTE: there is no need to 'extend' the arrays, numpy can do Broadcasting
        
        ## Apply scaling factor
        ## The white Monte Carlo model is simulated with mua=0 and mus=1, so a scaling
        ## factor of 1/mus_tissue is applied to the time and space 'axis' of the motherR matrix
        t_MC = MC['t_MC'] / mus[i]
        t_interval = MC['t_interval'] / mus[i]
        rho_MC = MC['rho_MC'] / mus[i]
        rho_interval = MC['rho_interval'] / mus[i]
        
        R_at_rhoMC = np.sum( # Reflectance as function of distance [per m^2 per s]
                MC['motherR'] / (1-fresnel(1,n,0)) * # Surface reflection
                (np.exp(-mua[i]*v*t_MC) * t_interval),axis=1,keepdims=True) * mus[i]**3 # Absorption
## Original Matlab code
#    % Weight for arbitrary mua by applying a pathlength correction => sum[ R(rho,t)*dt ] 
#    R_at_rhoMC = sum( motherR/(1-fresnel(1,n,0)).*
#                       repmat(exp(-mua*v*t_MC/musp).*t_interval/musp,length(rho_MC),1), 2)*musp^3;
        
        ## 2D Spatial Fourier transform. Since the Monte Carlo is computed in  polar coordinates,
        ## the HANKEL TRANSFORM is used to transform to the spatial frequency domain
        R_at_fx[:,i] = np.sum( R_at_rhoMC *
               ## rho_MC is a column vector. To broadcast, freqs must be a row vector
               j0(2*np.pi*freqs * rho_MC) * # Bessel function
               2*np.pi*rho_MC * rho_interval,axis=0)
## Original Matlab code
#    % Apply Spatial Fourier Transform => sum[ R(rho)*e^(i*2*pi*f)*2*pi*rho*drho ]
#    R_at_fx(:,j) = sum( repmat(R_at_rhoMC.',[length(f),1]) .* ...
#    besselj( 0, repmat(2*pi*f,[1,length(rho_MC)]).*repmat((rho_MC.')./musp,[length(f),1]) ) .* ...
#    repmat(2*pi*(rho_MC.')./musp,[length(f),1]) .* ...
#    repmat((rho_interval.')/musp,[length(f),1]), 2);
    
    return R_at_fx

def reflecMCSFD2(mua,mus,n,freqs,MC={}):
    """Function to get frequency response of reflectance."""
    ## TODO: fix doc
    if (n < 1.4):
        mc_sim_type = 'motherMC_n1p33_g0p71'
    else:
        mc_sim_type = 'motherMC_n1p4_g0p9'
    
    # Just to be sure, need to do array operations
    mua = np.array(mua).reshape(1,-1)
    mus = np.array(mus).reshape(1,-1)
    freqs = np.array(freqs).reshape(1,-1)
    
#    load white MC simulation -> motherR(rho_MC,t_MC)
#    rho_MC - source-detector separation
#    rho_interval - thickness of detection ring for corresponding rho_MC
#    t_MC - time of photon detection
#    t_interval - detection time interval for corresponding t_MC
#    mother R - reflectance array (rho_MC down vs t_MC across)
    if len(MC) == 0: # Trying to implement persistence
        path = os.path.abspath('common/models/')
        #TODO: this will only work if the working directory is the "start" one
        temp = loadmat(os.path.join(path,mc_sim_type)) # if the Monte Carlo has not been loaded
        MC.update(temp) # Append values to the dictionary containing MC variables
    
    v = c/n/1e6 # speed of light in medium [mm/ns]
    #R_at_fx = np.zeros((freqs.size,mua.size),dtype='float') # Refl. as function of fx
    
    ## Apply scaling factor
    ## The white Monte Carlo model is simulated with mua=0 and mus=1, so a scaling
    ## factor of 1/mus_tissue is applied to the time and space 'axis' of the motherR matrix
    
    ## Note: here instead of looping, we vectorialize and go to a higher dimension
    t_MC = MC['t_MC'] / mus.T
    t_interval = MC['t_interval'] / mus.T
    rho_MC = MC['rho_MC'] / mus
    rho_interval = MC['rho_interval'] / mus
    
    #### Continue here
    R_at_rhoMC = np.sum( # Reflectance as function of distance [per m^2 per s]
                np.expand_dims(MC['motherR'],axis=2) / (1-fresnel(1,n,0)) * # Surface reflection
                (np.exp(-mua.T*v*t_MC).T * t_interval.T),axis=1,keepdims=True) * mus**3 # Absorption
    
## Original Matlab code
#    % Weight for arbitrary mua by applying a pathlength correction => sum[ R(rho,t)*dt ] 
#    R_at_rhoMC = sum( motherR/(1-fresnel(1,n,0)).*
#                       repmat(exp(-mua*v*t_MC/musp).*t_interval/musp,length(rho_MC),1), 2)*musp^3;
        
        ## 2D Spatial Fourier transform. Since the Monte Carlo is computed in  polar coordinates,
        ## the HANKEL TRANSFORM is used to transform to the spatial frequency domain
    R_at_fx = np.sum( R_at_rhoMC *
           ## rho_MC is a column vector. To broadcast, freqs must be a row vector
           j0(2*np.pi*freqs.T * np.expand_dims(rho_MC,axis=1)) * # Bessel function
           2*np.pi*np.expand_dims(rho_MC,axis=1) * np.expand_dims(rho_interval,axis=1),axis=0)
## Original Matlab code
#    % Apply Spatial Fourier Transform => sum[ R(rho)*e^(i*2*pi*f)*2*pi*rho*drho ]
#    R_at_fx(:,j) = sum( repmat(R_at_rhoMC.',[length(f),1]) .* ...
#    besselj( 0, repmat(2*pi*f,[1,length(rho_MC)]).*repmat((rho_MC.')./musp,[length(f),1]) ) .* ...
#    repmat(2*pi*(rho_MC.')./musp,[length(f),1]) .* ...
#    repmat((rho_interval.')/musp,[length(f),1]), 2);
    
    return R_at_fx


if __name__ == '__main__':
    R = reflecMCSFD([1,1,1],[1,1,1],1.3,np.array([[0,0.1,0.2,0.3]]))
    R2 = reflecMCSFD2([1,1,1],[1,1,1],1.3,np.array([[0,0.1,0.2,0.3]]))
    
    
    