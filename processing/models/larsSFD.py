# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:42:44 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Diffuse reflectance model using diffuse aproximation

"""
import numpy as np

## TODO: fix doc

def larsSFD(mua,mus,n,freqs):
    # Just to be sure, need to do array operations
    mua = np.array(mua).reshape(1,-1)
    mus = np.array(mus).reshape(1,-1)
    freqs = np.array(freqs).reshape(1,-1)
    
    mut = mua+mus # transport coefficient
    
    Reff=-1.440/(n**2) + 0.710/n + 0.668 + 0.0636*n # Effective reflection coefficient. Assumes air-tissue interface
    A=(1-Reff)/2/(1+Reff) # A proportionality constant
    
    ##for w in range(len(mua)):
    # Try without looping
    mueff = np.sqrt(3*mua*mut * np.ones((freqs.size,1)) + (2*np.pi*freqs.T * np.ones((1,mua.size)))**2)
    R_at_fx = 3*A * (mus/mut) / (mueff/mut + 1) / (mueff/mut + 3*A)
    
    return R_at_fx      
    

def larsSFD2(mua,mus,n,freqs):
    # Just to be sure, need to do array operations
    mua = np.array(mua).reshape(-1)
    mus = np.array(mus).reshape(-1)
    freqs = np.array(freqs).reshape(-1)
    
    mut = mua+mus # transport coefficient
    
    Reff=-1.440/(n**2) + 0.710/n + 0.668 + 0.0636*n # Effective reflection coefficient. Assumes air-tissue interface
    A=(1-Reff)/2/(1+Reff) # A proportionality constant
    ## Looping version
    R_at_fx = np.zeros((freqs.size,mua.size),dtype='float') # initialize Reflectance array
    
    for i in range(mua.size):
        for j in range(freqs.size):
            mueff = np.sqrt(3*mua[i]*mut[i] + (2*np.pi*freqs[j])**2)
            R_at_fx[j,i] = 3*A * (mus[i]/mut[i])/(mueff/mut[i] + 1)/(mueff/mut[i] +3*A)
    
    return R_at_fx


if __name__ == '__main__':
    R1 = larsSFD([1,1,1],[1,1,1],1.3,np.array([[0,0.1,0.2,0.3]])) # timing: ~21us
    R2 = larsSFD2([1,1,1],[1,1,1],1.3,np.array([[0,0.1,0.2,0.3]])) # timing: ~65us