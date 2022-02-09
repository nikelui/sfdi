# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 14:05:28 2022

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
from itertools import product
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from sfdi.analysis.depthMC import depthMC

def phi(mua, mus, fx, z):
    """Function to calculate fluence of light in depth
    - mua, mus: vectors of optical properties (N x M)
    - fx: average fx in range (N x 1)
    - z: depth (1 x Z)
        N: number of frequencies
        M: number of wavelengths
        Z: number of depths
        
    RETURN
    - phi: array of light fluences (N x M x Z)
"""
    mut = mua + mus
    mueff = np.sqrt(np.abs(3 * mua * mut))
    mueff1 = np.sqrt(mueff**2 + (2*np.pi*fx[:,np.newaxis])**2)
    a1 = mus / mut  # albedo
    # effective reflection coefficient. Assume n = 1.4
    Reff = 0.0636*1.4 + 0.668 + 0.71/1.4 - 1.44/1.4**2
    A = (1 - Reff)/(2*(1 + Reff))  # coefficient
    C = (-3*a1*(1 + 3*A))/((mueff1**2/mut**2 - 1) * (mueff1/mut + 3*A))
    fluence = (3*a1 / (mueff1**2 / mut**2 - 1))[:,:,np.newaxis] * \
        np.exp(-mut[:,:,np.newaxis] * z[np.newaxis,:]) +\
        C[:,:,np.newaxis] * np.exp(-mueff1[:,:,np.newaxis] * z[np.newaxis,:])    
    return fluence

def depth_d(mua, mus, fx):
    """Function to calculate effective penetration depth based on diffusion approximation
    - mua, mus: vectors of optical properties (N x M)
    - fx: average fx in range (N x 1)
        N: number of frequencies
        M: number of wavelengths
    
    RETURN
    - d: penetration depth array (N x M)
"""
    mut = mua + mus
    mueff = np.sqrt(np.abs(3 * mua * mut))
    mueff1 = np.sqrt(mueff**2 + (2*np.pi*fx[:,np.newaxis])**2)
    d = 1/mueff1
    return d

def depth_phi(mua, mus, fx, zmax=20, zstep=0.01):
    """Function to calculate effective penetration depth based on fluence rate (1/e)
    - mua, mus: vectors of optical properties (N x M)
    - fx: average fx in range (N x 1)
        N: number of frequencies
        M: number of wavelengths

    RETURN
    - d: penetration depth array (N x M)
"""
    z = np.arange(0, zmax, zstep)  # default: calculate fluence over 20mm with 0.01mm resolution
    fluence = phi(mua, mus, fx, z) 
    d = np.zeros(mua.shape, dtype=float)
    for _f, _w in product(range(mua.shape[0]),range(mua.shape[1])):
        line = fluence[_f, _w,:]
        idx = np.argwhere(line**2 <= np.max(line**2)/np.e)[0][0]
        d[_f, _w] = z[idx]  # where fluence < (1/e * max(fluence))
    return d

def depth_mc(mua, mus, fx, frac=4):
    """Function to calculate effective penetration depth based on fluence rate (1/e)
    - mua, mus: vectors of optical properties (N x M)
    - fx: average fx in range (N x 1)
        N: number of frequencies
        M: number of wavelengths
    - frac: fraction of photons that reaches depth d. Pass the appropriate index
        frac -> [10 25 50 75 90]%

    RETURN
    - d: penetration depth array (N x M)
"""
    d = np.zeros(mua.shape, dtype=float)  
    for _f, f in enumerate(fx):
        temp = depthMC(mua[_f,:], mus[_f,:], f)
        d[_f,:] = temp[frac,:,:]
    return d

def two_layer_model(mus1, mus2, depths, thick):
    """Linear combination of mus on 2-layer model
    - mus1: scattering vector, thin layer (N x M)
    - mus2: scattering vector, thick layer (N x M)
    - depths: penetration depths, top layer (N x M)
    - thick: thicness of top layer (D x 1)
        N: number of frequencies
        M: number of wavelengths
        D: number of thin phantoms
    
    """
    mus_model = []
    for th in thick:
        ret = (mus1 * th + mus2 * (depths - th)) / depths
        idx = np.where(th/depths >= 1)
        if len(idx[0]) > 0:
            ret[idx] = mus1[idx]
        mus_model.append(ret)
    return mus_model

## Parameters
wv = np.array([458,520,536,556,626])  # wavelengths
F = np.arange(0,0.51,0.05)
fx = np.array([np.mean(F[a:a+3]) for a in range(len(F)-3)])  # average fx
thick = np.array([0.125, 0.265, 0.51, 0.67, 1.17])  # layer thickness

## optical properties are stacked over fx (assume homogeneous)
# Titanium oxide
mua_ti = np.tile(np.array([0.0315582, 0.0368862, 0.0368169, 0.0367965, 0.0428225]), (8,1))
a1,b1 = np.array([7.61528368e+03, 1.25263885e+00])
mus_ti = np.tile((a1 * np.power(wv, -b1)), (8,1))
# Aluminum oxide
mua_al = np.tile(np.array([0.02110538, 0.0223065 , 0.02223373, 0.02278003, 0.02402116]), (8,1))
a2,b2 = np.array([9.88306737, 0.3909216 ])
mus_al = np.tile((a2 * np.power(wv, -b2)), (8,1))

# penetration depths
dd_ti = depth_d(mua_ti, mus_ti, fx)
dp_ti = depth_phi(mua_ti, mus_ti, fx)
dmc_ti = depth_mc(mua_ti, mus_ti, fx)
dd_al = depth_d(mua_al, mus_al, fx)
dp_al = depth_phi(mua_al, mus_al, fx)
dmc_al = depth_mc(mua_al, mus_al, fx)

#%% quick plot - penetration depths
if False:
    fig, ax = plt.subplots(num=1, ncols=1, nrows=1, figsize=(7,4))
    ax.plot(wv, dd_ti.T, linestyle='-', marker='*')
    ax.set_title('penetration depth (TiO$_2$) - diffusion')
    ax.set_xlabel(r'$\lambda$ (nm)')
    ax.set_ylabel('mm')
    ax.grid(True, linestyle=':')
    ax.set_ylim([0.25, 2])
    # shrink box to get external legend
    box = ax.get_position()
    # Put a legend to the right of the current axis
    ax.legend(['fx = {:.2f}'.format(f) for f in fx], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
#%% simulation: calculate mua = L(mua1 + mua2)
mus_ti_model = two_layer_model(mus_ti, mus_al, dp_ti, thick)

for _d, d in enumerate(mus_ti_model):
    fig, ax = plt.subplots(nrows=1, ncols=1, num=100+_d, figsize=(7,4))
    ax.plot(wv, d[:5,:].T, '-', marker='*')
    ax.plot(wv, mus_al[0,:], '--k')
    ax.plot(wv, mus_ti[0,:], '--k')
    ax.set_title(r"$\mu'_s$ (d = {}mm) - phi$^2$".format(thick[_d]))
    ax.set_xlabel('$\lambda$ (nm)')
    ax.set_ylabel('mm$^-1$')
    ax.grid(True, linestyle=':')
    # shrink box to get external legend
    box = ax.get_position()
    # Put a legend to the right of the current axis
    ax.legend(['fx = {:.2f}'.format(f) for f in fx[:5]], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()