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
from matplotlib import cm as cmap
from sfdi.analysis.depthMC import depthMC

def scatter_fun(lamb, a, b):
    """Power law function to fit scattering data to"""
    return a * np.power(lamb, -b)

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
    
    RETURN
    - mus_model: list of arrays containing mus of 2-layer for each thickness
    - params: list of arrays, containing fitted A,B parameters
    """
    mus_model = []
    params = []
    for th in thick:
        ret = (mus1 * th + mus2 * (depths - th)) / depths
        idx = np.where(th/depths >= 1)
        if len(idx[0]) > 0:
            ret[idx] = mus1[idx]
        mus_model.append(ret)
        temp = np.zeros((mus1.shape[0], 2), dtype=float)
        for _f in range(mus1.shape[0]):
            temp[_f,:], _ = curve_fit(scatter_fun, wv, ret[_f,:], p0=[100, 1],
                        method='trf', loss='soft_l1', max_nfev=2000)
        params.append(temp)
    return mus_model, params

def two_layer_model2(mus1, mus2, depths, thick):
    """Linear combination of mus on 2-layer model
    - mus1: scattering vector, thin layer (N x M)
    - mus2: scattering vector, thick layer (N x M)
    - depths: equivalent penetration depths [(N x M) x D]
    - thick: thicness of top layer (D x 1)
        N: number of frequencies
        M: number of wavelengths
        D: number of thin phantoms
    
    RETURN
    - mus_model: list of arrays containing mus of 2-layer for each thickness
    - params: list of arrays, containing fitted A,B parameters
    """
    mus_model = []
    params = []
    for _t, th in enumerate(thick):
        ret = (mus1 * th + mus2 * (depths[_t] - th)) / depths[_t]
        idx = np.where(th/depths[_t] >= 1)
        if len(idx[0]) > 0:
            ret[idx] = mus1[idx]
        mus_model.append(ret)
        temp = np.zeros((mus1.shape[0], 2), dtype=float)
        for _f in range(mus1.shape[0]):
            temp[_f,:], _ = curve_fit(scatter_fun, wv, ret[_f,:], p0=[100, 1],
                        method='trf', loss='soft_l1', max_nfev=2000)
        params.append(temp)
    return mus_model, params

## Parameters
wv = np.array([458,520,536,556,626])  # wavelengths
F = np.arange(0,0.51,0.05)
fx = np.array([np.mean(F[a:a+3]) for a in range(len(F)-3)])  # average fx
thick = np.array([0.125, 0.265, 0.51, 0.67, 1.17])  # layer thickness

## optical properties are stacked over fx (assume homogeneous)
# Top layer: Titanium oxide
mua_top = np.tile(np.array([0.0315582, 0.0368862, 0.0368169, 0.0367965, 0.0428225]), (8,1))
a1,b1 = np.array([7.61528368e+03, 1.25263885e+00])
mus_top = np.tile((a1 * np.power(wv, -b1)), (8,1))
# Bottom layer: Aluminum oxide
mua_bottom = np.tile(np.array([0.02110538, 0.0223065 , 0.02223373, 0.02278003, 0.02402116]), (8,1))
a2,b2 = np.array([9.88306737, 0.3909216 ])
mus_bottom = np.tile((a2 * np.power(wv, -b2)), (8,1))

# Calculate penetration depths (Nfreq. x NWv)
dd_top = depth_d(mua_top, mus_top, fx)
dp_top = depth_phi(mua_top, mus_top, fx)
dmc_top = depth_mc(mua_top, mus_top, fx)
dd_bottom = depth_d(mua_bottom, mus_bottom, fx)
dp_bottom = depth_phi(mua_bottom, mus_bottom, fx)
dmc_bottom = depth_mc(mua_bottom, mus_bottom, fx)

# "Equivalent" penetration depths [(Nfreq. x Nwv) x Nthick.]
dd_eq = []
dp_eq = []
dmc_eq = []
for d in thick:
    dd_eq.append(d + (dd_top - d)*dd_bottom/dd_top)
    idx = np.where(dd_top <= d)
    dd_eq[-1][idx] = dd_top[idx]
    
    dp_eq.append(d + (dp_top - d)*dp_bottom/dp_top)
    idx = np.where(dp_top <= d)
    dp_eq[-1][idx] = dp_top[idx]
    
    dmc_eq.append(d + (dmc_top - d)*dmc_bottom/dmc_top)
    idx = np.where(dmc_top <= d)
    dmc_eq[-1][idx] = dmc_top[idx]
#%% quick plot - penetration depths
if False:
    for _d, d in enumerate(thick):
        fig, ax = plt.subplots(num=1+_d, ncols=1, nrows=1, figsize=(7,4))
        ax.plot(wv, dd_eq[_d].T, linestyle='-', marker='*')
        ax.set_title('Equivalent penetration depth - diffusion - d={}mm'.format(d))
        ax.set_xlabel(r'$\lambda$ (nm)')
        ax.set_ylabel('mm')
        ax.grid(True, linestyle=':')
        ax.set_ylim([0, 2.5])
        # shrink box to get external legend
        box = ax.get_position()
        # Put a legend to the right of the current axis
        ax.legend(['fx = {:.2f}'.format(f) for f in fx], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
    
#%% simulation: calculate mua = L(mua1 + mua2)
mus_top_model, params = two_layer_model(mus_top, mus_bottom, dd_top, thick)

for _d, d in enumerate(mus_top_model):
    fig, ax = plt.subplots(nrows=1, ncols=1, num=100+_d, figsize=(7,4))
    ax.plot(wv, d[:5,:].T, '-', marker='*')
    ax.plot(wv, mus_bottom[0,:], '--k')
    ax.plot(wv, mus_top[0,:], '--k')
    ax.set_title(r"$\mu'_s$ (d = {}mm) - MC".format(thick[_d]))
    ax.set_xlabel('$\lambda$ (nm)')
    ax.set_ylabel('mm$^{-1}$')
    ax.grid(True, linestyle=':')
    # shrink box to get external legend
    box = ax.get_position()
    # Put a legend to the right of the current axis
    ax.legend(['fx = {:.2f}'.format(f) for f in fx[:5]], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

blues = cmap.Blues(np.linspace(0.2,1,8))
reds = cmap.OrRd(np.linspace(0.2,1,8))
fig, ax = plt.subplots(nrows=1, ncols=2, num=66, figsize=(12,4))
for _p, par in enumerate(params):
    ax[0].plot(fx, par[:,0], '*', color=reds[_p], linestyle='-', linewidth=1.5, markersize=9)
    ax[1].plot(fx, par[:,1], '*', color=reds[_p], linestyle='-', linewidth=1.5, markersize=9)
# reference parameters (AlO and TiO)
ax[0].plot([0.05, 0.4], [a1, a1], '--k')
ax[0].plot([0.05, 0.4], [a2, a2], '--k')
ax[1].plot([0.05, 0.4], [b1, b1], '--k')
ax[1].plot([0.05, 0.4], [b2, b2], '--k')

ax[0].set_yscale('log')
ax[0].set_xlabel('fx')
ax[0].set_title('A parameter')
ax[0].grid(True, linestyle=':')
ax[1].set_xlabel('fx')
ax[1].set_title('B parameter')
ax[1].grid(True, linestyle=':')

box = ax[1].get_position()
# Put a legend to the right of the current axis
ax[1].legend(['{}mm'.format(x) for x in thick], loc='center left',
             bbox_to_anchor=(1, 0.5), title='Thickness', title_fontsize='large')
plt.tight_layout()

#%% Reverse fit?
