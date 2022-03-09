# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:52:02 2022

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import pickle
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from sfdi.common.getPath import getPath
from sfdi.common.readParams import readParams

def load_obj(name, path):
    """Utility function to load python objects using pickle module"""
    with open('{}/obj/{}.pkl'.format(path, name), 'rb') as f:
        return pickle.load(f)

def depth_diff(mua, mus, fx):
    """Function to calculate effective penetration depth based on diffusion approximation
        
    - mua, mus: FLOAT array (MxN)
        optical properties
    - fx: FLOAT array (1xM)
        average fx in range
        
        M: number of spatial frequencies
        N: number of wavelengths
        
    Returns
    ------
    d: FLOAT array (MxN)
        Penetration depth (diffuse) dependent on fx and wavelength
"""
    mut = mua + mus  # 1xN
    mueff = np.sqrt(np.abs(3 * mua * mut))  # 1xN
    mueff1 = np.sqrt(mueff**2 + (2*np.pi*fx[:,np.newaxis])**2)  # MxN
    d = 1/mueff1  # MxN
    return d

def fluence(z, mua, mus, fx, n=1.4, g=0.8):
    musp = mus*(1-g)  # reduced scattering coefficient
    Reff = 0.0636*n + 0.668 + 0.71/n - 1.44/n**2  # effective reflection coefficient
    mut = mua + musp  # transport coefficient (1xN)
    mueff = np.sqrt(3*mua*mut)  # effective transport coefficient (1xN)
    mueff1 = np.sqrt(mueff**2 + (2*np.pi*fx[:,np.newaxis])**2)  # mueff, with fx (MxN)
    #TODO: debug the equation (maybe calculate phi?). it return negative values for alpha    
    A = (3*musp/mut)/(mueff1**2/mut**2 - 1)  # MxN
    R = (1-Reff)/(2*(1+Reff))
    B = -3*(musp/mut)*(1+3*R) / ((mueff1**2/mut**2 - 1)*(mueff1/mut + 3*R))  # MxN
    exp1 = np.exp(-mut[:,:,np.newaxis]*z)
    exp2 = np.exp(-mueff1[:,:,np.newaxis]*z)
    
    phi = A[:,:,np.newaxis]*exp1 + B[:,:,np.newaxis]*exp2  # MxNxZ
    return phi

def fluence_d(z, mua, mus, n=1.4, g=0.8):
    """Fluence estimation with delta-P1 approximation"""
    C = -0.13755*n**3 + 4.339*n**2 - 4.90366*n + 1.6896
    gs = g/(g+1)
    muss = mus*(1-g**2)
    mut = mua + mus
    muts = mua + muss
    mueff=np.sqrt(3*mua*mut)
    h = muts*2/3
    
    A = 3*muss*(muts + gs*mua)/(mueff**2 - muts**2)
    B = (-A*(1 + C*h*muts) - 3*C*h*gs*muss)/(1 + C*h*mueff)
    
    exp1 = np.exp(-muts[:,:,np.newaxis]*z)
    exp2 = np.exp(-mueff[:,:,np.newaxis]*z)
    phi = A[:,:,np.newaxis]*exp1 + B[:,:,np.newaxis]*exp2
    return phi
    
def weights_fun(x, Bm, B1, B2):
    """Function to fit 2-layer scattering slope to a linear combination: B = x*B1 + (1-x)*B2
    - x: FLOAT
        x = weight of the linear combination
    - Bm: FLOAT, measured slope of 2-layer
    - B1, B2: FLOAT, measured slope of individual layers"""
    return np.sum(np.sqrt((x*B1 + (1-x)*B2 - Bm)**2))

def weights_fun2(x, mus, mus1, mus2):
    """Function to fit 2-layer scattering slope to a linear combination: mus = x*mus1 + (1-x)*mus2
    - x: FLOAT
        weight of the linear combination
    - mus: FLOAT
        measured scattering coefficient of 2-layer
    - mus1, mus2: FLOAT
        measured scattering coefficient of individual layers"""
    return np.sum(np.sqrt((x*mus1 + (1-x)*mus2 - mus)**2))

def alpha_diff(d, mua, mus, fx, g=0.8, n=1.4):
    """
    Function to calculate the expected weigth (alpha) and fluence (phi) in a 2 layer model using
    the diffusion approximtion, given the thickness d and the measured optical properties.

Model:
__________
____1_____ | |----(alpha)
           |---
           |---(1-alpha)
____2____  |---

    Parameters
    ----------
    d : FLOAT
        Thickness of the thin layer.
    mua : FLOAT array (MxN)
        Absorption coefficient, dependent on wavelength and fx.
    mus : FLOAT array (MxN)
        Scattering coefficient, dependent on wavelength and fx.
    fx :FLOAT array (Mx1)
        Array with spatial frequencies
    g : FLOAT, optional
        Anysotropy coefficient. The default is 0.8.
    n : FLOAT, optional
        Index of refraction. The default is 1.4.

    Returns
    -------
    alpha : FLOAT array (MxN)
        Expected partial volume in a 2-layer model.
        M: number of spatial frequencies
        N: number of wavelengths
    """
    # Coefficients. N: number of wavelengths, M: number of fx
    # import pdb; pdb.set_trace()  # DEBUG

    musp = mus*(1-g)  # reduced scattering coefficient
    Reff = 0.0636*n + 0.668 + 0.71/n - 1.44/n**2  # effective reflection coefficient
    mut = mua + musp  # transport coefficient (1xN)
    mueff = np.sqrt(3*mua*mut)  # effective transport coefficient (1xN)
    mueff1 = np.sqrt(mueff**2 + (2*np.pi*fx[:,np.newaxis])**2)  # mueff, with fx (MxN)
    #TODO: debug the equation (maybe calculate phi?). it return negative values for alpha    
    A = (3*musp/mut)/(mueff1**2/mut**2 - 1)  # MxN
    R = (1-Reff)/(2*(1+Reff))
    B = -3*(musp/mut)*(1+3*R) / ((mueff1**2/mut**2 - 1)*(mueff1/mut + 3*R))  # MxN
    
    alpha = -(A/mut * np.exp(-mut*d) + B/mueff1 * np.exp(-mueff1*d)) / (A/mut + B/mueff1) + 1
    phi = (A+1)*np.exp(-mut*d)+B*np.exp(-mueff1*d)
    
    return alpha, phi, mueff1
    

if __name__ == '__main__':
    fx = np.arange(0.05, 0.45, 0.05)
    
    #%% Load pre-processed data
    data_path = getPath('select data path')
    par = readParams('{}/processing_parameters.ini'.format(data_path))  # optional
    data = load_obj('dataset', data_path)
    data.par = par
    
    TiO = data.singleROI('TiObaseTop', fit='single', I=3e3, norm=None)
    mua_TiO = np.mean(TiO['op_ave'][:,:,0], axis=0)  # averages over fx for homogeneous phantom
    mus_TiO = np.mean(TiO['op_ave'][:,:,1], axis=0)
    par_TiO = np.mean(TiO['par_ave'], axis=0)

    AlO = data.singleROI('AlObaseTop', fit='single', I=3e3, norm=None)
    mua_AlO = np.mean(AlO['op_ave'][:,:,0], axis=0)  # averages over fx for homogeneous phantom
    mus_AlO = np.mean(AlO['op_ave'][:,:,1], axis=0)
    par_AlO = np.mean(AlO['par_ave'], axis=0)
    #%% Load dataset
    ret = data.singleROI('TiO30ml', fit='single', I=3e3, norm=None)
    mua = ret['op_ave'][:,:,0]  # average measured absorption coefficient
    mus = ret['op_ave'][:,:,1]  # average measured scattering coefficient
    bm = ret['par_ave'][:, 1]  # average measured scattering slope
    
    delta = depth_diff(mua, mus, fx)/2
    # average over wv?
    d = np.mean(ret['depths'], axis=1)[:]  # delta/2
    d2 = np.mean(ret['depth_phi'], axis=1)[:]  # 1/e * phi^2
    d3 = np.mean(ret['depth_MC'], axis=1)[:]  # calculated via Monte Carlo model
    
    alpha = np.zeros(mus.shape, dtype=float)
    for _f in range(mus.shape[0]):
        for _w in range(mus.shape[1]):
            opt = minimize(weights_fun2, x0=np.array([.5]),
                           args=(mus[_f,_w], mus_TiO[_w], mus_AlO[_w]),
                           method='Nelder-Mead',
                           # bounds=Bounds([0], [1]),
                           options={'maxiter':3000, 'adaptive':False})
            alpha[_f, _w] = opt['x']
       
    # expected values of alpha for thickness d
    d = 0.67  # mm
    al, _, mueff1 = alpha_diff(d, mua, mus, fx, g=0)
    delta = 1/mueff1
    
    #%% ITERATIVE fitting for d
    dz = 0.0001  # 1 um
    Z = np.arange(0, 10, dz)
        
    # Load dataset
    ret = data.singleROI('TiO10ml', fit='single', I=3e3, norm=None)
    mua = ret['op_ave'][:,:,0]  # average measured absorption coefficient
    mus = ret['op_ave'][:,:,1]  # average measured scattering coefficient
    # bm = ret['par_ave'][:, 1]  # average measured scattering slope
    
    phi = fluence(Z, mua, mus, fx, g=0)
    phi_d = fluence_d(Z, mua, mus/0.2)
    sum_phi = np.sum(phi*dz, axis=-1)
    sum_phid = np.sum(phi_d*dz, axis=-1)
    # obtain alpha
    thick = np.ones(mus.shape, dtype=float)*-1
    alpha = np.zeros(mus.shape, dtype=float)
    for _f in range(mus.shape[0]):  # loop frequencies
        for _w in range(mus.shape[1]):  # loop wavelengths
            opt = minimize(weights_fun2, x0=np.array([.5]),
                           args=(mus[_f,_w], mus_TiO[_w], mus_AlO[_w]),
                           method='Nelder-Mead',
                           # bounds=Bounds([0], [1]),
                           options={'maxiter':3000, 'adaptive':False})
            alpha[_f, _w] = opt['x']  # fitted alpha
            for _z, z in enumerate(Z):  # loop dept to find the value of z that best approximates alpha
                if np.sqrt((np.sum(phi_d[_f,_w,:_z]*dz)/sum_phid[_f,_w] - alpha[_f,_w])**2) <= 1e-3:
                    thick[_f,_w] = z
    
    
    #%%
    plt.figure(1)
    plt.plot(Z, phi[0,:,:].T)
    plt.grid(True, linestyle=':')
    plt.xlabel('z [mm]')
    plt.ylabel(r'$\varphi$(z)')
    plt.title('Diffusion approximation')
    plt.xlim([0, 10])
    plt.ylim([0, 4])
    plt.legend([str(x)+' nm' for x in data.par['wv']])
    plt.tight_layout()
    
    plt.figure(2)
    plt.plot(Z, phi_d[0,:,:].T)
    plt.grid(True, linestyle=':')
    plt.xlabel('z [mm]')
    plt.ylabel(r'$\varphi$(z)')
    plt.title(r'$\delta$-P1')
    plt.xlim([0, 10])
    plt.ylim([0, 4])
    plt.legend([str(x)+' nm' for x in data.par['wv']])
    plt.tight_layout()