# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:52:02 2022

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import pickle
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import Bounds
from matplotlib import pyplot as plt
from sfdi.common.getPath import getPath
from sfdi.common.readParams import readParams

def load_obj(name, path):
    """Utility function to load python objects using pickle module"""
    with open('{}/obj/{}.pkl'.format(path, name), 'rb') as f:
        return pickle.load(f)

def scatter_fun(lamb, a, b):
    """Power law function to fit scattering data to"""
    return a * np.power(lamb, -b)

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

def fluence_d(z, mua, mus, fx, n=1.4, g=0.8):
    """Fluence estimation with delta-P1 approximation"""
    C = -0.13755*n**3 + 4.339*n**2 - 4.90366*n + 1.6896
    gs = g/(g+1)
    muss = mus*(1-g**2)
    mut = mua + mus  # transport coefficient MxN
    muts = mua + muss  # transport coefficient* MxN
    mueff=np.sqrt(3*mua*mut)  # effective transport coefficient MxN
    mueff1 = np.sqrt(mueff**2 + (2*np.pi*fx[:,np.newaxis])**2)  # mueff, with fx (MxN)
    h = muts*2/3
    
    A = 3*muss*(muts + gs*mua)/(mueff1**2 - muts**2)
    B = (-A*(1 + C*h*muts) - 3*C*h*gs*muss)/(1 + C*h*mueff1)
    
    exp1 = np.exp(-muts[:,:,np.newaxis]*z)
    exp2 = np.exp(-mueff1[:,:,np.newaxis]*z)
    phi = (1+A[:,:,np.newaxis])*exp1 + B[:,:,np.newaxis]*exp2
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

def weights_fun3(x, mus, lamb=0):
    """Function to fit 2-layer scattering slope to a linear combination: mus = x*mus1 + (1-x)*mus2
    - x: FLOAT array
        (alpha, mus1, mus2)
    - mus: FLOAT
        measured scattering coefficient of 2-layer"""
    # alpha, mus1, mus2 = x  # Unpack
    alpha = x[0]
    mus1 = x[1:len(mus)+1]
    mus2 = x[len(mus)+1:]
    # import pdb; pdb.set_trace()
    return np.sum(((alpha*mus1 + (1-alpha)*mus2)/2 - mus)**2) + lamb*np.sum(x**2)
    
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
    
def alpha_p1(d, mua, mus, fx, g=0.8, n=1.4):
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
    C = -0.13755*n**3 + 4.339*n**2 - 4.90366*n + 1.6896
    gs = g/(g+1)
    muss = mus*(1-g**2)
    mut = mua + mus  # transport coefficient MxN
    muts = mua + muss  # transport coefficient* MxN
    mueff=np.sqrt(3*mua*mut)  # effective transport coefficient MxN
    mueff1 = np.sqrt(mueff**2 + (2*np.pi*fx[:,np.newaxis])**2)  # mueff, with fx (MxN)
    h = muts*2/3
    
    A = 3*muss*(muts + gs*mua)/(mueff1**2 - muts**2)
    B = (-A*(1 + C*h*muts) - 3*C*h*gs*muss)/(1 + C*h*mueff1)
    
    exp1 = np.exp(-muts*d)
    exp2 = np.exp(-mueff1*d)
    alpha = -((1+A)/muts*exp1 + B/mueff1*exp2)/((1+A)/muts + B/mueff1) + 1
    phi = (1+A)*exp1 + B*exp2
    
    return alpha, phi, mueff1
    
def fit_fun(x, mua, mus, fx, wv, g=0.8, n=1.4):
    """Function to minimize and fit for d, mus1, mus2
alpha is derived from delta-P1 approximation."""
    # unpack
    d = x[0]
    a1 = 10**x[1]
    b1 = x[2]
    a2 = 10**x[3]
    b2 = x[4]
    # mus1 = x[1:mus.shape[1]+1]
    # mus2 = x[mus.shape[1]+1:]
    mus1 = scatter_fun(wv, a1, b1)
    mus2 = scatter_fun(wv, a2, b2)
    
    # Coefficients. N: number of wavelengths, M: number of fx
    C = -0.13755*n**3 + 4.339*n**2 - 4.90366*n + 1.6896
    gs = g/(g+1)
    musp = mus*(1-g)
    muss = mus*(1-g**2)
    mut = mua + mus  # transport coefficient MxN
    muts = mua + muss  # transport coefficient* MxN
    mueff=np.sqrt(3*mua*mut)  # effective transport coefficient MxN
    mueff1 = np.sqrt(mueff**2 + (2*np.pi*fx[:,np.newaxis])**2)  # mueff, with fx (MxN)
    h = muts*2/3
    
    A = 3*muss*(muts + gs*mua)/(mueff1**2 - muts**2)
    B = (-A*(1 + C*h*muts) - 3*C*h*gs*muss)/(1 + C*h*mueff1)
    exp1 = np.exp(-muts*d)
    exp2 = np.exp(-mueff1*d)
    # import pdb; pdb.set_trace()
    
    min_fun = ((((1+A)/muts + B/mueff1 - ((1+A)/muts*exp1 + B/mueff1*exp2))*(mus1 - mus2)) /
                ((1+A)/muts + B/mueff1) + mus2 - musp)
    return np.sum(min_fun**2)
    

#%% Load pre-processed data
if __name__ == '__main__':
    data_path = getPath('select data path')
    par = readParams('{}/processing_parameters.ini'.format(data_path))  # optional
    data = load_obj('dataset', data_path)
    data.par = par
    fx = np.array([np.mean(par['fx'][i:i+4]) for i in range(len(par['fx'])-3)])
    wv = np.array([458,520,536,556,626])  # wavelengths
    
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
    d = 1.17  # mm
    al, _, mueff1 = alpha_diff(d, mua, mus/0.2, fx)
    alp, _, _ = alpha_p1(d, mua, mus/0.2, fx)
    delta = 1/mueff1
    
    #%% ITERATIVE fitting for d
    dz = 0.0001  # 0.1 um
    Z = np.arange(0, 10, dz)
        
    # Load dataset
    ret = data.singleROI('TiO10ml', fit='single', I=3e3, norm=None)
    mua = ret['op_ave'][:,:,0]  # average measured absorption coefficient
    mus = ret['op_ave'][:,:,1]  # average measured scattering coefficient
    # bm = ret['par_ave'][:, 1]  # average measured scattering slope
    
    phi = fluence(Z, mua, mus, fx, g=0)**2
    phi_d = fluence_d(Z, mua, mus/0.2, fx)**2
    sum_phi = np.sum(phi*dz, axis=-1)
    sum_phid = np.sum(phi_d*dz, axis=-1)
    # obtain alpha
    thick = np.ones(mus.shape, dtype=float)*-1
    alpha = np.zeros(mus.shape, dtype=float)
    
    # Approach 2: use mus[0] as top layer scattering and mus[-1] as bottom layer scattering
    for _f in range(mus.shape[0]):  # loop frequencies
        for _w in range(mus.shape[1]):  # loop wavelengths
            opt = minimize(weights_fun2, x0=np.array([.5]),
                           args=(mus[_f, _w], mus[0, _w], mus[-1, _w]),
                           method='Nelder-Mead',
                           bounds=Bounds([0], [1]),
                           options={'maxiter':3000, 'adaptive':False})
            alpha[_f, _w] = opt['x']  # fitted alpha
            # for _z, z in enumerate(Z):  # loop dept to find the value of z that best approximates alpha
            #     if np.sqrt((np.sum(phi_d[_f,_w,:_z]*dz)/sum_phid[_f,_w] - alpha[_f,_w])**2) <= 1e-3:
            #         thick[_f,_w] = z
    
    #%% ITERATIVE fitting for mus1, mus2 and d
    # dz = 0.0001  # 0.1 um
    # d = 0.125
    # Z = np.arange(0, 10, dz)
    # idx = np.where(Z >= d)[0][0]
    
    # Load dataset
    ret = data.singleROI('TiO30ml', fit='single', I=3e3, norm=None)
    mua = ret['op_ave'][:,:,0]  # average measured absorption coefficient
    mus = ret['op_ave'][:,:,1]  # average measured scattering coefficient
    
    # phi = fluence(Z, mua, mus/0.2, fx)
    # phi_d = fluence_d(Z, mua, mus/0.2, fx)
    # sum_phi = np.sum(phi*dz, axis=-1)
    # sum_phid = np.sum(phi_d*dz, axis=-1)
    # al = np.sum(phi[:,:,:idx]*dz, axis=-1)/sum_phi
    # alp = np.sum(phi_d[:,:,:idx]*dz, axis=-1)/sum_phid
    
    # obtain alpha
    thick = np.ones(mus.shape[1], dtype=float)*-1
    mus_1 = np.ones(mus.shape[1], dtype=float)*-1
    mus_2 = np.ones(mus.shape[1], dtype=float)*-1
    # for _f in range(mus.shape[0]):
    # opt = minimize(fit_fun, x0=np.concatenate((np.array([.1]),
    #                                                 np.ones(mus_1.shape),
    #                                                 np.ones(mus_1.shape))),
    opt = minimize(fit_fun, x0=np.array([.1, 2, 1, 2, 1]),
                    args=(mua, mus, fx, wv),
                    method='Nelder-Mead',
                    bounds=Bounds([0]*5, [np.inf]*5),
                    options={'maxiter':3000})
    thick = opt['x'][0]  # fitted alpha
    mus_1 = scatter_fun(wv, 10**opt['x'][1], opt['x'][2])
    mus_2 = scatter_fun(wv, 10**opt['x'][3], opt['x'][4])
        
        # for _z, z in enumerate(Z):  # loop dept to find the value of z that best approximates alpha
        #     if np.sqrt((np.sum(phi[_f,_w,:_z]*dz)/sum_phi[_f,_w] - alpha[_f,_w])**2) <= 1e-3:
        #         thick[_f,_w] = z


    #%% plot fluence over fx
    plt.figure(1)
    plt.plot(Z, phi[:,2,:].T)
    plt.grid(True, linestyle=':')
    plt.xlabel('z [mm]')
    plt.ylabel(r'$\varphi$(z) - {}nm'.format(data.par['wv'][2]))
    plt.title('Diffusion approximation')
    plt.xlim([0, 10])
    plt.ylim([0, 3.5])
    plt.legend([r'{:.3f} mm$^-1$'.format(x) for x in fx])
    plt.tight_layout()
    
    plt.figure(2)
    plt.plot(Z, phi_d[:,2,:].T)
    plt.grid(True, linestyle=':')
    plt.xlabel('z [mm]')
    plt.ylabel(r'$\varphi$(z) - {}nm'.format(data.par['wv'][2]))
    plt.title(r'$\delta$-P1')
    plt.xlim([0, 10])
    plt.ylim([0, 3.8])
    plt.legend([r'{:.3f} mm$^-1$'.format(x) for x in fx])
    plt.tight_layout()
    
    #%% Plot mus over fx
    plt.figure(22)
    plt.plot(fx, mus, '*', linestyle='solid')
    plt.grid(True, linestyle=':')
    plt.xlabel('fx')
    plt.ylabel(r"$\mu'_s$")
    plt.legend([str(x)+'nm' for x in data.par['wv']], title=r'$\lambda$')
    plt.tight_layout()
