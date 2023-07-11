# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:05:01 2022

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

A script to collect commonly used physics models of light transport

Available functions:
x    - phi_diff: fluence based on standard diffusion approximation (with fx)
x    - phi_diffusion: fluence based on SDA (Seo's thesis)
x    - phi_deltaP1: fluence based on delta-P1 approximation (my approach, variation of Vasen original
                                                            with fx. To obtain the original
                                                            formulation, use fx=0)
    
x   - phi_dP1: fluence based on delta-P1 approximation (Seo's thesis, with fx)

x   - phi_2lc: 2-layer model of fluence, continuous. phi = A*phi1 + (1-A)*phi2
    - phi_2lp: 2-layer model of fluence, piecewise continuous.
"""
import numpy as np
from cycler import cycler

def phi_diff(z, mua, mus, fx, n=1.4, g=0.8):
    """Function to calculate fluence of light in depth based on the standard diffuse approximation.
    - mua, mus: vectors of optical properties (N x M). Note, mus is the scattering coefficient,
                not the reduced scattering coefficient (multiply by (1-g))
    - fx: average fx in range (N x 1)
    - z: depth (1 x Z)
    - n: refractive index (default = 1.4)
    - g: anisotropy coefficient (default = 0.8)
        N: number of frequencies
        M: number of wavelengths
        Z: number of depths
        
    RETURN
    - fluence: array of light fluences (N x M x Z)
"""
    musp = mus * (1-g)
    mut = mua + musp
    mueff = np.sqrt(np.abs(3 * mua * mut))
    mueff1 = np.sqrt(mueff**2 + (2*np.pi*fx[:,np.newaxis])**2)
    a1 = musp / mut  # albedo
    # effective reflection coefficient. Assume n = 1.4
    Reff = 0.0636*n + 0.668 + 0.71/n - 1.44/n**2
    A = (1 - Reff)/(2*(1 + Reff))  # coefficient
    C = (-3*a1*(1 + 3*A))/((mueff1**2/mut**2 - 1) * (mueff1/mut + 3*A))
    fluence = (3*a1 / (mueff1**2 / mut**2 - 1))[:,:,np.newaxis] * \
        np.exp(-mut[:,:,np.newaxis] * z[np.newaxis,:]) +\
        C[:,:,np.newaxis] * np.exp(-mueff1[:,:,np.newaxis] * z[np.newaxis,:])    
    return fluence

def phi_diffusion(z, mua, mus, fx, n=1.4, g=0.8):
    """Function to calculate fluence of light in depth based on the SDA (from Seo's thesis).
    - mua, mus: vectors of optical properties (N x M). Note, mus is the scattering coefficient,
                not the reduced scattering coefficient (multiply by (1-g))
    - fx: average fx in range (N x 1)
    - z: depth (1 x Z)
    - n: refractive index (default = 1.4)
    - g: anisotropy coefficient (default = 0.8)
        N: number of frequencies
        M: number of wavelengths
        Z: number of depths
        
    RETURN
    - fluence: array of light fluences (N x M x Z)
"""
    musp = mus*(1-g)
    mut = mua + musp
    mueff = np.sqrt(np.abs(3*mua*mut))
    mueff1 = np.sqrt(mueff**2 + (2*np.pi*fx[:,np.newaxis])**2)
    a1 = musp/mut
    Reff = 0.0636*n + 0.668 + 0.71/n - 1.44/n**2
    # Reff = 0.493
    # NOTE: this polynomial for Reff evaluates to 0.5295, In Seo thesis is 0.493
    A = (1 - Reff)/(2*(1 + Reff))  # coefficient
    
    Cp_dc = 3*mut*musp/(mueff**2 - mut**2)
    Cp_ac = 3*mut*musp/(mueff1**2 - mut**2)
    Ch_dc = -Cp_dc*(3*A+1) / ((mueff/mut) + 3*A)
    Ch_ac = -Cp_ac*(3*A+1) / ((mueff1/mut) + 3*A)
    
    phi_dc = Ch_dc[:,:,np.newaxis] * np.exp(-mueff[:,:,np.newaxis]*z[np.newaxis,:]) +\
        Cp_dc[:,:,np.newaxis] * np.exp(-mut[:,:,np.newaxis]*z[np.newaxis,:])
    phi_ac = Ch_ac[:,:,np.newaxis] * np.exp(-mueff1[:,:,np.newaxis]*z[np.newaxis,:]) +\
        Cp_ac[:,:,np.newaxis]*np.exp(-mut[:,:,np.newaxis]*z[np.newaxis,:])
    # return phi_dc,phi_ac
    return phi_ac

def phi_deltaP1(z, mua, mus, fx, n=1.4, g=0.8):
    """Function to calculate fluence of light in depth based on the delta-P1 approximation.
    - mua, mus: vectors of optical properties (N x M). Note, mus is the scattering coefficient,
                not the reduced scattering coefficient (multiply by (1-g))
    - fx: average fx in range (N x 1). For original d-p1, use zeros.
    - z: depth (1 x Z)
    - n: refractive index (default = 1.4)
    - g: anisotropy coefficient (default = 0.8)
        N: number of frequencies
        M: number of wavelengths
        Z: number of depths
        
    RETURN
    - fluence: array of light fluences (N x M x Z)
"""
    C = -0.13755*n**3 + 4.339*n**2 - 4.90366*n + 1.6896
    gs = g/(g+1)
    muss = mus*(1-g**2)
    mut = mua + mus  # transport coefficient MxN
    muts = mua + muss  # transport coefficient* MxN
    mueff = np.sqrt(np.abs(3 * mua * mut))  # effective transport coefficient MxN
    mueff1 = np.sqrt(mueff**2 + (2*np.pi*fx[:,np.newaxis])**2)  # mueff, with fx (MxN)
    h = mut*2/3
    
    A = 3*muss*(muts + gs*mua)/(mueff1**2 - muts**2)
    B = (-A*(1 + C*h*muts) - 3*C*h*gs*muss)/(1 + C*h*mueff1)
    
    exp1 = np.exp(-muts[:,:,np.newaxis]*z)
    exp2 = np.exp(-mueff1[:,:,np.newaxis]*z)
    fluence = (1 + A[:,:,np.newaxis])*exp1 + B[:,:,np.newaxis]*exp2
    return fluence


def phi_dP1(z, mua, mus, fx, n=1.4, g=0.8):
    """Fluence estimation with delta-P1 approximation, as seen in Seo thesis.
NOTE: only the AC component of fluence is derived in the thesis.
    - mua, mus: vectors of optical properties (N x M). Note, mus is the scattering coefficient,
                not the reduced scattering coefficient (multiply by (1-g))
    - fx: average fx in range (N x 1).
    - z: depth (1 x Z)
    - n: refractive index (default = 1.4)
    - g: anisotropy coefficient (default = 0.8)
        N: number of frequencies
        M: number of wavelengths
        Z: number of depths
        
    RETURN
    - fluence: array of light fluences (N x M x Z)
"""
    R = -0.13755*n**3 + 4.339*n**2 - 4.90366*n + 1.6896
    # gs = g/(g+1)
    muss = mus*(1-g**2)
    mut = mua + mus  # transport coefficient MxN
    muts = mua + muss  # transport coefficient* MxN
    mueff = np.sqrt(np.abs(3 * mua * mut))  # effective transport coefficient MxN
    mueff1 = np.sqrt(mueff**2 + (2*np.pi*fx[:,np.newaxis])**2)  # mueff, with fx (MxN)
    # h = mut*2/3
    
    C = 3/2 * muss*mut/(mueff1)
    zb = 2/(3*mut)*R
    
    exp1 = np.exp(-muts[:,:,np.newaxis]*z)
    exp2 = np.exp(-mueff1[:,:,np.newaxis]*z)
    exp3 = np.exp(-mueff1[:,:,np.newaxis]*(z+2*zb[:,:,np.newaxis]))
    # import pdb; pdb.set_trace()
    
    phi = (C[:,:,np.newaxis]/(mueff1[:,:,np.newaxis] - muts[:,:,np.newaxis]) * (exp1 - exp2) +
           C[:,:,np.newaxis]/(mueff1[:,:,np.newaxis] + muts[:,:,np.newaxis]) * (exp1 - exp3))
    return phi


def phi_2lc(alpha, phi_bottom, phi_top):
    """Linear model of 2-layer fluence (moslty used to fit for alpha).
    phi_2lc = alpha*phi_bottom + (1-alpha)*phi_top
    - phi_bottom: fluence array of bottom layer (N x M x Z)
    - phi_top: fluence array of top layer (N x M x Z)
    - alpha: percentage (or "weight") of fluence contribution from bottom layer (1 x A)
        N: number of frequencies
        M: number of wavelengths
        Z: number of depths
    RETURN
    - fluence: fluence array given by linear combination of phi1, phi2 (N x M x Z x A)
"""
    return alpha*phi_bottom[...,np.newaxis] + (1-alpha)*phi_top[...,np.newaxis]

def phi_2lc_2(x, phi_bottom, phi_top):
    """Linear model of 2-layer fluence, with 2 parameters (moslty used to fit for alpha).
    phi_2lc = alpha*phi_bottom + (1-alpha)*phi_top
    - phi_bottom: fluence array of bottom layer (N x M x Z)
    - phi_top: fluence array of top layer (N x M x Z)
    - x: percentage (or "weight") of fluence contribution from top and bottom layer (2 x A)
        N: number of frequencies
        M: number of wavelengths
        Z: number of depths
    RETURN
    - fluence: fluence array given by linear combination of phi1, phi2 (N x M x Z x A)
"""
    alpha = x[0]
    beta = x[1]
    return alpha*phi_bottom[...,np.newaxis] + beta*phi_top[...,np.newaxis]


def phi_2lp(d, phi_top, phi_bottom, z):
    """Piecewise-continuous model of 2-layer fluence.
    - d: array with thickness of top layer (1 x A) [mm]
    - phi_top, phi_bottom: fluence array of top and bottom layers (N x M x Z)
    - z: array of depths at which fluence was calculated (1 x Z)
        N: number of frequencies
        M: number of wavelengths
        Z: number of depths
    RETURN
    - phi: fluence array of 2-layer model, piecewise continuous (N x M x Z x A)
    """
    phi = np.zeros((phi_top.shape + d.shape))
    for _i, _d in enumerate(d):
        idx = np.where(z >= _d)[0][0]
        phi[:, :, :idx, _i] = phi_top[:, :, :idx]
        phi[:, :, idx:, _i] = phi_bottom[:, :, idx:] *\
            phi_top[:,:,idx,np.newaxis] / phi_bottom[:,:,idx,np.newaxis]
    return phi
    
    
    

def alpha(phi, z, d, ax=-1):
    """"Function to calculate the "weigth" of the contribution of a thin layer based on fluence.
    
Given the light fluence phi, alpha is calculated as 
    
    - phi: array of light fluence in depth
    - z: depths array at which phi was calculated
    - d: thickness of the thin layer at which alpha is calculated
    - ax: axis containing the depth dimension, to perform the sum [default = -1]

"""
    # import pdb; pdb.set_trace()    
    dz = z[1] - z[0]
    idx = np.where(z >= d)[0][0]
    return np.sum(phi[...,:idx] * dz, axis=ax) / np.sum(phi * dz, axis=ax)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from matplotlib import cm
    import addcopyfighandler    
    
    if True:
        dz = 0.001
        mua = np.array([[0.05]])
        mus = np.array([[5]])*5
        z = np.arange(0,10,dz)
        fx = np.array([0, 0.1, 0.2, 0.3])
        
        phi1 = phi_diff(z, mua, mus, fx)
        phi2 = phi_deltaP1(z, mua, mus, fx)
        phi3 = phi_dP1(z, mua, mus, fx)
        
        titles=['SDA', r'$\delta-P1$', r'mod-$\delta-P1$']
        cyc = (cycler(color=['tab:blue','tab:orange','tab:green','tab:red'])+
               cycler(linestyle=['-', '--', '-.', ':']))
        
        fig, ax = plt.subplots(1,3, figsize=(10,3.5))
        for _i, phi in enumerate([phi1, phi3, phi2]):
            ax[_i].plot(z, np.squeeze(phi).T/np.sum(np.squeeze(phi*dz), axis=-1))
            plt.rc('axes', prop_cycle=cyc)
            if _i == 0:
                ax[_i].legend([r'fx={}mm$^{{-1}}$'.format(x) for x in fx])
                ax[_i].set_ylabel(r'$\phi(z)$')
            ax[_i].set_xlim([0, 3])
            ax[_i].set_ylim([0, 2])
            ax[_i].set_title(titles[_i])
            ax[_i].set_xlabel('z (mm)')
            ax[_i].grid(True, linestyle=':')
        plt.tight_layout()
    
    
    if False:
        dz = 0.001
        mua1 = np.array([[0.05]])
        mus1 = np.array([[5]])
        mua2 = np.array([[0.05]])
        mus2 = np.array([[20]])
        z = np.arange(0,10,dz)
        # fx = np.array([0, 0.1, 0.2, 0.3])
        fx = np.array([0])
        
        phi_top = np.squeeze(phi_deltaP1(z, mua2, mus2, fx))
        phi_bottom = np.squeeze(phi_deltaP1(z, mua1, mus1, fx))
        alpha = np.arange(0, 1.1, 0.2)
        col = cm.get_cmap('Blues', len(alpha))
        # phi_d = alpha * phi_bottom[:,np.newaxis] + (1 - alpha)* phi_top[:,np.newaxis]
        phi_d = phi_2lc(alpha, phi_bottom, phi_top)
        phi_dn = phi_d / np.sum(phi_d * dz, axis=0)  # normalized to total fluence
        
        # "RAW" fluence
        fig = plt.figure(num=1, figsize=(6,4))
        for _i,phi in enumerate(phi_d.T):
            if _i == 0:
                color = '#FF0000'  # Top layer
            elif _i == len(phi_d.T)-1:
                color = '#00FF00'  # Bottom layer
            else:
                color = col(_i)
            plt.plot(z, phi, color=color, label=r'$\alpha$={:.1f}'.format(alpha[_i]))
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.xlabel('mm')
        plt.title(r'$\varphi = \alpha \varphi_b + (1 - \alpha) \varphi_t$')
        plt.tight_layout()
        
        # normalized fluence
        fig = plt.figure(num=2, figsize=(6,4))
        for _i,phi in enumerate(phi_dn.T):
            if _i == 0:
                color = '#FF0000'  # Top layer
            elif _i == len(phi_d.T)-1:
                color = '#00FF00'  # Bottom layer
            else:
                color = col(_i)
            plt.plot(z, phi, color=color, label=r'$\alpha$={:.1f}'.format(alpha[_i]))
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.xlabel('mm')
        plt.title(r'$\varphi = \alpha \varphi_b + (1 - \alpha) \varphi_t$')
        plt.tight_layout()