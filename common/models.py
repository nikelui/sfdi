# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:05:01 2022

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

A script to collect commonly used physics models of light transport

Available functions:
x    - phi_diff: fluence based on standard diffusion approximation (with fx)
x    - phi_deltaP1: fluence based on delta-P1 approximation (my approach, variation of Vasen original
                                                            with fx. To obtain the original
                                                            formulation, use fx=0)
    
x    - phi_dP1: fluence based on delta-P1 approximation (Seo's thesis, with fx)

    - phi_2lc: 2-layer model of fluence, continuous. phi = A*phi1 + (1-A)*phi2
    - phi_2lp: 2-layer model of fluence, piecewise continuous.
    
"""
import numpy as np

def phi_diff(mua, mus, fx, z, n=1.4, g=0.8):
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


def phi_deltaP1(mua, mus, fx, z, n=1.4, g=0.8):
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


def phi_dP1(mua, mus, fx, z, n=1.4, g=0.8):
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



if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from matplotlib import cm
    import addcopyfighandler    
    
    if False:
        dz = 0.001
        mua = np.array([[0.05]])
        mus = np.array([[10]])
        z = np.arange(0,10,dz)
        fx = np.array([0, 0.1, 0.2, 0.3])
        
        phi1 = phi_diff(mua, mus, fx, z)
        phi2 = phi_deltaP1(z, mua, mus, fx)
        phi3 = phi_dP1(z, mua, mus, fx)
        
        titles=['diffusion', r'$\delta-P1$ (Luigi)', r'$\delta-P1$ (Seo)']
        
        for _i, phi in enumerate([phi1, phi2, phi3]):
            plt.figure(num=_i+1, figsize=(6,4))
            plt.plot(z, np.squeeze(phi).T)
            plt.legend([r'fx={}mm$^{{-1}}$'.format(x) for x in fx])
            plt.title(titles[_i])
            plt.xlabel('mm')
            plt.ylabel(r'$\varphi$')
            plt.grid(True, linestyle=':')
            plt.tight_layout()
    
    
    if True:
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
