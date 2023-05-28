# -*- coding: utf-8 -*-
"""
Created on Fri May 13 08:40:14 2022

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp2d
from sfdi.analysis.depthCalculator.data import __path__ as dep_path
import addcopyfighandler

class fluence:
    def __init__(self, z, mua, mus, fx, method, n=1.4, g=0.8):
        self.z = z
        self.mua = mua
        self.mus = mus
        self.fx = fx
        self.n = n
        self.g = g
        self.method = method
        
        self.phi = np.squeeze(method(z, mua, mus, fx, n, g))**2
        self.cumsum = np.cumsum(self.phi*(z[1]-z[0]), axis=1)
        self.phitot = np.sum(self.phi*(z[1]-z[0]), axis=1)

def diffusion(z, mua, mus, fx, n=1.4, g=0.8):
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

def delta_p1(z, mua, mus, fx, n=1.4, g=0.8):
    """Fluence estimation with delta-P1 approximation"""
    C = -0.13755*n**3 + 4.339*n**2 - 4.90366*n + 1.6896
    gs = g/(g+1)
    muss = mus*(1-g**2)
    mut = mua + mus  # transport coefficient MxN
    muts = mua + muss  # transport coefficient* MxN
    mueff=np.sqrt(3*mua*muts)  # effective transport coefficient MxN
    mueff1 = np.sqrt(mueff**2 + (2*np.pi*fx[:,np.newaxis])**2)  # mueff, with fx (MxN)
    h = mut*2/3
    
    A = 3*muss*(muts + gs*mua)/(mueff1**2 - muts**2)
    B = (-A*(1 + C*h*muts) - 3*C*h*gs*muss)/(1 + C*h*mueff1)
    
    exp1 = np.exp(-muts[:,:,np.newaxis]*z)
    exp2 = np.exp(-mueff1[:,:,np.newaxis]*z)
    phi = (1+A[:,:,np.newaxis])*exp1 + B[:,:,np.newaxis]*exp2
    return phi
    
def delta_p1_v(z, mua, mus, fx, n=1.4, g=0.8):
    """Fluence estimation with delta-P1 approximation, as seen in Seo thesis.
NOTE: only the AC component of fluence is derived in the thesis."""
    R = -0.13755*n**3 + 4.339*n**2 - 4.90366*n + 1.6896
    # gs = g/(g+1)
    muss = mus*(1-g**2)
    mut = mua + mus  # transport coefficient MxN
    muts = mua + muss  # transport coefficient* MxN
    mueff=np.sqrt(3*mua*mut)  # effective transport coefficient MxN
    mueff1 = np.sqrt(mueff**2 + (2*np.pi*fx[:,np.newaxis])**2)  # mueff, with fx (MxN)
    h = mut*2/3
    
    C = muss/(h*mueff1)
    zb = h*R
    
    exp1 = np.exp(-muts[:,:,np.newaxis]*z)
    exp2 = np.exp(-mueff1[:,:,np.newaxis]*z)
    exp3 = np.exp(-mueff1[:,:,np.newaxis]*(z+2*zb))
    
    phi = (C[:,:,np.newaxis]/(mueff1[:,:,np.newaxis] - muts) * (exp1 - exp2) +
           C[:,:,np.newaxis]/(mueff1[:,:,np.newaxis] + muts) * (exp1 - exp3))
    return phi

def diffusion_v(z, mua, mus, fx, n=1.4, g=0.8):
    musp = mus*(1-g)  # reduced scattering coefficient
    Reff = 0.493
    mut = mua + musp  # transport coefficient (1xN)
    A = (1-Reff)/(2*(1+Reff))
    mueff = np.sqrt(3*mua*mut)  # effective transport coefficient (1xN)
    mueff1 = np.sqrt(mueff**2 + (2*np.pi*fx[:,np.newaxis])**2)  # mueff, with fx (MxN)
    Cp_dc = 3*mut*musp/(mueff**2-mut**2)
    Cp_ac = 3*mut*musp/(mueff1**2-mut**2)
    Ch_dc = -Cp_dc*(3*A+1)/(mueff/mut + 3*A)
    Ch_ac = -Cp_ac*(3*A+1)/(mueff1/mut + 3*A)
    
    exp1 = np.exp(-mueff[:,:,np.newaxis]*z)
    exp2 = np.exp(-mut[:,:,np.newaxis]*z)
    exp3 = np.exp(-mueff1[:,:,np.newaxis]*z)
    
    phi_dc = Ch_dc[:,:,np.newaxis] * exp1 + Cp_dc[:,:,np.newaxis] * exp2
    phi_ac = Ch_ac[:,:,np.newaxis] * exp3 + Cp_ac[:,:,np.newaxis] * exp2
    
    phi = phi_dc + phi_ac
    
    return phi#, phi_dc, phi_ac

if __name__ == "__main__":
    Z = np.arange(0, 50, 0.01)
    fx = np.arange(0, 0.31, 0.05)
    
    cdflevels=[10, 25, 50, 75, 90]  # CDF level tables
    tablemuspmua=np.array([1, 1.6, 2, 3, 4, 5, 8, 10, 16, 20, 30,
                  50, 80, 100, 160, 250, 300, 1000])  # Monte Carlo mus/mua 
    tablefxs=np.array([0, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07,
              0.075, 0.08, 0.09, 0.1, 0.12, 0.125, 0.14, 0.15, 0.16,
              0.175, 0.18, 0.2, 0.25, 0.3, 0.5, 0.7])  # Monte Carlo fx
        
    mua = np.array([[0.2]])
    muspmua = 75
    musp = muspmua * mua[0][0]
    mus = np.array([[musp*0.2]])
    lstar = 1/(mua+mus)
    
    idx_fx = np.array([0,6,12,16,20,21,22])
    idx_muspmua = np.where(tablemuspmua == muspmua)[0]
    
    sda = fluence(Z, mua, mus, fx, diffusion)
    sda_v = fluence(Z, mua, mus, fx, diffusion_v)
    dp1 = fluence(Z, mua, mus, fx, delta_p1)
    dp1_v= fluence(Z, mua, mus, fx, delta_p1_v)
    
    # TODO: calculate cumulative sum / depths of 4 analytical models
    
    depths_MC = np.zeros((len(cdflevels), len(fx)))
    depths_sda = np.zeros((len(cdflevels), len(fx)))
    depths_sda_v = np.zeros((len(cdflevels), len(fx)))
    depths_dp1 = np.zeros((len(cdflevels), len(fx)))
    depths_dp1_v = np.zeros((len(cdflevels), len(fx)))
    
    for _i, cdf in enumerate(cdflevels):
        # load each CDF level table
        table = np.genfromtxt('{}/cdflevel{}table.csv'.format(dep_path._path[0], cdf), delimiter=',')
        f = interp2d(tablemuspmua, tablefxs/lstar, table.T*lstar, kind='linear')
        depths_MC[_i,:] = f(muspmua, fx).T
        
        for _j, freq in enumerate(fx):
            # diffusion
            idx = np.where(sda.cumsum[_j,:] >= sda.phitot[_j]*cdf/100)[0][0]
            depths_sda[_i, _j] = Z[idx]
            # diffusion_vasen
            idx = np.where(sda_v.cumsum[_j,:] >= sda_v.phitot[_j]*cdf/100)[0][0]
            depths_sda_v[_i, _j] = Z[idx]
            # delta-p1
            idx = np.where(dp1.cumsum[_j,:] >= dp1.phitot[_j]*cdf/100)[0][0]
            depths_dp1[_i, _j] = Z[idx]
            # delta-p1_vasen
            idx = np.where(dp1_v.cumsum[_j,:] >= dp1_v.phitot[_j]*cdf/100)[0][0]
            depths_dp1_v[_i, _j] = Z[idx]
        
    _i = 0
    freq = fx[_i]
    plt.figure(1, figsize=(8,5.5))
    plt.plot(depths_MC[:,_i], cdflevels, '--sk', markersize=8, label='Monte Carlo')  # reference: MC
    plt.plot(depths_sda[:,_i], cdflevels, '--^', markersize=8, label='SDA')  # diffusion
    plt.plot(depths_sda_v[:,_i], cdflevels, '--v', markersize=8, label='SDA_Vasen')  # diffusion_vasen
    plt.plot(depths_dp1[:,_i], cdflevels, '--*', markersize=8, label=r'$\delta$-P1')  # delta-p1
    plt.plot(depths_dp1_v[:,_i], cdflevels, '--p', markersize=8, label=r'$\delta$-P1_Vasen')  # delta-p1_vasen
    
    plt.grid(True, linestyle=':')
    plt.legend(loc='lower right')
    plt.title(r"$\mu_s'$/$\mu_a$ = {}".format(muspmua))
    plt.xlabel('depth (mm)')
    plt.ylabel('% of photons')
    plt.tight_layout()
