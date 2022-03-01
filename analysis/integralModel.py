# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:52:02 2022

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt

def alpha_diff(d, mua, mus, fx, g=0.8, n=1.4):
    """
    Function to calculate the expected weigth (alpha) in a 2 layer model using
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
    mua : FLOAT array
        Absorption coefficient, dependent on wavelength.
    mus : FLOAT array
        Scattering coefficient, dependent on wavelength.
    fx :FLOAT array
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
    import pdb; pdb.set_trace()

    musp = mus*(1-g)  # reduced scattering coefficient
    Reff = 0.0636*n + 0.668 + 0.71/n - 1.44/n**2  # effective reflection coefficient
    mut = mua + musp  # transport coefficient (1xN)
    mueff = np.sqrt(3*mua*mut)  # effective transport coefficient (1xN)
    mueff1 = np.sqrt(mueff[np.newaxis,:]**2 + (2*np.pi*fx[:,np.newaxis])**2)  # mueff, with fx (MxN)
    #TODO: debug the equation (maybe calculate phi?). it return negative values for alpha    
    A = (3*musp/mut)/(mueff1**2/mut**2 - 1)  # MxN
    R = (1-Reff)/(2*(1+Reff))
    B = -3*(musp/mut)*(1+3*R) / ((mueff1**2/mut**2 - 1)*(mueff1/mut + 3*R))  # MxN
    
    alpha = (A/mut * np.exp(-mut*d) + B/mueff1 * np.exp(-mueff1*d)) / (A/mut + B/mueff1) - 1
    
    return alpha
    