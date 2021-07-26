# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 16:31:14 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import numpy as np

def diffuseReflectance(mua, mus, n, fx):
    """
    Function to calculate diffuse reflectance at target spatial frequencies.

    Parameters
    ----------
    mua : float array
        Tissue absorption coefficient in mm^-1
    mus : float array
        Tissue reduced scattering coefficient in mm^-1
    n : float array
        Tissue refractive index.
    fx : float array
        Target spatial frequencies for output.

    Returns
    -------
    Rd : float array
        Diffuse reflectance Rd(fx)

    """
    # TODO: debug for multiple wavelengths # DONE
    # cast inputs to numpy arrays
    (mua,mus,n,fx) = (np.array(x)[:,np.newaxis] for x in (mua,mus,n,fx))
    kx = fx*2*np.pi  # convert frequency to wavenumber, to not forget later
    mut = mua + mus  # transport coefficient
    mueff = np.sqrt(3*mua*mut)  # effective transport coefficient
    mueff1 = np.sqrt(mueff**2 + kx.T**2)  # reduced effective transport coefficient
    a = mus / mut  # albedo
    Reff = 0.0636*n + 0.668 + 0.71/n - 1.44/n**2  # Effective reflection coefficient
    A = (1 - Reff)/(2*(1 + Reff))  # Proportionality coefficient
    Rd = 3*A*a / ((mueff1/mut + 1) * (mueff1/mut + 3*A))
    
    return Rd
    