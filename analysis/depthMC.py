# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 09:10:58 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Description: SFD_depth_calculator determines spatial frequency domain (SFD)
  penetration depth estimates using a scaled lookup table.  The table is
  populated with results from Monte Carlo simulations at incremental
  mus'/mua and spatial frequency (fx) values. 

Inputs:
  mus  = reduced scattering coefficient [/mm] of the tissue of interest
  mua  = absorption coefficient [/mm] of the tissue of interest
  fxs  = spatial frequencies of interest

Outputs:
  depths = 3D matrix of penetration depth estimates, size [5, len(fxs), len(mua)]
           - the first axis is X (definition below) with values [10 25 50 75 90]
           - the second axis are the input spatial frequencies
           - the third axis are the input wavelengths (if mua, mus are arrays)

The penetration depth estimates are determined by calculating the maximum
visitation depths of detected photons.  This forms a function P_zmax that
when integrated with respect to z to a depth d produces 
  P_zmax(z<=d) = int_0^d P_zmax(z)dz.
If d=positive infinity, then the integration results in total diffuse 
reflectance, Rd.  Division of P_zmax by Rd
  X = P_zmax(z<=d) / Rd
provides the fraction (X) of the detected light that visited tissue depths d
or less. Details are in accompanying manuscript.

Author: Carole Hayakawa
Date: 1/17/18
"""
import numpy as np
from scipy.interpolate import interp2d
import warnings
from sfdi.analysis.depthCalculator.data import __path__ as dep_path

def depthMC(mua, mus, fx):
    # Tables values
    cdflevels=[10, 25, 50, 75, 90]  # CDF level tables
    tablemuspmua=[1, 1.6, 2, 3, 4, 5, 8, 10, 16, 20, 30,
                  50, 80, 100, 160, 250, 300, 1000]  # Monte Carlo mus/mua 
    tablefxs=[0, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07,
              0.075, 0.08, 0.09, 0.1, 0.12, 0.125, 0.14, 0.15, 0.16,
              0.175, 0.18, 0.2, 0.25, 0.3, 0.5, 0.7]  # Monte Carlo fx
    # assure that the inputs are arrays
    if hasattr(mua, "__len__") and (not isinstance(mua, str)):
        mua = np.array(mua, dtype=float)  # convert to numpy array
    else:
        mua = np.array([mua], dtype=float)
    if hasattr(mus, "__len__") and (not isinstance(mus, str)):
        mus = np.array(mus, dtype=float)  # convert to numpy array
    else:
        mus = np.array([mus], dtype=float)
    if hasattr(fx, "__len__") and (not isinstance(fx, str)):
        fx = np.array(fx, dtype=float)  # convert to numpy array
    else:
        fx = np.array([fx], dtype=float)
    
    # Initialize return variable 'depths' of dimensions [cdflevel, fxs, wv]
    # import pdb; pdb.set_trace()
    depths = np.zeros((len(cdflevels), len(fx), len(mua)), dtype=float)
    muspmua = mus/mua  # ratio
    lstar = 1/(mua + mus)  # path length  
    
    # check variables nd issue warnings
    if np.any(mua < 0) or np.any(mus < 0):
        warnings.warn('The input mua and musp values need to be > 0 -> NaN results')
    if any(f < 0 for f in fx):
        warnings.warn('The input fxs need to be > 0 -> NaN results')
    if np.max(fx) > np.amax(tablefxs / lstar[:,np.newaxis]):
        warnings.warn('The input fxs need to be < 0.7/l* -> NaN results')
    if any(muspmua) < np.amin(tablemuspmua):
        warnings.warn('The input musp/mua needs to be > 1 -> NaN results')
    if any(muspmua) > np.amax(tablemuspmua):
        warnings.warn('The input musp/mua needs to be < 1000 -> NaN results')

    for _i, cdf in enumerate(cdflevels):
        for _j in range(len(mua)):
            # load each CDF level table
            table = np.genfromtxt('{}/cdflevel{}table.csv'.format(dep_path._path[0], cdf), delimiter=',')
            f = interp2d(tablemuspmua, tablefxs/lstar[_j], table.T * lstar[_j], kind='linear')
            depths[_i, :, _j] = f(muspmua[_j], fx)
    return depths

 