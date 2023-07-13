# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:36:37 2023

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Inverse solver
Script to get multi-fx scattering data on 2 layer tissue and estimate tissue thickness /
layer specific scattering
"""
from sfdi.common.getFile import getFile
from sfdi.common import models  # fluence models
SDA = models.phi_diff
dP1 = models.phi_dP1
mod_dP1 = models.phi_deltaP1

import numpy as np
from scipy.io import loadmat, savemat
from scipy.optimize import Bounds
from scipy.optimize import least_squares

def model_fun(x0, mua_meas, mus_meas, z, fx, model=SDA):
    """Function to calculate difference between the 2-layer model and the measurements
Input:
    - x0 = [d, mus_top, mus_bot]
    - mua_meas, mus_meas: arrays with the multi-layered optical properties measurements
                          shape: (N x M)
    - model: function to calculate light fluence
            model input:
                - z: depth array (1 x Z)
                - mua: absorption (N x M)
                - mus: scattering (N x M)
                - fx: average fz (N x 1)
            model return:
                - phi: fluence array (N x M x Z)
"""
    d, mus_top, mus_bot = x0  # unpack
    phi = model(z, mua_meas, mus_meas/0.2, fx)
    a = alpha(phi, z, d)
    
    mus_model = a * mus_top + (1-a) * mus_bot
    return mus_model

def target_fun(x0, mua_meas, mus_meas, z, fx, model=SDA):
    """Calculate residuals, for minimization using least squares"""    
    mus_model = model_fun(x0, mua_meas, mus_meas, z, fx, model)
    # diff = np.sqrt((np.sum((mus_model - mus_meas)**2, axis=0) / mus_meas.shape[0]))
    diff = np.squeeze(mus_model - mus_meas)
    return diff
    
def alpha(phi, z, d):
    """Integral function to calculate the relative contribution of scattering from the
top layer in a 2-layer model.

alpha = (int_0^d phi**2(z) dz) / (int_0^inf phi**2(z) dz)

    N: number of frequencies
    M: number of wavelengths
    Z: number of depths

NOTE: this is not a real integral function, it does a discrete sum over the Z array.
      The smaller the dz steps, the smaller the discretization error will be.

Input:
    - phi [N x M x Z]: light fluence
    - z [1 x Z]: depth axis
    - d [float]: thickness of top layer
"""
    dz = z[0,1] - z[0,0]  # discrete depth step
    # Find index to "integrate" up to d
    idx = np.where(z >= d)[1][0]
    alpha = np.sum(phi[:,:,:idx]**2 * dz, axis=-1)/np.sum(phi**2 * dz, axis=-1)
    return alpha

def jacob(x0, mua_meas, mus_meas, z, fx, model=SDA):
    """Analytically compute jacobian matrix to be used in least squares
NOTE: it needs the same input as the target function
Input:
    - x0 = [d, mus_t, mus_b]  (1 x 3 float)
    - phi (N x M x Z float)
    - z (1 x Z float)
"""
    d, mus_t, mus_b = x0  # unpack
    dz = z[0,1] - z[0,0]  # discrete depth step
    # Find index of d in z axis
    idx = np.where(z >= d)[1][0]
    
    phi = model(z, mua_meas, mus_meas/0.2, fx)
    phi_d = phi[:,:,idx]  # phi(d) [N x M x 1]
    phi_sum = np.sum(phi**2 * dz, axis=-1)  # integral of phi**2 [N x M]
    
    a = alpha(phi, z, d)  # [N x M x Z]
    # import pdb; pdb.set_trace()  # DEBUG
    J = np.zeros((phi.shape[0], x0.size))  # initialize
    J[:,0] = np.squeeze(1/phi_sum * phi_d**2)
    J[:,1] = np.squeeze(a)
    J[:,2] = np.squeeze(1-a)
    
    return J
#%%
# Scattering measurements
data_path = getFile('Select data path')
data = loadmat('{}'.format(data_path))

#%%
# Some lists of parameters
keys = [x for x in data.keys() if 'ml' in x]
keys.sort()  # just in case they are not in order

F = np.arange(0,0.55,0.05)
FX = np.array([np.mean(F[x:x+4]) for x in range(8)])  # spatial frequency (average)

Z = np.arange(0,10,0.001)  # depth, micrometer resolution
Z = Z [np.newaxis,:]  # reshape, dimensions: (1 x Z)

WV = np.array([458, 520, 536, 556, 626])  # wavelengths

phi_mod = mod_dP1  # model to calculate fluence

# mus array: [D x FX x WV]
mus_meas = np.zeros((len(keys), data[keys[0]].shape[0], data[keys[0]].shape[1]), dtype=float)
mua_meas = np.zeros(mus_meas.shape, dtype=float)
for _k, key in enumerate(keys):
    mus_meas[_k,:,:] = data[key][:,:,1]
    mua_meas[_k,:,:] = data[key][:,:,0]  # need it to calculate fluence

# Phantom thickness (for evaluating performance)
d_real = np.array([0.1295, 0.2685, 0.49, 0.675, 1.149])
d_std = np.array([0.0035, 0.0136, 0.0216, 0.0401, 0.0655])

# Top and bottom layer scattering (for evaluating performance)
mus_top = data['TiObaseTop'][0,:,1]
mus_bot = data['AlObaseTop'][0,:,1]

opt_ret = [[]]  # save all the results of optimization, for debug. Should be 2D [D x WW] array
# bound = Bounds(lb=[0,0,0], ub=[10,20,20])  # Add some physical limits to problem
bound = ([0,0,0], [10,20,20])  # Add some physical limits to problem

# results array, for easy copy-paste to excel
ret_d = np.zeros([5,5])
ret_must = np.zeros([5,5])
ret_musb = np.zeros([5,5])

for _d, d in enumerate(d_real):  # loop over thickness
    if _d > 0:
        opt_ret.append([])
    for _w, w in enumerate(WV):  # loop over wavelengths
        x0 = np.array([0.1, 1, 1])  # initial guess
        temp = least_squares(target_fun, x0, jac=jacob, bounds=bound, method='trf',
                             x_scale=np.array([0.1,1,1]), loss='linear', max_nfev=1e3, verbose=1,
                             args=(mua_meas[_d,:,_w:_w+1], mus_meas[_d,:,_w:_w+1], Z, FX),
                             kwargs={'model':phi_mod} )
        opt_ret[_d].append(temp)
        ret_d[_d,_w] = temp['x'][0]
        ret_must[_d,_w] = temp['x'][1]
        ret_musb[_d,_w] = temp['x'][2]

#%% Plots
from matplotlib import pyplot as plt
W = 0  # wavelength index
D = 0  # thickness index

# f_ax = np.arange(0,0.5,0.01)
mus_mod = model_fun((ret_d[D, W], ret_must[D, W], ret_musb[D, W]),
                    mua_meas[D,:,:], mus_meas[D,:,:], Z, FX)