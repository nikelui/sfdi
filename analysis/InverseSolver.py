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
import time
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
##############################################
## Some model parameters, change as needed  ##
##############################################

keys = [x for x in data.keys() if not x.startswith('_')]
keys.sort()  # just in case they are not in order

# F = par['fx']
F = [0, 0.03333333, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
FX = np.array([np.mean(F[x:x+4]) for x in range(len(F)-3)])  # spatial frequency (average)

Z = np.arange(0,10,0.001)  # depth, micrometer resolution
Z = Z [np.newaxis,:]  # reshape, dimensions: (1 x Z)

WV = np.array([458, 520, 536, 556, 626])  # wavelengths

phi_mod = mod_dP1 # model to calculate fluence

##############################################

# mus array: [D x FX x WV]
mus_meas = np.zeros((len(keys), data[keys[0]].shape[0], data[keys[0]].shape[1]), dtype=float)
mua_meas = np.zeros(mus_meas.shape, dtype=float)
for _k, key in enumerate(keys):
    mus_meas[_k,:,:] = data[key][:,:,1]
    mua_meas[_k,:,:] = data[key][:,:,0]  # need it to calculate fluence

# Phantom thickness (for evaluating performance)
# d_real = np.array([0.1295, 0.2685, 0.49, 0.675, 1.149])
# d_std = np.array([0.0035, 0.0136, 0.0216, 0.0401, 0.0655])

# Top and bottom layer scattering (for evaluating performance)
# mus_top = data['TiObaseTop'][0,:,1]
# mus_bot = data['AlObaseTop'][0,:,1]

opt_ret = [[]]  # save all the results of optimization, for debug. Should be 2D [D x WW] array
# bound = Bounds(lb=[0,0,0], ub=[10,20,20])  # Add some physical limits to problem
bound = ([0,0,0], [20,20,20])  # Add some physical limits to problem

## results array, for easy copy-paste to excel
ret_d = np.zeros([6,5])
ret_must = np.zeros([6,5])
ret_musb = np.zeros([6,5])


# initialize N random starting points
N = 1000
x0_array = list(zip(np.random.uniform(low=0, high=0.5, size=(N)),   # thickness
                    np.random.uniform(low=0, high=5.0, size=(N)),   # mus_top
                    np.random.uniform(low=0, high=5.0, size=(N))))  # mus_bot

loss_fun = [[{}] * len(WV)] * len(keys)

start = time.time()

for _d, d in enumerate(keys):  # loop over thickness / datasets
    if _d > 0:
        opt_ret.append([])
    for _w, w in enumerate(WV):  # loop over wavelengths
        best_ret = None  # To save best fit
        for _x, x0 in enumerate(x0_array):
            if _x % 100 == 0:
                print("Dataset {} of {} - Initial guess #{}".format(_d+1, len(keys), _x))
            x0 = np.array(x0)  # initial guess
            temp = least_squares(target_fun, x0, jac=jacob, bounds=bound, method='trf',
                                 x_scale=np.array([.1,1,1]), loss='linear', max_nfev=1e3, verbose=0,
                                 args=(mua_meas[_d,:,_w:_w+1], mus_meas[_d,:,_w:_w+1], Z, FX),
                                 kwargs={'model':phi_mod} )
            # DEBUG
            loss_fun[_d][_w][tuple(x0)] = temp.cost
            # Check if solution has improved since previous initial guess
            if best_ret is None:  # First iteration
                best_ret = temp
                opt_ret[_d].append(temp)
                ret_d[_d,_w] = temp['x'][0]
                ret_must[_d,_w] = temp['x'][1]
                ret_musb[_d,_w] = temp['x'][2]
            elif temp.cost < best_ret.cost:  
                best_ret = temp
                opt_ret[_d][_w] = temp
                ret_d[_d,_w] = temp['x'][0]
                ret_must[_d,_w] = temp['x'][1]
                ret_musb[_d,_w] = temp['x'][2]

end = time.time()
print("Elapsed time: {:02d}:{:05.2f}".format(int((end-start)//60), (end-start) % 60))

#%% Plots
from matplotlib import pyplot as plt
from matplotlib import cm
from cycler import cycler
import addcopyfighandler

W = 0  # wavelength index
D = 0  # thickness index

# cyc = (cycler(color=['tab:blue','tab:orange','tab:green','tab:red']))
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']

# f_ax = np.arange(0,0.5,0.01)
mus_fitted = np.zeros((5,8,5))
mus_model = np.zeros((5,8,5))
for _d in range(5):
    for _w in range(5):
        mus_model[_d,:,_w] = model_fun((d_real[_d], mus_top[_w], mus_bot[_w]),
                                        mua_meas[_d,:,_w:_w+1], mus_meas[_d,:,_w:_w+1],
                                        Z, FX, model=phi_mod).squeeze()
        mus_fitted[_d,:,_w] = model_fun((ret_d[_d, _w], ret_must[_d, _w], ret_musb[_d, _w]),
                                        mua_meas[_d,:,_w:_w+1], mus_meas[_d,:,_w:_w+1],
                                        Z, FX, model=phi_mod).squeeze()

fig, ax = plt.subplots(1,1, figsize=(7,4), num=1)
for _d in range(len(d_real)):
    ax.plot(FX, mus_meas[_d,:,W].T, '*', color=colors[_d])
    ax.plot(FX, mus_model[_d,:,W].T, '-', color=colors[_d])
    ax.plot(FX, mus_fitted[_d,:,W].T, '--', color=colors[_d])
ax.set_xlabel('fx (mm$^{-1}$)', fontsize=12)
ax.set_ylabel(r"$\mu'_s$ (mm$^{-1}$)", fontsize=12)
ax.set_title('mod-$\delta$-P1 (@{} nm)'.format(WV[W]), fontsize=15)
ax.grid(True, linestyle=':')
plt.tight_layout()

colors=cm.get_cmap('Blues_r', 11)

fig, ax = plt.subplots(1,1, figsize=(7,4), num=2)
# for _f in range(len(FX)):
    # ax.plot(d_real, mus_meas[:,_f,W], '*', color=colors(_f+2))
    # ax.plot(d_real, mus_model[:,_f,W], '-', color=colors(_f+2))
    # ax.plot(d_real, mus_fitted[:,_f,W], '--', color=colors(_f+2))
ax.plot(d_real, mus_meas[:,:,W], '*')
ax.set_prop_cycle(None)
ax.plot(d_real, mus_model[:,:,W], '-')
ax.set_prop_cycle(None)
ax.plot(d_real, mus_fitted[:,:,W], '--')
ax.set_xlabel('d (mm)', fontsize=12)
ax.set_ylabel(r"$\mu'_s$ (mm$^{-1}$)", fontsize=12)
ax.set_title('mod-$\delta$-P1 (@{} nm)'.format(WV[W]), fontsize=15)
ax.grid(True, linestyle=':')
plt.tight_layout()

#%% Test for new fit

al = 3 * d_real.reshape((1,-1))
bet = 3.5 * FX.reshape((-1,1))

mus_b = mus_bot[0]
mus_t = mus_top[0]

mus_m = mus_t - (mus_t - mus_b)*np.exp(-al * bet)

plt.figure(figsize=(7,4), num=3)
plt.plot(FX, mus_m)
plt.gca().set_prop_cycle(None)
plt.plot(FX, mus_meas[:,:,0].T, '*')
plt.grid(True, linestyle=':')
plt.ylim([0.9, 3.5])
plt.tight_layout()


plt.figure(figsize=(7,4), num=4)
plt.plot(d_real, mus_m.T)
plt.gca().set_prop_cycle(None)
plt.plot(d_real, mus_meas[:,:,0], '*')
plt.grid(True, linestyle=':')
plt.ylim([0.9, 3.5])
plt.tight_layout()
