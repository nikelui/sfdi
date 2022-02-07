#%% Import and function definition
"""
Created on Thu Feb  4 13:10:56 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import pickle
import json
import numpy as np
from scipy.optimize import least_squares
from sfdi.common.getPath import getPath
from sfdi.common.readParams import readParams
from sfdi.analysis.dataDict import dataDict

def load_obj(name, path):
    """Utility function to load python objects using pickle module"""
    with open('{}/obj/{}.pkl'.format(path, name), 'rb') as f:
        return pickle.load(f)
    
def two_layer_fun(x, delta, bm):
    """Partial volumes equation for a two layer model
__________
____1_____ | |- (d, b1)
           |
           |
____2____  |--- (delta, b2)

x = [b1, b2, d]
bm = measured b coefficient
"""
    b1, b2, d = x  # unpack
    return ((b1 * d) + b2*(delta - d))/delta - bm

def two_layer_fun2(x, delta, bm):
    b1, b2, d = x  # unpack
    mask = d/delta >= 1
    ret = ((b1 * d) + b2*(delta - d))/delta - bm
    ret[mask] = b1 - bm[mask]  # correction
    return ret

def new_two_layer_fun(x, delta, mus, wv):
    """Partial volumes equation for a two layer model
__________
____1_____ | |- (la, mus_a)
           |
           |
____2____  |--- (delta, mus_b)
    
    model: mus = (a1*lambda^-b1 * la + 2*lambda^-b2 * (delta-la))/delta

    Parameters
    ----------
    x : FLOAT array
        array of unknowns  ->  [ln(a1), b1, ln(a2), b2, d]
        - a1, b1 -> scattering parameters of layer 1 [mm^-1]
        - a2, b2 -> scattering parameters of layer 2 [mm^-1]
        - d -> thickness of layer 1 [mm]
        * NEW: now the 'a' parameter is passed as a logarithm, since it changes
                much more than the other 2 parameters.
    delta : FLOAT
        Estimated penetration depth of light [mm]
    mus : FLOAT
        Measured scattering coefficient [mm^-1]
    wv : FLOAT
        Wavelength (need for fitting)

    Returns
    -------
    Square difference between mus measurement and 2-layer model (to be used in least square)
"""
    (a1, b1, a2, b2, d) = x  # unpack
    a1 = np.power(10, a1*5)  # exponentiate logarithm
    a2 = np.power(10, a2*5)
    return np.sum(((a1*wv[:,np.newaxis]**(-b1) * d) + 
                   (a2*wv[:,np.newaxis]**(-b2) * (delta[np.newaxis,:] - d))) / delta 
                    - mus, axis=0)

def new_two_layer_fun2(x, delta, mus, wv):
    (a1, b1, a2, b2, d) = x  # unpack
    a1 = np.power(10, a1*5)  # exponentiate logarithm
    a2 = np.power(10, a2*5)
    ret = np.zeros((len(delta), len(wv)), dtype=float)
    for _w, w in enumerate(wv):
        for _f in range(len(delta)):
            if d / delta[_f] < 1:
                ret[_f, _w] = (a1*w**(-b1) * d + a2*w**(-b2) * (delta[_f] - d)) / delta[_f] - mus[_w, _f]
            else:
                ret[_f, _w] = a1*w**(-b1) - mus[_w, _f]  # only first layer
    return np.sum(ret, axis=0)

# def read_param(fpath):
#     params = {}'
#     with open(fpath, 'r') as file:
#         for line in file.readlines():
#             if line.startswith('#') or len(line.strip()) == 0:  # ignore comments and newlines
#                 pass
#             else:
#                 key, item = (x.strip() for x in line.split('='))
#                 if item.startswith('['):
#                     end = item.find(']')
#                     item = json.loads(item[:end+1])
#                 params[key] = item
#     return params

#%% Load pre-processed data
data_path = getPath('select data path')
par = readParams('{}/processing_parameters.ini'.format(data_path))  # optional
data = load_obj('dataset', data_path)
data.par = par

#%% Least square fit
ret = data.singleROI('TiO15ml', fit='single', I=3e3, norm=None)

d = np.mean(ret['depths'], axis=1)[:]  # delta/2
d2 = np.mean(ret['depth_phi'], axis=1)[:]  # 1/e * phi^2
d3 = np.mean(ret['depth_MC'], axis=1)[:]  # calculated via Monte Carlo model
mus = ret['op_ave'][:,:,1]  # average measured scattering coefficient
bm = ret['par_ave'][:, 1]  # average measured scattering slope

# Old model
opt = least_squares(two_layer_fun2, x0=[1, 1, 0.1], kwargs={'delta': d, 'bm': bm},
                    bounds=([0, 0, 0],[4, 4, np.inf]), method='trf',
                    loss='soft_l1', max_nfev=1000)

opt2 = least_squares(two_layer_fun2, x0=[1, 1, 0.1], kwargs={'delta': d2, 'bm': bm},
                    bounds=([0, 0, 0],[4, 4, np.inf]), method='trf',
                    loss='soft_l1', max_nfev=1000)

opt3 = least_squares(two_layer_fun2, x0=[1, 1, 0.1], kwargs={'delta': d3, 'bm': bm},
                    bounds=([0, 0, 0],[4, 4, np.inf]), method='trf',
                    loss='soft_l1', max_nfev=1000)

#%%
# New model
opt = least_squares(new_two_layer_fun2, x0=[0.4,1,0.4,1,0.1],
                    kwargs={'delta': d, 'mus': mus.T, 'wv':np.array(data.par['wv'])},
                    bounds=([0, 0, 0, 0, 0],[6, 4, 6, 4, np.inf]), method='trf',
                    loss='soft_l1', max_nfev=1000)

opt2 = least_squares(new_two_layer_fun2, x0=[0.4,1,0.4,1,0.1],
                    kwargs={'delta': d2, 'mus': mus.T, 'wv':np.array(data.par['wv'])},
                    bounds=([0, 0, 0, 0, 0],[6, 4, 6, 4, np.inf]), method='trf',
                    loss='soft_l1', max_nfev=1000)

opt3 = least_squares(new_two_layer_fun2, x0=[0.4,1,0.4,1,0.1],
                    kwargs={'delta': d3, 'mus': mus.T, 'wv':np.array(data.par['wv'])},
                    bounds=([0, 0, 0, 0, 0],[6, 4, 6, 4, np.inf]), method='trf',
                    loss='soft_l1', max_nfev=1000)