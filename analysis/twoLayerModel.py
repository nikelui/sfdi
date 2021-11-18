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
____a_____ | |- (la, ba)
           |
           |
____b____  |--- (delta, bb)

x = [ba, bb, la]
bm = measured b coefficient
"""
    return ((x[0] * x[2]) + x[1]*(delta - x[2]))/delta - bm


def new_two_layer_fun(x, delta, mus, wv):
    """Partial volumes equation for a two layer model
__________
____a_____ | |- (la, mus_a)
           |
           |
____b____  |--- (delta, mus_b)
    
    model: mus = (a1*lambda^-b1 * la + 2*lambda^-b2 * (delta-la))/delta

    Parameters
    ----------
    x : FLOAT array
        array of unknowns  ->  [a1, b1, a2, b2, la]
        - a1, b1 -> scattering parameters of layer a [mm^-1]
        - a2, b2 -> scattering parameters of layer b [mm^-1]
        - la -> thickness of layer a [mm]
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
    return np.sum(((x[0]*wv[:,np.newaxis]**(-x[1]) * x[4]) + 
                   (x[2]*wv[:,np.newaxis]**(-x[3]) * (delta[np.newaxis,:] - x[4])) / 
                       delta - mus), axis=0)


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
ret = data.singleROI('AlO3ml', fit='single', I=3e3, norm=None)

d = np.mean(ret['depths'], axis=1)[:-1]  # delta/2
d2 = np.mean(ret['depth_phi'], axis=1)[:-1]  # 1/e * phi^2
d3 = np.mean(ret['depth_MC'], axis=1)[:-1]  # calculated via Monte Carlo model
mus = ret['op_ave'][:-1,:,1]  # average measured scattering coefficient
bm = ret['par_ave'][:-1, 1]  # average measured scattering slope

# Old model
opt = least_squares(two_layer_fun, x0=[1, 1, 0.1], kwargs={'delta': d, 'bm': bm},
                    bounds=[0, np.inf], method='trf')

opt2 = least_squares(two_layer_fun, x0=[1, 1, 0.1], kwargs={'delta': d2, 'bm': bm},
                    bounds=[0, np.inf], method='trf')

opt3 = least_squares(two_layer_fun, x0=[1, 1, 0.1], kwargs={'delta': d3, 'bm': bm},
                    bounds=[0, np.inf], method='trf')

#%%
# New model
opt = least_squares(new_two_layer_fun, x0=[100,1,100,1,0.1],
                    kwargs={'delta': d, 'mus': mus.T, 'wv':np.array(data.par['wv'])},
                    bounds=[0, np.inf], method='trf')

opt2 = least_squares(new_two_layer_fun, x0=[100,1,100,1,0.1],
                    kwargs={'delta': d2, 'mus': mus.T, 'wv':np.array(data.par['wv'])},
                    bounds=[0, np.inf], method='trf')

opt3 = least_squares(new_two_layer_fun, x0=[100,1,100,1,0.1],
                    kwargs={'delta': d3, 'mus': mus.T, 'wv':np.array(data.par['wv'])},
                    bounds=[0, np.inf], method='trf')