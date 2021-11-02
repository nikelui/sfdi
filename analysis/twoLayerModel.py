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
from sfdi.analysis.dataDict import dataDict

def load_obj(name, path):
    """Utility function to load python objects using pickle module"""
    with open('{}/obj/{}.pkl'.format(path, name), 'rb') as f:
        return pickle.load(f)
    
def two_layer_fun(x, li, bm):
    """Partial volumes equation for a two layer model
__________
____a_____ | |- (la, ba)
           |
           |
____b____  |--- (li, bb)

x = [ba, bb, la]
bm = measured b coefficient
"""
    return ((x[0] * x[2]) + x[1]*(li - x[2])) - bm


def new_two_layer_fun(x, delta, mus):
    """Partial volumes equation for a two layer model
__________
____a_____ | |- (la, mus_a)
           |
           |
____b____  |--- (delta, mus_b)
    
    model: mus = (mus_a*la + mus_b*lb)/delta

    Parameters
    ----------
    x : FLOAT array
        array of unknowns  ->  [mus_a, mus_b, la]
        - mus_a -> scattering coefficient of layer a [mm^-1]
        - mus_b -> scattering coefficient of layer b [mm^-1]
        - la -> thickness of layer a [mm]
    delta : FLOAT
        Estimated penetration depth of light [mm]
    mus : FLOAT
        Measured scattering coefficient [mm^-1]

    Returns
    -------
    Difference between mus measurement and 2-layer model (to be used in least square)
"""
    import pdb; pdb.set_trace()    
    return (x[0]*x[2] + x[1]*(delta - x[2]))/delta - mus


def read_param(fpath):
    params = {}
    with open(fpath, 'r') as file:
        for line in file.readlines():
            if line.startswith('#') or len(line.strip()) == 0:  # ignore comments and newlines
                pass
            else:
                key, item = (x.strip() for x in line.split('='))
                if item.startswith('['):
                    end = item.find(']')
                    item = json.loads(item[:end+1])
                params[key] = item
    return params

#%% Load pre-processed data
data_path = getPath('select data path')
# par = read_param('{}/README.txt'.format(data_path))  # optional
data = load_obj('dataset', data_path)
# data.par = par
ret = data.singleROI('SC2', fit='single', I=2e3, norm=-1)

#%% Least square fit
d = np.mean(ret['depths'], axis=1)  # delta/2
d2 = np.mean(ret['depth_phi'], axis=1)  # 1/e * phi^2
d3 = np.mean(ret['depth_MC'], axis=1)  # calculated via Monte Carlo model
mus = ret['op_ave'][:,:,1]  # average measured scattering coefficient

opt = least_squares(new_two_layer_fun, x0=[[1,1,1,1,1], [1,1,1,1,1], [0.1,0.1,0.1,0.1,0.1,]],
                    kwargs={'delta': d, 'mus': mus}, bounds=[0, np.inf], method='trf')


opt2 = least_squares(new_two_layer_fun, x0=[1, 1, 0.1], kwargs={'delta': d2[1:], 'mus': mus[1:]},
                    bounds=[0, np.inf], method='trf')
opt3 = least_squares(new_two_layer_fun, x0=[1, 1, 0.1], kwargs={'delta': d3[1:], 'mus': mus[1:]},
                     bounds=[0, np.inf], method='trf')