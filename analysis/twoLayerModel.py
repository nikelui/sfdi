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
from sfdi.common.sfdi.getPath import getPath


def load_obj(name, path):
    """Utility function to load python objects using pickle module"""
    with open(path + '/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def two_layer_fun(x, li, bm):
    """Partial volumes equation for a two layer model
__________
a_________ | |- (la, ba)
           |
           |
b________  |--- (li, bb)

x = [ba, bb, la]
bm = measured b coefficient
"""
    return ((x[0] * x[2]) + x[1]*(li - x[2])) - bm

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
par = read_param('{}/README.txt'.format(data_path))  # optional
data = load_obj('dataset', data_path)
data.par = par
ret = data.singleROI('K1', fit='single')

#%% Least square fit
d = np.mean(ret['depths'], axis=1)  # delta/2
d2 = ret['depth_phi']**2  # 1/e * phi
bm = ret['par_ave'][:,1]
opt = least_squares(two_layer_fun, x0=[1, 1, 0.1], kwargs={'li': d2[1:], 'bm': bm[1:]},
                    bounds=[0, np.inf], method='trf')
