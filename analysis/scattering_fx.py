# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 10:49:15 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Script to fit mus' to a power law of the kind A * lambda^(-b), select a ROI and
compare the variation at different fx
"""
import os, sys, re
import datetime
import pickle
import json
import numpy as np
from scipy.io import loadmat  # new standard: work with Matlab files for compatibility
from scipy.optimize import curve_fit
from sfdi.common.sfdi.getPath import getPath

from dataDict import dataDict  # moved class to other file

# support functions
def save_obj(obj, name, path):
    """Utility function to save python objects using pickle module"""
    with open(path + '/obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, path):
    """Utility function to load python objects using pickle module"""
    with open(path + '/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    

def fit_fun(lamb, a, b):
    """Power law function to fit data to"""
    return a * np.power(lamb, -b)


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

#%%

wv = np.array([458, 520, 536, 556, 626])  # wavelengts (nm). Import from params?
regex = re.compile('.*f\d\.mat')  # regular expression for optical properties
regex2 = re.compile('.*calR.mat')  # regular expression for calibrated reflectance
data_path = getPath('Select data path')
par = read_param('{}/README.txt'.format(data_path))  # optional

# If the dataset has already been processed, load it
if '-load' in sys.argv and os.path.exists('{}/obj/dataset.pkl'.format(data_path)):
    data = load_obj('dataset', data_path)
    data.par = par
# If you need to process / modify it. NOTE: the old set will be overwritten
else:
    files = [x for x in os.listdir(data_path) if re.match(regex, x)]
    datasets = set(x.split('_')[1] for x in files)  # sets have unique values
    data = dataDict(parameters=par)
    # load the data into a custom dictionary
    start = datetime.datetime.now()  # calculate execution time
    for _d, dataset in enumerate(datasets, start=1):
        data[dataset] = {}  # need to initialize it
        temp = [x for x in files if dataset in x]   # get filenames
        freqs = [x.split('_')[-1][:-4] for x in temp]  # get frequency range
        for file,fx in zip(temp, freqs):
            data[dataset][fx] = loadmat('{}/{}'.format(data_path, file))
            # here fit the data
            print('Fitting dataset {}_{}...[{} of {}]'.format(dataset, fx, _d, len(datasets)))
            op_map = data[dataset][fx]['op_fit_maps']  # for convenience
            p_map = np.zeros((op_map.shape[0], op_map.shape[1], 2), dtype=float)  #initialize
            for _i in range(op_map.shape[0]):
                for _j in range(op_map.shape[1]):
                    try:
                        temp, _ = curve_fit(fit_fun, wv, op_map[_i,_j,:,1], p0=[10, 1],
                                            method='trf', loss='soft_l1', max_nfev=2000)
                    except RuntimeError:
                        continue
                    p_map[_i, _j, :] = temp
            data[dataset][fx]['par_map'] = p_map
    end = datetime.datetime.now()
    print('Elapsed time: {}'.format(str(end-start).split('.')[0]))
    # save fitted dataset to file for easier access
    if not os.path.isdir('{}/obj'.format(data_path)):
        os.makedirs('{}/obj'.format(data_path))
    save_obj(data, 'dataset', data_path)

# Post- processing
data.mask_on()  # mask outliers
#data.plot_cal('K1', data_path)
