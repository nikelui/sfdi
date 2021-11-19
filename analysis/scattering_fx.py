# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 10:49:15 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Script to fit mus' to a power law of the kind A * lambda^(-b), select a ROI and
compare the variation at different fx
"""
import os, sys, re
from datetime import datetime
import pickle
# import json
import numpy as np
from scipy.io import loadmat  # new standard: work with Matlab files for compatibility
from scipy.optimize import curve_fit

from sfdi.common.getPath import getPath
from sfdi.analysis.dataDict import dataDict  # moved class to other file
from sfdi.common.readParams import readParams

# support functions
def save_obj(obj, name, path):
    """Utility function to save python objects using pickle module"""
    with open('{}/obj/{}.pkl'.format(path, name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, path):
    """Utility function to load python objects using pickle module"""
    with open('{}/obj/{}.pkl'.format(path, name), 'rb') as f:
        return pickle.load(f)

def fit_fun(lamb, a, b):
    """Power law function to fit data to"""
    return a * np.power(lamb, -b)

data_path = getPath('Select data path')
par = readParams('{}/processing_parameters.ini'.format(data_path))  # optional
if 'wv' in par.keys():
    wv = par['wv']
else:
    wv = np.array([458, 520, 536, 556, 626])  # wavelengts (nm). Import from params?
regex = re.compile('.*f\d*\.mat')  # regular expression for optical properties
regex2 = re.compile('.*calR.mat')  # regular expression for calibrated reflectance

# If the dataset has already been processed, load it
if '-load' in sys.argv and os.path.exists('{}/obj/dataset.pkl'.format(data_path)):
    data = load_obj('dataset', data_path)
    data.par = par
# If you need to process / modify it. NOTE: the old set will be overwritten
else:
    files = [x for x in os.listdir(data_path) if re.match(regex, x)]
    datasets = set(x.split('_')[-3] for x in files)  # sets have unique values
    data = dataDict(parameters=par)
    # load the data into a custom dictionary
    start = datetime.now()  # calculate execution time
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
    end = datetime.now()
    print('Elapsed time: {}'.format(str(end-start)))
    # save fitted dataset to file for easier access
    if not os.path.isdir('{}/obj'.format(data_path)):
        os.makedirs('{}/obj'.format(data_path))
    save_obj(data, 'dataset', data_path)

# Post- processing
# data.mask_on()  # mask outliers
# data.plot_cal('AlO05ml', data_path)
# data.plot_mus('AlO05ml')
# ret = data.singleROI('TiObase', norm=-1, fit='single', f=[0,1,2,3,4])
# ret = data.singleROI('AlO1ml', norm=-1, fit='single', f=[0,1,2,3,4])
