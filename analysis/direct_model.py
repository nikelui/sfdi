# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:06:53 2022

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Direct model - 2 layer model of scattering using fluence to estimate penetration depth
"""
import os, sys, re
from datetime import datetime
import pickle
import numpy as np
from scipy.io import loadmat  # new s tandard: work with Matlab files for compatibility
from scipy.optimize import curve_fit

from sfdi.common.getPath import getPath
from sfdi.analysis.dataDict import dataDict  # moved class to other file
from sfdi.common.readParams import readParams
from sfdi.common import models

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

# Load dataset
data_path = getPath('Select data path')
par = readParams('{}/processing_parameters.ini'.format(data_path))  # optional
if 'wv' in par.keys():
    wv = par['wv']
else:
    wv = np.array([458, 520, 536, 556, 626])  # wavelengts (nm). Import from params?
regex = re.compile('.*f\d*\.mat')  # regular expression for optical properties
regex2 = re.compile('.*calR.mat')  # regular expression for calibrated reflectance
regex3 = re.compile('SFDS.*\.mat')  # regular expression for SFDS data

# If the dataset has already been processed, load it
if os.path.exists('{}/obj/dataset.pkl'.format(data_path)):
    data = load_obj('dataset', data_path)
else:
    files = [x for x in os.listdir(data_path) if re.match(regex, x)]
    datasets = set(x.split('_')[-3] for x in files)  # sets have unique values
    sfds_path = [x for x in os.listdir(data_path) if re.match(regex3, x)]  # should be only one
    if sfds_path:  # need a check, because it might not exist
        sfds = loadmat('{}/{}'.format(data_path,sfds_path[0]))
        par['wv_sfds'] = np.squeeze(sfds['wv'])
    data = dataDict()
    data.par = par
    # load the SFDI data into a custom dictionary
    start = datetime.now()  # calculate execution time
    for _d, dataset in enumerate(datasets, start=1):
        data[dataset] = {}  # need to initialize it
        temp = [x for x in files if dataset in x]   # get filenames
        freqs = [x.split('_')[-1][:-4] for x in temp]  # get frequency range
        for file,fx in zip(temp, freqs):
            data[dataset][fx] = loadmat('{}/{}'.format(data_path, file))
            if sfds_path and dataset in sfds.keys():
                data[dataset][fx]['sfds'] = {}
                data[dataset][fx]['sfds']['op_fit'] = sfds[dataset][:,freqs.index(fx),:]
            # here fit the data
            print('Fitting dataset {}_{}...[{} of {}]'.format(dataset, fx, _d, len(datasets)))
            # SFDI data
            op_map = data[dataset][fx]['op_fit_maps']  # for convenience
            p_map = np.zeros((op_map.shape[0], op_map.shape[1], 2), dtype=float)  # initialize
            for _i in range(op_map.shape[0]):
                for _j in range(op_map.shape[1]):
                    try:
                        temp, _ = curve_fit(fit_fun, wv, op_map[_i,_j,:,1], p0=[10, 1],
                                            method='trf', loss='soft_l1', max_nfev=2000)
                    except RuntimeError:
                        continue
                    p_map[_i, _j, :] = temp
            data[dataset][fx]['par_map'] = p_map
            # SFDS data
            if sfds_path and dataset in sfds.keys():
                temp, _ = curve_fit(fit_fun, par['wv_sfds'], data[dataset][fx]['sfds']['op_fit'][:,1],
                                    p0=[10, 1], method='trf', loss='soft_l1', max_nfev=2000)
                data[dataset][fx]['sfds']['par'] = temp
            
    end = datetime.now()
    print('Elapsed time: {}'.format(str(end-start)))
    # save fitted dataset to file for easier access
    if not os.path.isdir('{}/obj'.format(data_path)):
        os.makedirs('{}/obj'.format(data_path))
    save_obj(data, 'dataset', data_path)
    

# %%
