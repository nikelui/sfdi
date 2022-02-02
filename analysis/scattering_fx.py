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
from scipy.io import loadmat  # new s tandard: work with Matlab files for compatibility
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
regex3 = re.compile('SFDS.*\.mat')  # regular expression for SFDS data

# If the dataset has already been processed, load it
if '-load' in sys.argv and os.path.exists('{}/obj/dataset.pkl'.format(data_path)):
    data = load_obj('dataset', data_path)
    data.par = par
# If you need to process / modify it. NOTE: the old set will be overwritten
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

# Post- processing
# data.mask_on()  # mask outliers
# data.plot_cal('AlO05ml', data_path)
# data.plot_mus('AlO05ml')
# ret = data.singleROI('TiObase', norm=-1, fit='single', f=[0,1,2,3,4])
ret = data.singleROI('TiO10ml', norm=None, fit='single', f=[0,1,2,3,4], I=3e3)

#%% plotting
from matplotlib import pyplot as plt
from sfdi.common.phantoms import __path__ as ph_path
TS2 = np.genfromtxt('{}/TS2.txt'.format(ph_path._path[0]))  # reference
plt.figure(figsize=(10,4))
labels = ['f0','f1','f2','f3','f4','f5','f6','f7']
for _j in range(ret['op_ave'].shape[0]-3):
    plt.subplot(1,2,1)
    plt.errorbar(wv, ret['op_ave'][_j,:,0], yerr=ret['op_std'][_j,:,0], fmt='s', capsize=5,
                 linestyle='solid', linewidth=2, label=labels[_j])
    plt.grid(True, linestyle=':')

    plt.subplot(1,2,2)
    plt.errorbar(wv, ret['op_ave'][_j,:,1], yerr=ret['op_std'][_j,:,1], fmt='s', capsize=5,
                 linestyle='solid', linewidth=2, label=labels[_j])
    plt.grid(True, linestyle=':')

plt.subplot(1,2,1)
plt.plot(TS2[:4,0], TS2[:4,1], '*k', linestyle='--', label='TS2', linewidth=2, zorder=100, markersize=10)
plt.title(r'$\mu_a$')
plt.xlabel('nm')
plt.legend()
plt.subplot(1,2,2)
plt.plot(TS2[:4,0], TS2[:4,2], '*k', linestyle='--', label='TS2', linewidth=2, zorder=100, markersize=10)
plt.title(r"$\mu'_s$")
plt.xlabel('nm')
plt.tight_layout()