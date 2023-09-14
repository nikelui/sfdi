# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:35:50 2023

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import numpy as np
from scipy.stats import ttest_ind  # independent t-test
from scipy.optimize import curve_fit  # to get A and B parameters
from scipy.io import savemat, loadmat
from sfdi.common.getFile import getFile

def fit_fun(lamb, a, b):
    """Power law function to fit data to"""
    return a * np.power(lamb, -b)

#%% Load data
data_path = getFile("Select pig data")
data = loadmat(data_path)

#%%
###################
# PC1: control
# PD2: control
# PD4: stem cells
###################

# Parameters
wv = np.array([458, 520, 536, 556, 626])
F = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
fx = np.array([np.mean(F[x:x+4]) for x in range(len(F)-3)])

# Combine controls and rename for ease of use
control_w0 = np.concatenate((data['PC1_w0'], data['PD2_w0']), axis=0)
control_w1 = np.concatenate((data['PC1_w1'], data['PD2_w1']), axis=0)
control_w2 = np.concatenate((data['PC1_w2'], data['PD2_w2']), axis=0)

stem_w0 = data['PD4_w0']
stem_w1 = data['PD4_w1']
stem_w2 = data['PD4_w2']

#%% some statistics
control_w0_mean = np.nanmean(control_w0, axis=0)
control_w1_mean = np.nanmean(control_w1, axis=0)
control_w2_mean = np.nanmean(control_w2, axis=0)

control_w0_std = np.nanstd(control_w0, axis=0)
control_w1_std = np.nanstd(control_w1, axis=0)
control_w2_std = np.nanstd(control_w2, axis=0)

# Now fit scattering
opt, cov = curve_fit(fit_fun, wv, control_w1_mean[2,:,1])
perr = np.sqrt(np.diag(cov))  # standard deviation of fit
#%% p-values, control vs stem cells
# Absorption at 530 nm
t0, p0 = ttest_ind(control_w0[:,:,2,0], stem_w0[:,:,2,0], axis=0, equal_var=False, nan_policy='omit')
t1, p1 = ttest_ind(control_w1[:,:,2,0], stem_w1[:,:,2,0], axis=0, equal_var=False, nan_policy='omit')
t2, p2 = ttest_ind(control_w2[:,:,2,0], stem_w2[:,:,2,0], axis=0, equal_var=False, nan_policy='omit')
