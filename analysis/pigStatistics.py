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
# First, re-assign variables for simplicity
# Negative control
mua_control_w0 = control_w0[:,1:3,2,0]
mua_control_w0 = mua_control_w0[~np.isnan(mua_control_w0).any(axis=1)]  # remove nan
mua_control_w1 = control_w1[:,1:3,2,0]
mua_control_w1 = mua_control_w1[~np.isnan(mua_control_w1).any(axis=1)]  # remove nan
mua_control_w2 = control_w2[:,1:3,2,0]
mua_control_w2 = mua_control_w2[~np.isnan(mua_control_w2).any(axis=1)]  # remove nan
# Stem cells
mua_stem_w0 = stem_w0[:,1:3,2,0]
mua_stem_w0 = mua_stem_w0[~np.isnan(mua_stem_w0).any(axis=1)]  # remove nan
mua_stem_w1 = stem_w1[:,1:3,2,0]
mua_stem_w1 = mua_stem_w1[~np.isnan(mua_stem_w1).any(axis=1)]  # remove nan
mua_stem_w2 = stem_w2[:,1:3,2,0]
mua_stem_w2 = mua_stem_w2[~np.isnan(mua_stem_w2).any(axis=1)]  # remove nan

# Negative control
mus_control_w0 = control_w0[:,1:3,:,1]
mus_control_w0 = mus_control_w0[~np.isnan(mus_control_w0).any(axis=(1,2))]  # remove nan
mus_control_w1 = control_w1[:,1:3,:,1]
mus_control_w1 = mus_control_w1[~np.isnan(mus_control_w1).any(axis=(1,2))]  # remove nan
mus_control_w2 = control_w2[:,1:3,:,1]
mus_control_w2 = mus_control_w2[~np.isnan(mus_control_w2).any(axis=(1,2))]  # remove nan
# Stem cells
mus_stem_w0 = stem_w0[:,1:3,:,1]
mus_stem_w0 = mus_stem_w0[~np.isnan(mus_stem_w0).any(axis=(1,2))]  # remove nan
mus_stem_w1 = stem_w1[:,1:3,:,1]
mus_stem_w1 = mus_stem_w1[~np.isnan(mus_stem_w1).any(axis=(1,2))]  # remove nan
mus_stem_w2 = stem_w2[:,1:3,:,1]
mus_stem_w2 = mus_stem_w2[~np.isnan(mus_stem_w2).any(axis=(1,2))]  # remove nan

# Now fit scattering. Might have to loop
# Negative control
A_control_w0 = np.zeros((mus_control_w0.shape[:-1]))
B_control_w0 = np.zeros((mus_control_w0.shape[:-1]))
A_control_w1 = np.zeros((mus_control_w1.shape[:-1]))
B_control_w1 = np.zeros((mus_control_w1.shape[:-1]))
A_control_w2 = np.zeros((mus_control_w2.shape[:-1]))
B_control_w2 = np.zeros((mus_control_w2.shape[:-1]))
# Stem cells
A_stem_w0 = np.zeros((mus_stem_w0.shape[:-1]))
B_stem_w0 = np.zeros((mus_stem_w0.shape[:-1]))
A_stem_w1 = np.zeros((mus_stem_w1.shape[:-1]))
B_stem_w1 = np.zeros((mus_stem_w1.shape[:-1]))
A_stem_w2 = np.zeros((mus_stem_w2.shape[:-1]))
B_stem_w2 = np.zeros((mus_stem_w2.shape[:-1]))

#%% This will take time, so run only once if possible
# control w0
for f in range(2):  # 2 frequencies
    for i in range(len(mus_control_w0)):
        temp, _ = curve_fit(fit_fun, wv, mus_control_w0[i,f,:],
                            p0=[10, 1], method='trf', loss='soft_l1', max_nfev=2000)
        A_control_w0[i, f] = temp[0]
        B_control_w0[i, f] = temp[1]
# control w1
for f in range(2):  # 2 frequencies
    for i in range(len(mus_control_w1)):
        temp, _ = curve_fit(fit_fun, wv, mus_control_w1[i,f,:],
                            p0=[10, 1], method='trf', loss='soft_l1', max_nfev=2000)
        A_control_w1[i, f] = temp[0]
        B_control_w1[i, f] = temp[1]
# control w2
for f in range(2):  # 2 frequencies
    for i in range(len(mus_control_w2)):
        temp, _ = curve_fit(fit_fun, wv, mus_control_w2[i,f,:],
                            p0=[10, 1], method='trf', loss='soft_l1', max_nfev=2000)
        A_control_w2[i, f] = temp[0]
        B_control_w2[i, f] = temp[1]

# stem cells w0
for f in range(2):  # 2 frequencies
    for i in range(len(mus_stem_w0)):
        temp, _ = curve_fit(fit_fun, wv, mus_stem_w0[i,f,:],
                            p0=[10, 1], method='trf', loss='soft_l1', max_nfev=2000)
        A_stem_w0[i, f] = temp[0]
        B_stem_w0[i, f] = temp[1]
# stem cells w1
for f in range(2):  # 2 frequencies
    for i in range(len(mus_stem_w1)):
        temp, _ = curve_fit(fit_fun, wv, mus_stem_w1[i,f,:],
                            p0=[10, 1], method='trf', loss='soft_l1', max_nfev=2000)
        A_stem_w1[i, f] = temp[0]
        B_stem_w1[i, f] = temp[1]
# stem cells w2
for f in range(2):  # 2 frequencies
    for i in range(len(mus_stem_w2)):
        temp, _ = curve_fit(fit_fun, wv, mus_stem_w2[i,f,:],
                            p0=[10, 1], method='trf', loss='soft_l1', max_nfev=2000)
        A_stem_w2[i, f] = temp[0]
        B_stem_w2[i, f] = temp[1]


#%% Absorption at 530 nm
# Control vs stem cells
t0, p0 = ttest_ind(mua_control_w0, mua_stem_w0, axis=0, equal_var=False, nan_policy='omit')
t1, p1 = ttest_ind(mua_control_w1, mua_stem_w1, axis=0, equal_var=False, nan_policy='omit')
t2, p2 = ttest_ind(mua_control_w2, mua_stem_w2, axis=0, equal_var=False, nan_policy='omit')

# Week0 vs week1
tc, pc = ttest_ind(mua_control_w1, mua_control_w2, axis=0, equal_var=False, nan_policy='omit' )
ts, ps = ttest_ind(mua_stem_w1, mua_stem_w2, axis=0, equal_var=False, nan_policy='omit' )

# f1 vs f2
t0, p0 = ttest_ind(mua_stem_w0[:,0], mua_stem_w0[:,1], axis=0, equal_var=False, nan_policy='omit')
t1, p1 = ttest_ind(mua_stem_w1[:,0], mua_stem_w1[:,1], axis=0, equal_var=False, nan_policy='omit')
t2, p2 = ttest_ind(mua_stem_w2[:,0], mua_stem_w2[:,1], axis=0, equal_var=False, nan_policy='omit')


#%% log(A) parameter
# Control vs stem cells
t0, p0 = ttest_ind(np.log(A_control_w0), np.log(A_stem_w0), axis=0, equal_var=False, nan_policy='omit')
t1, p1 = ttest_ind(np.log(A_control_w1), np.log(A_stem_w1), axis=0, equal_var=False, nan_policy='omit')
t2, p2 = ttest_ind(np.log(A_control_w2), np.log(A_stem_w2), axis=0, equal_var=False, nan_policy='omit')

# Week0 vs week1
tc, pc = ttest_ind(np.log(A_control_w1), np.log(A_control_w2), axis=0, equal_var=False, nan_policy='omit' )
ts, ps = ttest_ind(np.log(A_stem_w1), np.log(A_stem_w2), axis=0, equal_var=False, nan_policy='omit' )

# f1 vs f2
t0, p0 = ttest_ind(np.log(A_stem_w0[:,0]), np.log(A_stem_w0[:,1]), axis=0, equal_var=False, nan_policy='omit')
t1, p1 = ttest_ind(np.log(A_stem_w1[:,0]), np.log(A_stem_w1[:,1]), axis=0, equal_var=False, nan_policy='omit')
t2, p2 = ttest_ind(np.log(A_stem_w2[:,0]), np.log(A_stem_w2[:,1]), axis=0, equal_var=False, nan_policy='omit')

#%% B parameter
# Control vs stem cells
t0, p0 = ttest_ind(B_control_w0, B_stem_w0, axis=0, equal_var=False, nan_policy='omit')
t1, p1 = ttest_ind(B_control_w1, B_stem_w1, axis=0, equal_var=False, nan_policy='omit')
t2, p2 = ttest_ind(B_control_w2, B_stem_w2, axis=0, equal_var=False, nan_policy='omit')

# Week0 vs week1
tc, pc = ttest_ind(B_control_w1, B_control_w2, axis=0, equal_var=False, nan_policy='omit' )
ts, ps = ttest_ind(B_stem_w1, B_stem_w2, axis=0, equal_var=False, nan_policy='omit' )

# f1 vs f2
t0, p0 = ttest_ind(B_stem_w0[:,0], B_stem_w0[:,1], axis=0, equal_var=False, nan_policy='omit')
t1, p1 = ttest_ind(B_stem_w1[:,0], B_stem_w1[:,1], axis=0, equal_var=False, nan_policy='omit')
t2, p2 = ttest_ind(B_stem_w2[:,0], B_stem_w2[:,1], axis=0, equal_var=False, nan_policy='omit')


#%% BoxPlots
from matplotlib import pyplot as plt

## mua, control
fig,ax = plt.subplots(nrows=1, ncols=1, num=0, figsize=(5,3.5))
ax.boxplot(mua_control_w0, positions=[1,2], widths=0.8, sym="")
ax.boxplot(mua_control_w1, positions=[4,5], widths=0.8, sym="")
ax.boxplot(mua_control_w2, positions=[7,8], widths=0.8, sym="")
plt.xlim([0, 9])
plt.ylim([-.1, 3])
ax.set_xticks([1.5, 4.5, 7.5])
ax.set_xticklabels(['day-0', 'day-7', 'day-14'], fontsize=12)
ax.set_ylabel(r"mm$^{-1}$", fontsize=12)
ax.set_title(r"$\mu_a$ (530nm)", fontsize=10)
# ax.grid(True, axis="y", linestyle=':')
fig.suptitle("Neg. Control")

## mua, stem cells
fig,ax = plt.subplots(nrows=1, ncols=1, num=1, figsize=(5,3.5))
# plt.hold(True)
ax.boxplot(mua_stem_w0, positions=[1,2], widths=0.8, sym="")
ax.boxplot(mua_stem_w1, positions=[4,5], widths=0.8, sym="")
ax.boxplot(mua_stem_w2, positions=[7,8], widths=0.8, sym="")
plt.xlim([0, 9])
plt.ylim([-.1, 3])
ax.set_xticks([1.5, 4.5, 7.5])
ax.set_xticklabels(['day-0', 'day-7', 'day-14'], fontsize=12)
ax.set_ylabel(r"mm$^{-1}$", fontsize=12)
ax.set_title(r"$\mu_a$ (530nm)", fontsize=10)
# ax.grid(True, axis="y", linestyle=':')
fig.suptitle("pAD-MSCs")


## mus (A, B), control
fig, ax = plt.subplots(nrows=1, ncols=2, num=2, figsize=(8,3.5))
ax[0].boxplot((A_control_w0), positions=[1,2], widths=0.9, sym="")
ax[0].boxplot((A_control_w1), positions=[4,5], widths=0.9, sym="")
ax[0].boxplot((A_control_w2), positions=[7,8], widths=0.9, sym="")
plt.xlim([0, 9])
ax[0].set_yscale("log")
ax[0].set_ylim([1e-19, 1e11])
ax[0].set_xticks([1.5, 4.5, 7.5])
ax[0].set_xticklabels(['day-0', 'day-7', 'day-14'], fontsize=12)
ax[0].set_title("Amplitude", fontsize=10)

ax[1].boxplot((B_control_w0), positions=[1,2], widths=0.9, sym="")
ax[1].boxplot((B_control_w1), positions=[4,5], widths=0.9, sym="")
ax[1].boxplot((B_control_w2), positions=[7,8], widths=0.9, sym="")
plt.xlim([0, 9])
plt.ylim([-3, 7])
ax[1].set_xticks([1.5, 4.5, 7.5])
ax[1].set_xticklabels(['day-0', 'day-7', 'day-14'], fontsize=12)
ax[1].set_title("Slope", fontsize=10)
fig.suptitle("Neg. Control")

## mus (A, B), stem cells
fig, ax = plt.subplots(nrows=1, ncols=2, num=3, figsize=(8,3.5))
ax[0].boxplot((A_stem_w0), positions=[1,2], widths=0.9, sym="")
ax[0].boxplot((A_stem_w1), positions=[4,5], widths=0.9, sym="")
ax[0].boxplot((A_stem_w2), positions=[7,8], widths=0.9, sym="")
plt.xlim([0, 9])
ax[0].set_yscale("log")
ax[0].set_ylim([1e-19, 1e11])
ax[0].set_xticks([1.5, 4.5, 7.5])
ax[0].set_xticklabels(['day-0', 'day-7', 'day-14'], fontsize=12)
ax[0].set_title("Amplitude", fontsize=10)

ax[1].boxplot((B_stem_w0), positions=[1,2], widths=0.9, sym="")
ax[1].boxplot((B_stem_w1), positions=[4,5], widths=0.9, sym="")
ax[1].boxplot((B_stem_w2), positions=[7,8], widths=0.9, sym="")
plt.xlim([0, 9])
plt.ylim([-3, 7])
ax[1].set_xticks([1.5, 4.5, 7.5])
ax[1].set_xticklabels(['day-0', 'day-7', 'day-14'], fontsize=12)
ax[1].set_title("Slope", fontsize=10)
fig.suptitle("pAD-MSCs")

#%% Histograms
from matplotlib import pyplot as plt

## mua, control
fig,ax = plt.subplots(nrows=1, ncols=1, num=0, figsize=(5,3.5))
ax.bar(np.array([-0.5, 0.5])+1.5, np.mean(mua_control_w0, axis=0), yerr=np.std(mua_control_w0, axis=0),
       width=1, color=["#6facc5", "#c1413c"], capsize=5)
ax.bar(np.array([-0.5, 0.5])+3.7, np.mean(mua_control_w1, axis=0), yerr=np.std(mua_control_w1, axis=0),
       width=1, color=["#6facc5", "#c1413c"], capsize=5)
ax.bar(np.array([-0.5, 0.5])+5.9, np.mean(mua_control_w2, axis=0), yerr=np.std(mua_control_w2, axis=0),
       width=1, color=["#6facc5", "#c1413c"], capsize=5)
plt.ylim([0, 3])
ax.set_xticks([1.5, 3.7, 5.9])
ax.set_xticklabels(['day-0', 'day-7', 'day-14'], fontsize=12)
ax.set_ylabel(r"mm$^{-1}$", fontsize=12)
ax.set_title(r"$\mu_a$ (530nm)", fontsize=10)
fig.suptitle("Neg. Control")

## mua, stem cells
fig,ax = plt.subplots(nrows=1, ncols=1, num=1, figsize=(5,3.5))
ax.bar(np.array([-0.5, 0.5])+1.5, np.mean(mua_stem_w0, axis=0), yerr=np.std(mua_stem_w0, axis=0),
       width=1, color=["#6facc5", "#c1413c"], capsize=5)
ax.bar(np.array([-0.5, 0.5])+3.7, np.mean(mua_stem_w1, axis=0), yerr=np.std(mua_stem_w1, axis=0),
       width=1, color=["#6facc5", "#c1413c"], capsize=5)
ax.bar(np.array([-0.5, 0.5])+5.9, np.mean(mua_stem_w2, axis=0), yerr=np.std(mua_stem_w2, axis=0),
       width=1, color=["#6facc5", "#c1413c"], capsize=5)
plt.ylim([0, 3])
ax.set_xticks([1.5, 3.7, 5.9])
ax.set_xticklabels(['day-0', 'day-7', 'day-14'], fontsize=12)
ax.set_ylabel(r"mm$^{-1}$", fontsize=12)
ax.set_title(r"$\mu_a$ (530nm)", fontsize=10)
fig.suptitle("pAD-MSCs")


## mus (A, B), control
fig, ax = plt.subplots(nrows=1, ncols=2, num=2, figsize=(8,3.5))
ax[0].bar(np.array([-0.5, 0.5])+1.5, np.mean(np.log(A_control_w0), axis=0),
          yerr=np.std(np.log(A_control_w0), axis=0),
       width=1, color=["#6facc5", "#c1413c"], capsize=5)
ax[0].bar(np.array([-0.5, 0.5])+3.7, np.mean(np.log(A_control_w1), axis=0),
          yerr=np.std(np.log(A_control_w1), axis=0),
       width=1, color=["#6facc5", "#c1413c"], capsize=5)
ax[0].bar(np.array([-0.5, 0.5])+5.9, np.mean(np.log(A_control_w2), axis=0),
          yerr=np.std(np.log(A_control_w2), axis=0),
       width=1, color=["#6facc5", "#c1413c"], capsize=5)
# ax[0].set_yscale("log")
# plt.ylim([1e20, 1e54])
ax[0].set_xticks([1.5, 3.7, 5.9])
ax[0].set_xticklabels(['day-0', 'day-7', 'day-14'], fontsize=12)
ax[0].set_title("Amplitude", fontsize=10)

ax[1].bar(np.array([-0.5, 0.5])+1.5, np.mean(B_control_w0, axis=0), yerr=np.std(B_control_w0, axis=0),
       width=1, color=["#6facc5", "#c1413c"], capsize=5)
ax[1].bar(np.array([-0.5, 0.5])+3.7, np.mean(B_control_w1, axis=0), yerr=np.std(B_control_w1, axis=0),
       width=1, color=["#6facc5", "#c1413c"], capsize=5)
ax[1].bar(np.array([-0.5, 0.5])+5.9, np.mean(B_control_w2, axis=0), yerr=np.std(B_control_w2, axis=0),
       width=1, color=["#6facc5", "#c1413c"], capsize=5)
plt.ylim([-1.8, 5.2])
ax[1].set_xticks([1.5, 3.7, 5.9])
ax[1].set_xticklabels(['day-0', 'day-7', 'day-14'], fontsize=12)
ax[1].set_title("Slope", fontsize=10)
fig.suptitle("Neg. Control")

## mus (A, B), stem cells
fig, ax = plt.subplots(nrows=1, ncols=2, num=4, figsize=(8,3.5))
ax[0].bar(np.array([-0.5, 0.5])+1.5, np.mean(np.log(A_stem_w0), axis=0),
          yerr=np.std(np.log(A_stem_w0), axis=0),
       width=1, color=["#6facc5", "#c1413c"], capsize=5)
ax[0].bar(np.array([-0.5, 0.5])+3.7, np.mean(np.log(A_stem_w1), axis=0),
          yerr=np.std(np.log(A_stem_w1), axis=0),
       width=1, color=["#6facc5", "#c1413c"], capsize=5)
ax[0].bar(np.array([-0.5, 0.5])+5.9, np.mean(np.log(A_stem_w2), axis=0),
          yerr=np.std(np.log(A_stem_w2), axis=0),
       width=1, color=["#6facc5", "#c1413c"], capsize=5)
# ax[0].set_yscale("log")
# plt.ylim([1e20, 1e54])
ax[0].set_xticks([1.5, 3.7, 5.9])
ax[0].set_xticklabels(['day-0', 'day-7', 'day-14'], fontsize=12)
ax[0].set_title("Amplitude", fontsize=10)

ax[1].bar(np.array([-0.5, 0.5])+1.5, np.mean(B_stem_w0, axis=0), yerr=np.std(B_stem_w0, axis=0),
       width=1, color=["#6facc5", "#c1413c"], capsize=5)
ax[1].bar(np.array([-0.5, 0.5])+3.7, np.mean(B_stem_w1, axis=0), yerr=np.std(B_stem_w1, axis=0),
       width=1, color=["#6facc5", "#c1413c"], capsize=5)
ax[1].bar(np.array([-0.5, 0.5])+5.9, np.mean(B_stem_w2, axis=0), yerr=np.std(B_stem_w2, axis=0),
       width=1, color=["#6facc5", "#c1413c"], capsize=5)
plt.ylim([-1.8, 5.2])
ax[1].set_xticks([1.5, 3.7, 5.9])
ax[1].set_xticklabels(['day-0', 'day-7', 'day-14'], fontsize=12)
ax[1].set_title("Slope", fontsize=10)
fig.suptitle("pAD-MSCs")