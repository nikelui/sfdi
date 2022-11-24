# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:06:53 2022

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Direct model - 2 layer model of scattering using fluence to estimate penetration depth
"""
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.io import loadmat  # new standard: work with Matlab files for compatibility
import addcopyfighandler

from sfdi.common.getPath import getPath
from sfdi.analysis.dataDict import dataDict
from sfdi.common.readParams import readParams
from sfdi.analysis import dataDict
from sfdi.common import models
from sfdi.processing.crop import crop

# support functions
def load_obj(name, path):
    """Utility function to load python objects using pickle module"""
    with open('{}/obj/{}.pkl'.format(path, name), 'rb') as f:
        return pickle.load(f)

# Load dataset
data_path = getPath('Select data path')
par = readParams('{}/processing_parameters.ini'.format(data_path))  # optional
if 'wv' in par.keys():
    wv = par['wv']
else:
    wv = np.array([458, 520, 536, 556, 626])  # wavelengts (nm). Import from params?
# regex = re.compile('.*f\d*\.mat')  # regular expression for optical properties
# regex2 = re.compile('.*calR.mat')  # regular expression for calibrated reflectance
# regex3 = re.compile('SFDS.*\.mat')  # regular expression for SFDS data

# If the dataset has already been processed, load it
if os.path.exists('{}/obj/dataset.pkl'.format(data_path)):
    data = load_obj('dataset', data_path)

#%% Get relevant datasets
dz = 0.01  # resolution
# thick = np.array([0.125, 0.265, 0.51, 0.67, 1.17])  # thickness of thin phantoms batch3
thick = np.array([0.168, 0.196, 0.277, 0.479, 0.727])  # thickness of thin phantoms batch4
asd = loadmat(f'{data_path}/SFDS_8fx.mat')
# asd_plus = loadmat(f'{data_path}/SFDS_8fx_plus5.mat')
# asd_minus = loadmat(f'{data_path}/SFDS_8fx_minus5.mat')
ret = data.singleROI('TiObaseTop', norm=None, fit='single', f=[0,1,2,3,4], I=2e3)
w = 110  # field of view [mm]
dw = 0.05  # Difference in %

# df = 0.05  # mm^-1
fx = np.array([np.mean(par['fx'][i:i+4]) for i in range(len(par['fx'])-3)])

# fx_plus = fx + df
# fx_min = fx - df

fx_plus = fx * (w / (w*(1-dw)))
fx_min = fx * (w / (w*(1+dw)))

z = np.arange(0, 10, dz)

keys = [x for x in data.keys() if 'TiO' in x or 'AlObaseTop' in x]
# keys.remove('TiObaseBottom')
keys.sort()

## Here, do an average for SFDS measurements in multiple locations
# kkeys = [[x for x in keys if '_1' in x],
#          [x for x in keys if '_2' in x],
#          [x for x in keys if '_3' in x]]
# asd_mean = {}
# asd_std = {}
# for _i in range(len(kkeys[0])):
#     k = kkeys[0][_i][:-2]
#     asd_mean[k] = np.mean(np.array([asd[kkeys[0][_i]], asd[kkeys[1][_i]],asd[kkeys[2][_i]]]), axis=0)
#     asd_std[k] = np.std(np.array([asd[kkeys[0][_i]], asd[kkeys[1][_i]],asd[kkeys[2][_i]]]), axis=0)

## Here, transform SFDI dataset to have the same format as SFDS for compatibility
ROI = ret['ROI']  # for convenience
temp = {}
for key in keys:
    temp[key] = np.zeros((5,8,2), dtype=float)
    for _f, f in enumerate(data[key].keys()):
        temp[key][:,_f,:] = np.mean(crop(data[key][f]['op_fit_maps'], ROI), axis=(0,1))
asd = temp  # to use SFDI data without chaning the code

# Models of fluence
phi_diff = {}  # diffusion
phi_diffusion = {}  # diffusion, Seo
phi_deltaP1 = {}  # delta-P1, Vasen modified
phi_dp1 = {}  # delta-P1, Seo original
# Errorbars for delta-P1
phi_deltaP1_plus = {}
phi_deltaP1_min = {}
phi_dp1_plus = {}
phi_dp1_min = {}

# models of alpha
alpha_diff = {}
alpha_diffusion = {}
alpha_deltaP1 = {}
alpha_dp1 = {}
# Errorbars
alpha_deltaP1_plus = {}
alpha_deltaP1_min = {}
alpha_dp1_min = {}
alpha_dp1_plus = {}


# Models of scattering
mus_top = np.squeeze(asd['TiObaseTop'][:,:,1]).T
# mus_top[:,:] = mus_top[0,:]  # To fix mus to a single value (f0)
mus_bot = np.squeeze(asd['AlObaseTop'][:,:,1]).T
# mus_bot[:,:] = mus_bot[0,:]  # To fix mus to a single value (f0)
mus_meas = {k:asd[k][:,:,1].T for k in [x for x in asd.keys() if 'TiO' in x or 'AlObase' in x] if 'Top' not in k}
# mus_meas_plus = {k:asd_plus[k][:,:,1].T for k in [x for x in asd_plus.keys() if 'TiO' in x or 'AlObase' in x] if 'Top' not in k}
# mus_meas_minus = {k:asd_minus[k][:,:,1].T for k in [x for x in asd_minus.keys() if 'TiO' in x or 'AlObase' in x] if 'Top' not in k}

mus_model_diff = {}
mus_model_diffusion = {}
mus_model_deltaP1 = {}
mus_model_deltaP1_plus = {}
mus_model_deltaP1_min = {}
mus_model_dp1 = {}
mus_model_dp1_plus = {}
mus_model_dp1_min = {}

power = 2  # to quickly change from phi to phi**2
for _i, key in enumerate([x for x in asd.keys() if 'TiO' in x or 'AlObase' in x]):
    phi_diff[key] = models.phi_diff(z, asd[key][:,:,0].T, asd[key][:,:,1].T/0.2, fx)  # diffusion
    phi_diffusion[key] = models.phi_diffusion(z, asd[key][:,:,0].T, asd[key][:,:,1].T/0.2, fx)  # diffusion, Seo
    phi_deltaP1[key] = models.phi_deltaP1(z, asd[key][:,:,0].T, asd[key][:,:,1].T/0.2, fx)  # d-p1, Luigi
    phi_deltaP1_plus[key] = models.phi_deltaP1(z, asd[key][:,:,0].T, asd[key][:,:,1].T/0.2, fx_plus)
    phi_deltaP1_min[key] = models.phi_deltaP1(z, asd[key][:,:,0].T, asd[key][:,:,1].T/0.2, fx_min)
    phi_dp1[key] = models.phi_dP1(z, asd[key][:,:,0].T, asd[key][:,:,1].T/0.2, fx)  # d-p1, Seo
    phi_dp1_plus[key] = models.phi_dP1(z, asd[key][:,:,0].T, asd[key][:,:,1].T/0.2, fx_plus)
    phi_dp1_min[key] = models.phi_dP1(z, asd[key][:,:,0].T, asd[key][:,:,1].T/0.2, fx_min)
    
    
    
    if _i > 0 and _i < len([x for x in asd.keys() if 'TiO' in x or 'AlObase' in x])-1 and True:
        alpha_diff[key] = models.alpha(phi_diff[key]**power, z, thick[_i-1])
        alpha_diffusion[key] = models.alpha(phi_diffusion[key]**power, z, thick[_i-1])
        alpha_deltaP1[key] = models.alpha(phi_deltaP1[key]**power, z, thick[_i-1])
        alpha_deltaP1_plus[key] = models.alpha(phi_deltaP1_plus[key]**power, z, thick[_i-1])
        alpha_deltaP1_min[key] = models.alpha(phi_deltaP1_min[key]**power, z, thick[_i-1])
        alpha_dp1[key] = models.alpha(phi_dp1[key]**power, z, thick[_i-1])
        alpha_dp1_plus[key] = models.alpha(phi_dp1_plus[key]**power, z, thick[_i-1])
        alpha_dp1_min[key] = models.alpha(phi_dp1_min[key]**power, z, thick[_i-1])
        
        mus_model_diff[key] = alpha_diff[key] * mus_top + (1-alpha_diff[key])* mus_bot
        mus_model_diffusion[key] = alpha_diffusion[key] * mus_top + (1-alpha_diffusion[key])* mus_bot
        mus_model_deltaP1[key] = alpha_deltaP1[key] * mus_top + (1-alpha_deltaP1[key])* mus_bot
        mus_model_deltaP1_plus[key] = alpha_deltaP1_plus[key] * mus_top + (1-alpha_deltaP1_plus[key])* mus_bot
        mus_model_deltaP1_min[key] = alpha_deltaP1_min[key] * mus_top + (1-alpha_deltaP1_min[key])* mus_bot
        mus_model_dp1[key] = alpha_dp1[key] * mus_top + (1-alpha_dp1[key])* mus_bot
        mus_model_dp1_plus[key] = alpha_dp1_plus[key] * mus_top + (1-alpha_dp1_plus[key])* mus_bot
        mus_model_dp1_min[key] = alpha_dp1_min[key] * mus_top + (1-alpha_dp1_min[key])* mus_bot

if False:  # piecewise continuous model
    phi2_diff = models.phi_2lp(thick, phi_diff['TiObaseTop'], phi_diff['AlObaseTop'], z)
    phi2_diffusion = models.phi_2lp(thick, phi_diffusion['TiObaseTop'], phi_diffusion['AlObaseTop'], z)
    phi2_deltaP1 = models.phi_2lp(thick, phi_deltaP1['TiObaseTop'], phi_deltaP1['AlObaseTop'], z)
    phi2_dp1 = models.phi_2lp(thick, phi_dp1['TiObaseTop'], phi_dp1['AlObaseTop'], z)
    
    for _i, key in enumerate(asd.keys()):
        if _i > 0 and _i < len(asd.keys())-1:
            alpha_diff[key] = models.alpha(phi2_diff[:,:,:,_i-1]**power, z, thick[_i-1])
            alpha_diffusion[key] = models.alpha(phi2_diffusion[:,:,:,_i-1]**power, z, thick[_i-1])
            alpha_deltaP1[key] = models.alpha(phi2_deltaP1[:,:,:,_i-1]**power, z, thick[_i-1])
            alpha_dp1[key] = models.alpha(phi2_dp1[:,:,:,_i-1]**power, z, thick[_i-1])
            
            mus_model_diff[key] = alpha_diff[key] * mus_top + (1-alpha_diff[key])* mus_bot
            mus_model_diffusion[key] = alpha_diffusion[key] * mus_top + (1-alpha_diffusion[key])* mus_bot
            mus_model_deltaP1[key] = alpha_deltaP1[key] * mus_top + (1-alpha_deltaP1[key])* mus_bot
            mus_model_dp1[key] = alpha_dp1[key] * mus_top + (1-alpha_dp1[key])* mus_bot

# %% Simulation
lamb = 500  # nm
# WV = np.where(asd['wv'][:,0] >= lamb)[0][0]
WV = 0
F = 0  # spatial frequency to plot

# Plot fluence
if False:
    mua = np.ones((1,1), dtype=float) * 0.1
    mus = np.ones((1,1), dtype=float) * 1 / (0.2)  # convert musp to mus
    
    fx = np.arange(0, 0.3, 0.05)
    dz = 0.01
    z = np.arange(0, 10, dz)
    
    diff = np.squeeze(models.phi_diff(z, mua, mus, fx))
    diff_dc, diff_ac = (np.squeeze(x) for x in models.phi_diffusion(z, mua, mus, fx))
    diff_seo = diff_ac + diff_dc
    deltaP1 = np.squeeze(models.phi_deltaP1(z, mua, mus, fx))
    dp1 = np.squeeze(models.phi_dP1(z, mua, mus, fx))
    
    # SDA
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cm.Blues(np.linspace(1, 0.2, len(fx))))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), num=1)
    ax.plot(z, diff.T)
    ax.grid(True, linestyle=':')
    ax.set_xlim([0, 10])
    ax.set_xlabel('mm')
    ax.set_ylabel(r'$\varphi$', fontsize=14)
    ax.set_title(r"Diffusion, SDA ($\mu_a$ = {}, $\mu'_s$ = {})".format(mua[0][0], mus[0][0]*0.2))
    ax.legend(['fx = {:.2f} mm$^{{-1}}$'.format(x) for x in fx], fontsize=9)
    fig.tight_layout()
    
    # Diffusion - SEO AC
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), num=2)
    ax.plot(z, diff_ac.T)
    ax.grid(True, linestyle=':')
    ax.set_xlim([0, 10])
    ax.set_xlabel('mm')
    ax.set_ylabel(r'$\varphi$', fontsize=14)
    ax.set_title(r"Diffusion, Seo - AC ($\mu_a$ = {}, $\mu'_s$ = {})".format(mua[0][0], mus[0][0]*0.2))
    ax.legend(['fx = {:.2f} mm$^{{-1}}$'.format(x) for x in fx], fontsize=9)
    fig.tight_layout()
    
    # Diffusion - SEO DC
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), num=3)
    ax.plot(z, diff_dc.T)
    ax.grid(True, linestyle=':')
    ax.set_xlim([0, 10])
    ax.set_xlabel('mm')
    ax.set_ylabel(r'$\varphi$', fontsize=14)
    ax.set_title(r"Diffusion, Seo - DC ($\mu_a$ = {}, $\mu'_s$ = {})".format(mua[0][0], mus[0][0]*0.2))
    ax.legend(['fx = {:.2f} mm$^{{-1}}$'.format(x) for x in fx], fontsize=9)
    fig.tight_layout()
    
    # Diffusion - SEO
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), num=4)
    ax.plot(z, (diff_dc + diff_ac).T)
    ax.grid(True, linestyle=':')
    ax.set_xlim([0, 10])
    ax.set_xlabel('mm')
    ax.set_ylabel(r'$\varphi$', fontsize=14)
    ax.set_title(r"Diffusion, Seo ($\mu_a$ = {}, $\mu'_s$ = {})".format(mua[0][0], mus[0][0]*0.2))
    ax.legend(['fx = {:.2f} mm$^{{-1}}$'.format(x) for x in fx], fontsize=9)
    fig.tight_layout()
    
    # delta-P1 - SEO
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), num=5)
    ax.plot(z, dp1.T)
    ax.grid(True, linestyle=':')
    ax.set_xlim([0, 10])
    ax.set_xlabel('mm')
    ax.set_ylabel(r'$\varphi$', fontsize=14)
    ax.set_title(r"$\delta$-P1 - Seo ($\mu_a$ = {}, $\mu'_s$ = {})".format(mua[0][0], mus[0][0]*0.2))
    ax.legend(['fx = {:.2f} mm$^{{-1}}$'.format(x) for x in fx], fontsize=9)
    fig.tight_layout()
    
    # delta-P1 - Vasen
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), num=6)
    ax.plot(z, deltaP1.T)
    ax.grid(True, linestyle=':')
    ax.set_xlim([0, 10])
    ax.set_xlabel('mm')
    ax.set_ylabel(r'$\varphi$', fontsize=14)
    ax.set_title(r"$\delta$-P1 - Vasen modified ($\mu_a$ = {}, $\mu'_s$ = {})".format(mua[0][0], mus[0][0]*0.2))
    ax.legend(['fx = {:.2f} mm$^{{-1}}$'.format(x) for x in fx], fontsize=9)
    fig.tight_layout()

# Plot fluence, normalized
if True:
    mua = np.ones((1,1), dtype=float) * 0.02
    mus = np.ones((1,1), dtype=float) * 1 / (0.2)  # convert musp to mus
    
    fx = np.arange(0, 0.51, 0.05)
    fx = np.array([np.mean(fx[_i:_i+4]) for _i in range(len(fx)-3)])
    dz = 0.01
    z = np.arange(0, 10, dz)
    
    diff = np.squeeze(models.phi_diff(z, mua, mus, fx))
    diff_dc, diff_ac = (np.squeeze(x) for x in models.phi_diffusion(z, mua, mus, fx))
    diff_seo = diff_ac + diff_dc
    deltaP1 = np.squeeze(models.phi_deltaP1(z, mua, mus, fx))
    dp1 = np.squeeze(models.phi_dP1(z, mua, mus, fx))
    
    # diffusion
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), num=6)
    ax.plot(z, diff.T / np.sum(diff.T * dz, axis=0))
    # ax.grid(True, linestyle=':')
    ax.set_xlim([0, 5])
    ax.set_ylim([0, .9])
    ax.set_xlabel('mm')
    ax.set_ylabel(r'$\varphi$', fontsize=14)
    ax.set_title(r"SDA ($\mu_a$ = {}, $\mu'_s$ = {})".format(mua[0][0], mus[0][0]*0.2))
    # ax.legend(['fx = {:.3f} mm$^{{-1}}$'.format(x) for x in fx], fontsize=9)
    fig.tight_layout()
    
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cm.Blues(np.linspace(1, 0.2, len(fx))))
    # delta-P1 - SEO
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), num=5)
    ax.plot(z, dp1.T / np.sum(dp1.T * dz, axis=0))
    # ax.grid(True, linestyle=':')
    ax.set_xlim([0, 5])
    ax.set_ylim([0, .9])
    ax.set_xlabel('mm')
    ax.set_ylabel(r'$\varphi$', fontsize=14)
    ax.set_title(r"$\delta$-P1 - Seo ($\mu_a$ = {}, $\mu'_s$ = {})".format(mua[0][0], mus[0][0]*0.2))
    # ax.legend(['fx = {:.3f} mm$^{{-1}}$'.format(x) for x in fx], fontsize=9)
    fig.tight_layout()
    
    # delta-P1 - Luigi
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), num=6)
    # ax.plot(z, deltaP1.T / np.sum(deltaP1.T * dz, axis=0))
    # ax.grid(True, linestyle=':')
    # ax.set_xlim([0, 5])
    # ax.set_ylim([0, 1])
    # ax.set_xlabel('mm')
    # ax.set_ylabel(r'$\varphi$', fontsize=14)
    # ax.set_title(r"$\delta$-P1 - Luigi ($\mu_a$ = {}, $\mu'_s$ = {})".format(mua[0][0], mus[0][0]*0.2))
    # ax.legend(['fx = {:.3f} mm$^{{-1}}$'.format(x) for x in fx], fontsize=9)
    # fig.tight_layout()

# Plot fluence(fx) normalized
if False:
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cm.Purples(np.linspace(1, 0.2, len(fx))))
    for _i, key in enumerate(phi_deltaP1.keys()):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(7,8), num=_i+1)
        ax[1].plot(z, phi_deltaP1[key][:,WV,:].T / np.sum(phi_deltaP1[key][:,WV,:].T*dz, axis=0))
        ax[1].set_title(f'{key} - Luigi')
        ax[1].grid(True, linestyle=':')
        ax[1].set_xlabel('mm')
        ax[1].set_ylabel(r'$\varphi$', fontsize=14)
        ax[1].set_xlim([0, 5])
        ax[1].set_ylim([0, 1.75])
        ax[0].plot(z, phi_dp1[key][:,WV,:].T / np.sum(phi_dp1[key][:,WV,:].T*dz, axis=0))
        ax[0].set_title(f'{key} - Seo')
        ax[0].grid(True, linestyle=':')
        # ax[0].set_xlabel('mm')
        ax[0].set_ylabel(r'$\varphi$', fontsize=14)
        ax[0].set_xlim([0, 5])
        ax[0].set_ylim([0, 1.75])
        fig.tight_layout()

# Plot fluence, piecewise continuous
if False:
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cm.Blues(np.linspace(1, 0.2, len(thick))))
    # SDA
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), num=1)
    ax.plot(z, phi2_diff[0, WV,:,:])
    ax.grid(True, linestyle=':')
    ax.set_xlim([0, 6])
    ax.set_xlabel('mm')
    ax.set_ylabel(r'$\varphi$', fontsize=14)
    ax.set_title(r"Diffusion, SDA")
    ax.legend(['d = {:.3f}mm'.format(x) for x in thick])
    fig.tight_layout()
    
    # Diffusion - Seo
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), num=2)
    ax.plot(z, phi2_diffusion[0, WV,:,:])
    ax.grid(True, linestyle=':')
    ax.set_xlim([0, 6])
    ax.set_xlabel('mm')
    ax.set_ylabel(r'$\varphi$', fontsize=14)
    ax.set_title(r"Diffusion, Seo")
    ax.legend(['d = {:.3f}mm'.format(x) for x in thick])
    fig.tight_layout()
    
    # deltaP1 - Luigi
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), num=3)
    ax.plot(z, phi2_deltaP1[0, WV,:,:])
    ax.grid(True, linestyle=':')
    ax.set_xlim([0, 6])
    ax.set_xlabel('mm')
    ax.set_ylabel(r'$\varphi$', fontsize=14)
    ax.set_title(r"$\delta-P1$, Luigi")
    ax.legend(['d = {:.3f}mm'.format(x) for x in thick])
    fig.tight_layout()
    
    # deltaP1 - Seo
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), num=4)
    ax.plot(z, phi2_dp1[0, WV,:,:])
    ax.grid(True, linestyle=':')
    ax.set_xlim([0, 10])
    ax.set_xlabel('mm')
    ax.set_ylabel(r'$\varphi$', fontsize=14)
    ax.set_title(r"$\delta-P1$, Seo")
    ax.legend(['d = {:.3f}mm'.format(x) for x in thick])
    fig.tight_layout()
    
# Plot mus
if False:
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cm.Reds(np.linspace(1, 0.2, len(fx))))
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), num=1)
    for _i, key in enumerate(mus_meas.keys()):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), num=_i+1)
        ax.plot(fx, mus_model_diff[key][:,WV], linestyle='solid', label='SDA')
        ax.plot(fx, mus_model_diffusion[key][:,WV], linestyle='dotted', label='Diffusion - Seo')
        ax.plot(fx, mus_model_deltaP1[key][:,WV], linestyle='dashed', label='delta-P1 - Luigi')
        ax.plot(fx, mus_model_dp1[key][:,WV], linestyle='dashdot', label='delta-P1 - Seo')
        ax.plot(fx, mus_meas[key][:,WV], '*', label='Measured')
        ax.set_title(f'{key}')
        ax.grid(True, linestyle=':')
        ax.set_xlabel(r'fx (mm$^{{-1}})$')
        ax.set_ylabel(r"$\mu'_s$", fontsize=14)
        fig.tight_layout()
        # if _i == 0:
            # ax.legend()

# Plot mus (only delta-P1) with top and bottom
if False:
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cm.Reds(np.linspace(1, 0.2, len(fx))))
    cmap = cm.get_cmap('Dark2')
    # colors = [cmap(x) for x in np.linspace(0,1,len(mus_meas)+2)]
    colors = ['lime', 'yellowgreen', 'sandybrown', 'skyblue', 'mediumorchid', 'gold' , 'orangered']
    
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), num=1)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), num=1)
    ax.plot(fx, mus_top[:,WV], linestyle='solid', color=colors[0], label=r'TiO$_2$')
    ax.plot(fx, mus_bot[:,WV], linestyle='solid', color=colors[-1], label=r'Al$_2$O$_3$')
    for _i, key in enumerate(mus_meas.keys()):
        # ax.plot(fx, mus_model_deltaP1[key][:,WV], linestyle='dashed', color=colors[_i+1])
                # label=r'{}'.format(key))
        ax.plot(fx, mus_model_dp1[key][:,WV], linestyle='dotted', color=colors[_i+1])
        #         label=r'{}'.format(key))
        ax.plot(fx, mus_meas[key][:,WV], 'o', color=colors[_i+1],
                label=r'{}'.format(key))
    ax.set_title(r'$\delta$-P1 - Seo')
    ax.grid(True, linestyle=':')
    ax.set_xlabel(r'fx (mm$^{{-1}})$')
    ax.set_ylabel(r"$\mu'_s$", fontsize=14)
    fig.tight_layout()
    # ax.legend(framealpha=1, ncol=4)
    # ax.set_axis_off()

# Plot mus (only delta-P1) with top and bottom and errorbars
if False:
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cm.Reds(np.linspace(1, 0.2, len(fx))))
    cmap = cm.get_cmap('Dark2')
    # colors = [cmap(x) for x in np.linspace(0,1,len(mus_meas)+2)]
    colors = ['lime', 'yellowgreen', 'sandybrown', 'skyblue', 'mediumorchid', 'gold' , 'orangered']
    
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), num=1)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4.5), num=1)
    # ax.plot(fx, mus_top[:,WV], linestyle='solid', color=colors[0], label=r'TiO$_2$')
    # ax.plot(fx, mus_bot[:,WV], linestyle='solid', color=colors[-1], label=r'Al$_2$O$_3$')
    ## Normalize top and bottom
    ax.plot(fx, mus_top[:,WV]/mus_top[:,WV], linestyle='solid', color=colors[0], label=r'TiO$_2$')
    ax.plot(fx, mus_bot[:,WV]-mus_bot[:,WV], linestyle='solid', color=colors[-1], label=r'Al$_2$O$_3$')
    for _i, key in enumerate(mus_meas.keys()):
        scale_meas = (mus_meas[key]-mus_bot)/(mus_top-mus_bot)
        scale_deltaP1 = (mus_model_deltaP1[key]-mus_bot)/(mus_top-mus_bot)
        scale_dp1 = (mus_model_dp1[key]-mus_bot)/(mus_top-mus_bot)
        scale_diff = (mus_model_diff[key]-mus_bot)/(mus_top-mus_bot)
        # Luigi d-P1
        # ax.fill_between(fx, mus_model_'deltaP1_plus[key][:,WV], mus_model_deltaP1_min[key][:,WV],
        #                 alpha=0.5, edgecolor=colors[_i+1], facecolor=colors[_i+1])
        # ax.plot(fx, mus_model_deltaP1[key][:,WV], linestyle='dashed', color=colors[_i+1])
        # ax.plot(fx, scale_deltaP1[:,WV], linestyle='dashed', color=colors[_i+1])  # Normalized
        ax.plot(fx, scale_dp1[:,WV], linestyle='dashed', color=colors[_i+1])  # Normalized
        # ax.plot(fx, scale_diff[:,WV], linestyle='dotted', color=colors[_i+1])  # Normalized

        # Seo d-P1
        # ax.fill_between(fx, mus_meas_plus[key][:,WV], mus_meas_minus[key][:,WV],
        #                 alpha=0.25, edgecolor=colors[_i+1], facecolor=colors[_i+1])
        # ax.plot(fx, mus_model_dp1[key][:,WV], linestyle='dotted', color=colors[_i+1])
        # Measured
        # ax.plot(fx, mus_meas[key][:,WV], 'o', color=colors[_i+1],
        #         label=r'{}'.format(key))
        ax.plot(fx, scale_meas[:,WV], 'o', color=colors[_i+1],  # Normalized
                label=r'{}'.format(key))

    # ax.set_title(r'$\delta$-P1'.format(dw*100))
    ax.grid(True, linestyle=':')
    ax.set_xlabel(r'fx (mm$^{{-1}})$')
    ax.set_ylabel(r"$\mu'_s$", fontsize=14)
    fig.tight_layout()
    # ax.legend(framealpha=1, ncol=4)
    # ax.set_axis_off()

# Plot mus(lambda) at a single fx
if False:
    wv = np.array(par['wv'])
    colors = ['lime', 'yellowgreen', 'sandybrown', 'skyblue', 'mediumorchid', 'gold' , 'orangered']
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4.5), num=1)
    ax.plot(wv, mus_top[F,:], linestyle='solid', color=colors[0], label=r'TiO$_2$')
    ax.plot(wv, mus_bot[F,:], linestyle='solid', color=colors[-1], label=r'Al$_2$O$_3$')
    for _i, key in enumerate(mus_meas.keys()):
        # Luigi d-P1
        # ax.fill_between(fx, mus_model_deltaP1_plus[key][F,:], mus_model_deltaP1_min[key][F,:],
        #                 alpha=0.5, edgecolor=colors[_i+1], facecolor=colors[_i+1])
        ax.plot(wv, mus_model_deltaP1[key][F,:], linestyle='dashed', color=colors[_i+1])
        # Seo d-P1
        # ax.fill_between(wv, mus_meas_plus[key][F,:], mus_meas_minus[key][F,:],
        #                 alpha=0.25, edgecolor=colors[_i+1], facecolor=colors[_i+1])
        ax.plot(wv, mus_model_dp1[key][F,:], linestyle='dotted', color=colors[_i+1])
        # Measured
        ax.plot(wv, mus_meas[key][F,:], 'o', color=colors[_i+1],
                label=r'{}'.format(key))
    ax.set_title(r'$\delta$-P1'.format(dw*100))
    ax.grid(True, linestyle=':')
    ax.set_xlabel(r'wv (nm)')
    ax.set_ylabel(r"$\mu'_s$", fontsize=14)
    fig.tight_layout()
    ax.legend(framealpha=1, ncol=4)
    # ax.set_axis_off()
    

# Plot relative errors
if False:
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cm.YlGn(np.linspace(1, 0.2, 5)))
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), num=1)
    for _i, key in enumerate(mus_meas.keys()):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), num=_i+1)
        err = np.abs(mus_model_diff[key][:,WV]-mus_meas[key][:,WV])/mus_meas[key][:,WV] * 100
        ax.plot(fx, err, linestyle='solid', label='SDA')
        err = np.abs(mus_model_diffusion[key][:,WV]-mus_meas[key][:,WV])/mus_meas[key][:,WV] * 100
        ax.plot(fx, err, linestyle='dotted', label='Diffusion - Seo')
        err = np.abs(mus_model_deltaP1[key][:,WV]-mus_meas[key][:,WV])/mus_meas[key][:,WV] * 100
        ax.plot(fx, err, linestyle='dashed', label='delta-P1 - Luigi')
        err = np.abs(mus_model_dp1[key][:,WV]-mus_meas[key][:,WV])/mus_meas[key][:,WV] * 100
        ax.plot(fx, err, linestyle='dashdot', label='delta-P1 - Seo')
        
        # ax.plot(fx, mus_meas[key][:,WV], '*', label='Measured')
        ax.set_title(f'{key}')
        ax.grid(True, linestyle=':')
        ax.set_xlabel(r'fx (mm$^{{-1}})$')
        ax.set_ylabel('% error', fontsize=14)
        fig.tight_layout()
        # if _i == 0:
        #     ax.legend()
        #     ax.set_axis_off()
        
# Plot alpha
if False:
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cm.RdPu(np.linspace(1, 0.2, 5)))
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), num=1)
    for _i, key in enumerate(mus_meas.keys()):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,4), num=_i+1)
        ax.plot(fx, alpha_diff[key][:,WV], linestyle='solid', label='SDA')
        ax.plot(fx, alpha_diffusion[key][:,WV], linestyle='dotted', label='Diffusion - Seo')
        ax.plot(fx, alpha_deltaP1[key][:,WV], linestyle='dashed', label='delta-P1 - Luigi')
        ax.plot(fx, alpha_dp1[key][:,WV], linestyle='dashdot', label='delta-P1 - Seo')
        # ax.plot(fx, mus_meas[key][:,WV], '*', label='Measured')
        ax.set_title(f'{key}')
        ax.grid(True, linestyle=':')
        ax.set_xlabel(r'fx (mm$^{{-1}})$')
        ax.set_ylabel(r"$\alpha$", fontsize=14)
        fig.tight_layout()
        # if _i == 0:
        #     ax.legend()
        #     ax.set_axis_off()