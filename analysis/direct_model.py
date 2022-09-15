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
thick = np.array([0.125, 0.265, 0.51, 0.67, 1.17])  # thickness of thin phantoms
asd = loadmat(f'{data_path}/SFDS_8fx.mat')
fx = np.array([np.mean(par['fx'][i:i+4]) for i in range(len(par['fx'])-3)])
z = np.arange(0, 10, dz)
lamb = 500  # nm
WV = np.where(asd['wv'][:,0] >= lamb)[0][0]
F = 0  # spatial frequency to plot

keys = [x for x in data.keys() if 'TiO' in x or 'AlObaseTop' in x]
keys.remove('TiObaseBottom')
keys.sort()

# Models of fluence
phi_diff = {}  # diffusion
phi_diffusion = {}  # diffusion, Seo
phi_deltaP1 = {}  # delta-P1, Vasen modified
phi_dp1 = {}  # delta-P1, Seo original

# models of alpha
alpha_diff = {}
alpha_diffusion = {}
alpha_deltaP1 = {}
alpha_dp1 = {}

# Models of scattering
mus_top = np.squeeze(asd['TiObaseTop'][:,:,1]).T
mus_bot = np.squeeze(asd['AlObaseTop'][:,:,1]).T
mus_meas = {k:asd[k][:,:,1].T for k in keys[1:-1]}

mus_model_diff = {}
mus_model_diffusion = {}
mus_model_deltaP1 = {}
mus_model_dp1 = {}

power = 2  # to quickly change from phi to phi**2
for _i, key in enumerate(keys):
    phi_diff[key] = models.phi_diff(z, asd[key][:,:,0].T, asd[key][:,:,1].T/0.2, fx)  # diffusion
    phi_diffusion[key] = models.phi_diffusion(z, asd[key][:,:,0].T, asd[key][:,:,1].T/0.2, fx)  # diffusion, Seo
    phi_deltaP1[key] = models.phi_deltaP1(z, asd[key][:,:,0].T, asd[key][:,:,1].T/0.2, fx)  # d-p1, Luigi
    phi_dp1[key] = models.phi_dP1(z, asd[key][:,:,0].T, asd[key][:,:,1].T/0.2, fx)  # d-p1, Seo
    
    if _i > 0 and _i < len(keys)-1:
        alpha_diff[key] = models.alpha(phi_diff[key]**power, z, thick[_i-1])
        alpha_diffusion[key] = models.alpha(phi_diffusion[key]**power, z, thick[_i-1])
        alpha_deltaP1[key] = models.alpha(phi_deltaP1[key]**power, z, thick[_i-1])
        alpha_dp1[key] = models.alpha(phi_dp1[key]**power, z, thick[_i-1])

        mus_model_diff[key] = alpha_diff[key] * mus_top + (1-alpha_diff[key])* mus_bot
        mus_model_diffusion[key] = alpha_diffusion[key] * mus_top + (1-alpha_diffusion[key])* mus_bot
        mus_model_deltaP1[key] = alpha_deltaP1[key] * mus_top + (1-alpha_deltaP1[key])* mus_bot
        mus_model_dp1[key] = alpha_dp1[key] * mus_top + (1-alpha_dp1[key])* mus_bot

# %% Simulation
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
        if _i == 0:
            ax.legend()
        
# Plot mus
if True:
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