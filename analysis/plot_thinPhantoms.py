# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 15:52:54 2022

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Plot data measured in Japan on thin silicone phantoms
"""
# import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# from scipy.interpolate import interp1d
from scipy.io import loadmat
import addcopyfighandler

data_path = r'C:\Users\luibe59\Documents\AcquisitionCode\5wv_capture\Raw data\220804_Luigi_doubleSphere'
sfds_path = 'C:/Users/luibe59/Documents/AcquisitionCode/5wv_capture/Processed data/thinPhantomsBatch3_initial'
cal_path = 'C:/Users/luibe59/Documents/sfdi/common/phantoms/phantom_uci_201703.txt'

g = 0.8  # anisotropy coefficient

# column = dataset, row = wavelength
mua_TRS = np.array([[0.0206718, 0.0219627, 0.0236566],  # Al2O3_3
                    [0.033617, 0.0349809, 0.0378436],   # TiO2_3
                    [0.0792077, 0.0755053, 0.0852649],  # Al2O3_4
                    [0.0670533, 0.0685535, 0.0755324],  # TiO2_4
                    [0.0159364, 0.0165234, 0.0179663],     # Calibration 15mm
                    [0.0164206, 0.0164691, 0.0178269]]).T  # Calibration 30mm

mus_TRS = np.array([[.63766, .656042, .732063],  # Al2O3_3
                    [1.7251, 1.65468, 1.73187],  # TiO2_3
                    [1.06745, 1.00884, 1.18469], # Al2O3_4
                    [.861506, .831409, .935679], # TiO2_4
                    [.618704, .600712, .646949],    # Calibration 15mm
                    [.58984, .556567, .582782]]).T  # Calibration 30mm
wv_TRS = np.array([761, 797, 833])  # nm

# Double integrating spheres
data = pd.read_excel('{}/data.xlsx'.format(data_path), [0,1,2,3])

data_sfds = loadmat('{}/SFDS_8fx.mat'.format(sfds_path))
calib = np.genfromtxt(cal_path, delimiter='\t')

# BATCH 3
# Plot A2O3
fig, ax = plt.subplots(1,2, num=1, figsize=(9,3.5))
ax[0].plot(data_sfds['wv'][:,0], data_sfds['AlObaseTop'][:,0,0], label='SFDS')
ax[0].plot(data[0]['Wavelength [nm]'], data[0]['mua'], label='Double sphere')
ax[0].plot(wv_TRS, mua_TRS[:,0], 'o', label='TRS')
ax[0].grid(True, linestyle=':')
ax[0].set_xlabel('nm')
ax[0].set_ylabel(r'mm$^{{-1}}$')
ax[0].set_title(r'$\mu_a$')
ax[1].plot(data_sfds['wv'][:,0], data_sfds['AlObaseTop'][:,0,1], label='SFDS')
ax[1].plot(data[0]['Wavelength [nm]'], data[0]['mus']*(1-g), label='Double sphere')
ax[1].plot(wv_TRS, mus_TRS[:,0], 'o', label='TRS')
ax[1].grid(True, linestyle=':')
ax[1].set_xlabel('nm')
ax[1].set_title(r"$\mu'_s$")
ax[1].legend()
plt.suptitle('Al2O3')
plt.tight_layout()

# Plot TiO2
fig, ax = plt.subplots(1,2, num=2, figsize=(9,3.5))
ax[0].plot(data_sfds['wv'][:,0], data_sfds['TiObaseTop'][:,0,0], label='SFDS')
ax[0].plot(data[2]['Wavelength [nm]'], data[2]['mua'], label='Double sphere')
ax[0].plot(wv_TRS, mua_TRS[:,1], 'o', label='TRS_15mm')
ax[0].grid(True, linestyle=':')
ax[0].set_xlabel('nm')
ax[0].set_ylabel(r'mm$^{{-1}}$')
ax[0].set_title(r'$\mu_a$')
ax[1].plot(data_sfds['wv'][:,0], data_sfds['TiObaseTop'][:,0,1], label='SFDS')
ax[1].plot(data[2]['Wavelength [nm]'], data[2]['mus']*(1-g), label='Double sphere')
ax[1].plot(wv_TRS, mus_TRS[:,1], 'o', label='TRS_15mm')
ax[1].grid(True, linestyle=':')
ax[1].set_xlabel('nm')
ax[1].set_title(r"$\mu'_s$")
ax[1].legend()
plt.suptitle('TiO2')
plt.tight_layout()

# Plot Calibration phantom
fig, ax = plt.subplots(1,2, num=3, figsize=(9,3.5))
ax[0].plot(calib[:,0], calib[:,1], label='Reference')
ax[0].plot(wv_TRS, mua_TRS[:,4], 'o', label='TRS_15mm')
ax[0].plot(wv_TRS, mua_TRS[:,5], '*', label='TRS_30mm')
ax[0].grid(True, linestyle=':')
ax[0].set_xlabel('nm')
ax[0].set_ylabel(r'mm$^{{-1}}$')
ax[0].set_title(r'$\mu_a$')
ax[1].plot(calib[:,0], calib[:,2], label='Reference')
ax[1].plot(wv_TRS, mus_TRS[:,4], 'o', label='TRS_15mm')
ax[1].plot(wv_TRS, mus_TRS[:,5], '*', label='TRS_30mm')
ax[1].grid(True, linestyle=':')
ax[1].set_xlabel('nm')
ax[1].set_ylabel(r'mm$^{{-1}}$')
ax[1].set_title(r"$\mu'_s$")
ax[1].legend()
plt.suptitle('Calibration phantom')
plt.tight_layout()

# BATCH 4
# Plot A2O3
fig, ax = plt.subplots(1,2, num=4, figsize=(9,3.5))
# ax[0].plot(data_sfds['wv'][:,0], data_sfds['AlObaseTop'][:,0,0], label='SFDS')
ax[0].plot(data[1]['Wavelength [nm]'], data[1]['mua'], label='Double sphere')
ax[0].plot(wv_TRS, mua_TRS[:,0], 'o', label='TRS')
ax[0].grid(True, linestyle=':')
ax[0].set_xlabel('nm')
ax[0].set_ylabel(r'mm$^{{-1}}$')
ax[0].set_title(r'$\mu_a$')
# ax[1].plot(data_sfds['wv'][:,0], data_sfds['AlObaseTop'][:,0,1], label='SFDS')
ax[1].plot(data[1]['Wavelength [nm]'], data[1]['mus']*(1-g), label='Double sphere')
ax[1].plot(wv_TRS, mus_TRS[:,0], 'o', label='TRS')
ax[1].grid(True, linestyle=':')
ax[1].set_xlabel('nm')
ax[1].set_title(r"$\mu'_s$")
ax[1].legend()
plt.suptitle('Al2O3')
plt.tight_layout()

# Plot TiO2
fig, ax = plt.subplots(1,2, num=5, figsize=(9,3.5))
# ax[0].plot(data_sfds['wv'][:,0], data_sfds['TiObaseTop'][:,0,0], label='SFDS')
ax[0].plot(data[3]['Wavelength [nm]'], data[3]['mua'], label='Double sphere')
ax[0].plot(wv_TRS, mua_TRS[:,1], 'o', label='TRS_15mm')
ax[0].grid(True, linestyle=':')
ax[0].set_xlabel('nm')
ax[0].set_ylabel(r'mm$^{{-1}}$')
ax[0].set_title(r'$\mu_a$')
# ax[1].plot(data_sfds['wv'][:,0], data_sfds['TiObaseTop'][:,0,1], label='SFDS')
ax[1].plot(data[3]['Wavelength [nm]'], data[3]['mus']*(1-g), label='Double sphere')
ax[1].plot(wv_TRS, mus_TRS[:,1], 'o', label='TRS_15mm')
ax[1].grid(True, linestyle=':')
ax[1].set_xlabel('nm')
ax[1].set_title(r"$\mu'_s$")
ax[1].legend()
plt.suptitle('TiO2')
plt.tight_layout()