# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:06:11 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Script to fit mus' to a power law of the kind A * lambda^(-b)
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
#from matplotlib.colors import LogNorm
from matplotlib.backend_bases import MouseButton

from sfdi.common.getPath import getPath
from sfdi.common.colourbar import colourbar

def fit_fun(lamb, a, b):
    """Exponential function to fit data to"""
    return a * np.power(lamb, -b)

def mad(x,scale=1.4826,axis=None):
    """Median assoulte difference (since Scipy does not implement it anymore).
    This is preferred as a measure of dispersion, since it is more robust to
    outliers. The scale factor is to make it comparable with standard deviation
    (for normal distribution)"""
    med = np.nanmedian(x,axis=axis)
    return np.nanmedian(np.abs(x-med),axis=axis)*scale

def onclick(event):
    global coord
    if event.name == 'button_press_event':
        if event.xdata != None:
            coord['x'] = int(round(event.xdata))
            coord['y'] = int(round(event.ydata))
            # print('x: {}, y: {}'.format(coord['x'],coord['y']))
        
    if event.button == MouseButton.RIGHT:
        coord['x'] = None
        coord['y'] = None

# select data path
path = getPath('Select data path')

wv = np.array([458, 520, 536, 556, 626])  # wavelengts (nm)

op_fit_maps = []
param_maps = []
files = [x for x in os.listdir(path) if x.endswith('.npz') and 'calR' not in x]
files.sort()
titles = [y.split('_')[1] for y in files]  # file name
for file in files:
    data = np.load('{}/{}'.format(path, file))
    op_fit_maps.append(data['op_fit_maps'])

for _a, op_map in enumerate(op_fit_maps[:3], start=1):
    print('Fitting dataset {}...'.format(_a))
    p_map = np.zeros((op_map.shape[0], op_map.shape[1], 2), dtype=float)
    for _i in range(op_map.shape[0]):
        for _j in range(op_map.shape[1]):
            try:
                temp, _ = curve_fit(fit_fun, wv, op_map[_i,_j,:,1],
                                    p0=[1e-3, 1], method='lm')
            except RuntimeError:
                continue
            p_map[_i, _j, :] = temp
    param_maps.append(p_map)


# test
N = 0  # data set
global coord
coord = {'x': 0, 'y': 0}

fig, ax = plt.subplots(2,2, num=100)
im1 = ax[0,0].imshow(param_maps[N][:,:,0])
colourbar(im1)
ax[0,0].set_title('a')
im2 = ax[0,1].imshow(param_maps[N][:,:,1])
colourbar(im2)
ax[0,1].set_title('b')
ax[1,0].plot(wv, op_fit_maps[N][coord['y'], coord['x'],:,1], '-b', label='data')
mus = fit_fun(wv, param_maps[N][coord['y'], coord['x'],0],
                  param_maps[N][coord['y'], coord['x'],1])
ax[1,0].plot(wv, mus, 'or', linestyle='solid', label='fitted')
ax[1,0].set_xlabel('wavelength (nm)')
ax[1,0].legend()
ax[1,0].grid(True, linestyle=':')
ax[1,1].axis('off')
# Interactive plot
if True:  # put True to execute
    plt.pause(0.1)
    WV = np.arange(450,650)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    while coord['x'] is not None:
        # plt.figure(100, figsize=(5,3))
        ax[1,0].clear()
        ax[1,0].plot(wv, op_fit_maps[N][coord['y'], coord['x'],:,1], '-b', label='data')
        mus = fit_fun(wv, param_maps[N][coord['y'], coord['x'],0],
                          param_maps[N][coord['y'], coord['x'],1])
        ax[1,0].plot(wv, mus, 'or', linestyle='solid', label='fitted')
        ax[1,0].grid(True, linestyle=':')
        plt.pause(0.1)
    fig.canvas.mpl_disconnect(cid)

# plt.figure(400)
# plt.plot(wv, op_fit_maps[1][20,53,:,1],'-b')  # data
# mus = fit_fun(wv, param_maps[1][20,53,0], param_maps[1][20,53,1])
# plt.plot(wv, mus, '*r')  # fitted function

# for _i, param in enumerate(param_maps):
#     fig, ax = plt.subplots(1, 2, num=_i, figsize=(15,5))
#     im1 = ax[0].imshow(param_maps[_i][:,:,0], cmap='viridis', vmax=10)
#     colourbar(im1)
#     im2 = ax[1].imshow(param_maps[_i][:,:,1], cmap='magma', vmax=3)
#     colourbar(im2)
#     plt.tight_layout()