# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:34:08 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Script to plot mua, mus' maps at 5 wavelengths
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from zipfile import BadZipFile
from sfdi.common.colourbar import colourbar
import addcopyfighandler

phantom_path = "C:\\Users\\luibe59\\Documents\\sfdi\\common\\phantoms\\"
phantoms = ['TS1', 'TS1_SFDS', 'TS2', 'TS2_SFDS']

ph = []
for name in phantoms:
    temp = np.genfromtxt('{}{}.txt'.format(phantom_path, name), delimiter='\t')
    ph.append(temp[:,:-1])

fig, ax = plt.subplots(1, 2, figsize=(10,4))
ax[0].plot(ph[0][:,0], ph[0][:,1], 'ob', markerfacecolor="None", label='TS1_old')
ax[0].plot(ph[1][:,0], ph[1][:,1], '-b', label='TS1_new')
ax[0].plot(ph[2][:,0], ph[2][:,1], 'or', markerfacecolor="None", label='TS2_old')
ax[0].plot(ph[3][:,0], ph[3][:,1], '-r', label='TS2_new')
ax[0].grid(True, linestyle=':')
ax[0].set_title(r'$\mu_a$')

ax[1].plot(ph[0][:,0], ph[0][:,2], 'ob', markerfacecolor="None", label='TS1_old')
ax[1].plot(ph[1][:,0], ph[1][:,2], '-b', label='TS1_new')
ax[1].plot(ph[2][:,0], ph[2][:,2], 'or', markerfacecolor="None", label='TS2_old')
ax[1].plot(ph[3][:,0], ph[3][:,2], '-r', label='TS2_new')
ax[1].grid(True, linestyle=':')
ax[1].set_title(r"$\mu'_s$")
plt.legend()
plt.tight_layout()

fnames = [x for x in os.listdir(os.getcwd()) if x.endswith('npz')]
data = []
to_remove = []

for _i, name in enumerate(fnames):
    temp = np.load(name)
    try:
        data.append(temp['op_fit_maps'])
    except BadZipFile:
        print('Error loading {} file'.format(name))
        to_remove.append(name)
        
for name in to_remove:
    fnames.remove(name)
    
vmax = [1, 0.7, 0.7, 0.7, 0.05]

for _i, op_map in enumerate(data):
    fig, ax = plt.subplots(5, 2, figsize=(6,10), dpi=100, num=_i)
    fig.suptitle('{}'.format(fnames[_i].split('_')[1]))
    for _j,_k in enumerate([0,3,4,5,8]):
        im1 = ax[_j,0].imshow(op_map[:,:,_k,0], cmap='magma', vmin=0, vmax=vmax[_j])
        ax[_j,0].axis('off')
        c1 = colourbar(im1)
        im2 = ax[_j,1].imshow(op_map[:,:,_k,1], cmap='magma', vmin=0, vmax=4)
        ax[_j,1].axis('off')
        c2 = colourbar(im2)
        if _j == 0:
            ax[_j,0].set_title(r'Absorption $\mu_a$')
            ax[_j,1].set_title(r"Scattering $\mu'_s$")
    plt.tight_layout()
    plt.savefig('{}.png'.format(fnames[_i].split('_')[1]), dpi=100, pad_inches=0)