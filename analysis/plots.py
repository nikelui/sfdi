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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from zipfile import BadZipFile
from sfdi.common.colourbar import colourbar


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