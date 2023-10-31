# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 22:35:28 2023

@author: luibe59
"""

import numpy as np
from matplotlib import pyplot as plt
import addcopyfighandler

fig, ax = plt.subplots(1,2, figsize=(9,4))
# absorption coefficient
ax[0].plot(asd_sfds['wv'], np.mean(asd_sfds['TiObaseTop'][:,:3,0], axis=1),
           '-g', label=r'TiO$_2$')  # TiO2 - SFDS
ax[0].errorbar(wv, asd_mean['TiObaseTop'][:,0,0],
               yerr=asd_std['TiObaseTop'][:,0,0], marker='o', color='g',
               markerfacecolor='None', linestyle='None', markersize=6, capsize=5)  # TiO2 - SFDI

ax[0].plot(asd_sfds['wv'], np.mean(asd_sfds['AlObaseTop'][:,:3,0], axis=1),
           '-r', label=r'Al$_2$O$_3$')  # Al2O3 - SFDS
ax[0].errorbar(wv, asd_mean['AlObaseTop'][:,0,0],
               yerr=asd_std['AlObaseTop'][:,0,0], marker='o', color='r',
               markerfacecolor='None', linestyle='None', markersize=6, capsize=5)  # Al2O3 - SFDI

ax[0].set_xlim([450, 650])
ax[0].set_ylim([0, 0.05])
ax[0].grid(True, linestyle=':')
ax[0].set_xlabel('wavelength (nm)')
ax[0].set_ylabel(r'mm$^{-1}$')
ax[0].set_title(r'$\mu_a$')

# scattering coefficient
ax[1].plot(asd_sfds['wv'], np.mean(asd_sfds['TiObaseTop'][:,:3,1], axis=1),
           '-g', label=r'TiO$_2$')  # TiO2 - SFDS
ax[1].errorbar(wv, asd_mean['TiObaseTop'][:,0,1],
               yerr=asd_std['TiObaseTop'][:,0,1], marker='o', color='g',
               markerfacecolor='None', linestyle='None', markersize=6, capsize=5)  # TiO2 - SFDI

ax[1].plot(asd_sfds['wv'], np.mean(asd_sfds['AlObaseTop'][:,:3,1], axis=1),
           '-r', label=r'Al$_2$O$_3$')  # Al2O3 - SFDS
ax[1].errorbar(wv, asd_mean['AlObaseTop'][:,0,1],
               yerr=asd_std['AlObaseTop'][:,0,1], marker='o', color='r',
               markerfacecolor='None', linestyle='None', markersize=6, capsize=5)  # Al2O3 - SFDI

ax[1].set_xlim([450, 650])
ax[1].set_ylim([0, 4])
ax[1].grid(True, linestyle=':')
ax[1].set_xlabel('wavelength (nm)')
ax[1].set_ylabel(r'mm$^{-1}$')
ax[1].set_title(r"$\mu'_s$")
ax[1].legend(loc='upper right')

plt.tight_layout()