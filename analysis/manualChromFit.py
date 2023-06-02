# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:10:38 2023

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import addcopyfighandler

C_mat = np.zeros([3,5,3])

for _i, skip in enumerate([2, 8, 14, 20, 26]):
    for _j, col in enumerate(['C', 'M', 'W']):
        # Change the sheet_name, skiprows and usecols to the appropriate data origin
        df = pd.read_excel(r'C:\Users\luibe59\OneDrive - Linköpings universitet\PhD project\Pig study\New Pig Data.xlsx',
                           engine='openpyxl', sheet_name='Pig4 - wound2', skiprows=skip, nrows=5, usecols=col)
        
        # mua measurements and chromophores array
        # mua_meas = np.array([0.345087,0.254002,0.314276,0.258177,0.017585]).reshape((5,1)) # Change this
        mua_meas = np.array(df).reshape((5,1))
        
        # Hb spectrum, do not change
        mua_Hb = np.array([[39.56188073,17.34588508,20.96512948,14.40789132,4.793742697], # HbO
                           [66.32433649,19.15391672,23.67783192,14.84202762,8.331035023], # Hb
                           [30.3704019,19.44234976,18.50247754,14.38885067,9.366951109]]) # MetHb
        
        # Linear algebra fit
        inv_E = np.linalg.pinv(mua_Hb.T)
        C = inv_E @ mua_meas
        C_mat[:,_i,_j] = np.squeeze(C)


## Plotting scattering parameters
par_mat = np.zeros([4,5,3])
par_norm = np.zeros([2,5,3])
mua_mat = np.zeros([2,5,3])
mua_norm = np.zeros([1,5,3])
titles = ['PC4 - week0', 'PC4 - week1', 'PC4 - week2']  # adjust titles to correct pig / wound
labels = ['f{}'.format(x) for x in range(5)]
colors = ['Blues_r', 'Greens_r','Oranges_r']

########################################################
#  Notes on dataset:                                   #
#    - for plotting Hb (wound): skiprows=34, nrows=3   #
#    - for plotting mua (530nm): skiprows=39, nrows=1  #
#    - for plotting mua (skin): skiprows=133, nrows=1  #
#    - for plotting mus (wound): skiprows=48, nrows=2  #
#    - for plotting mus (skin): skiprows=130, nrows=2  #
#    - for plotting mus (error): skiprows=52, nrows=4  #
#    - for plotting mua (error): skiprows=41, nrows=2  #
########################################################

# Equalize axis
# ylim_a = [1e-5, 1e16]
# ylim_b = [-1.7, 6.1]
ylim_a = [1e-5, 1e10]
ylim_b = [0, 4.5]
ylim_mua = [0, 3]
wound = 1
for _i, col in enumerate(['C:G', 'M:Q', 'W:AA']):
    df = pd.read_excel(r'C:\Users\luibe59\OneDrive - Linköpings universitet\PhD project\Pig study\New Pig Data.xlsx',
                       engine='openpyxl', sheet_name='Pig4 - wound{}'.format(wound),
                       skiprows=41, nrows=2, usecols=col)
    # df2 = pd.read_excel(r'C:\Users\luibe59\OneDrive - Linköpings universitet\PhD project\Pig study\New Pig Data.xlsx',
    #                    engine='openpyxl', sheet_name='Pig4 - wound{}'.format(wound),
    #                    skiprows=125, nrows=1, usecols=col)
    
    if False: # mus
        par_mat[:,:,_i] = np.array(df)
        fig, ax = plt.subplots(1,2, figsize=(9,4.5), num=_i)
        cmap = plt.get_cmap(colors[_i])
        my_cmap = cmap(np.arange(0.2,0.7,0.1))
        # With errorbar
        # ax[0].bar(labels, par_mat[0,:,_i], yerr=par_mat[2,:,_i], color=my_cmap, width=1,
        #           ecolor='k', capsize=5)
        # Without errorbar
        ax[0].bar(labels, par_mat[0,:,_i], color=my_cmap, width=1)
        ax[0].set_yscale('log')
        ax[0].set_title('A')
        # ax[0].set_ylim(ylim_a)
        # With errorbar
        # ax[1].bar(labels, par_mat[1,:,_i], yerr=par_mat[3,:,_i], color=my_cmap, width=1,
        #           ecolor='k', capsize=5)
        # Without errorbar
        ax[1].bar(labels, par_mat[1,:,_i], color=my_cmap, width=1)
        ax[1].set_title('B')
        # ax[1].set_ylim(ylim_b)
        fig.suptitle('Wound {} - week{}'.format(wound, _i))
        plt.tight_layout()
    
    if True:  # mua
        mua_mat[:,:,_i] = np.array(df)
        # mua_norm[:,:,_i] = np.array(df2)
        fig, ax = plt.subplots(1,1, figsize=(6,4), num=_i)
        cmap = plt.get_cmap(colors[_i])
        my_cmap = cmap(np.arange(0.2,0.7,0.1))
        # With errorbar
        ax.bar(labels, mua_mat[0,:,_i], color=my_cmap, width=1, yerr=mua_mat[1,:,_i],
               ecolor='k', capsize=5)
        # Without errorbar
        ax.bar(labels, mua_mat[0,:,_i], color=my_cmap, width=1)
        # ax.set_yscale('log')
        ax.set_title(r'$\mu_a (530nm)$')
        ax.set_ylabel(r'mm$^{-1}$')
        ax.set_ylim(ylim_mua)
        fig.suptitle('Wound {} - week{}'.format(wound, _i))
        plt.tight_layout()

if False:  # mus    
    figb, axb = plt.subplots(1,2, figsize=(8,4), num=67)
    x = np.array([0, 1.1, 2.2])
    X = list(f'week{x}' for x in range(3))
    # Normalized
    # axb[0].bar(x-0.25, par_mat[0,1,:]/par_norm[0,1,:], width=0.5)
    # axb[0].bar(x+0.25, par_mat[0,2,:]/par_norm[0,2,:], width=0.5)
    axb[0].bar(x-0.25, par_mat[0,1,:], width=0.5, yerr=par_mat[2,1,:], ecolor='k', capsize=5)
    axb[0].bar(x+0.25, par_mat[0,2,:], width=0.5, yerr=par_mat[2,2,:], ecolor='k', capsize=5)
    axb[0].set_yscale('log')
    # axb[0].set_ylabel('% (log scale)')
    axb[0].set_xticks(ticks=x)
    axb[0].set_xticklabels(X)
    axb[0].legend(['f1', 'f2'])
    axb[0].set_title('A')
    axb[0].set_ylim(ylim_a)
    axb[0].set_axisbelow(True)
    axb[0].grid(True, axis='y', linestyle=':')
    # Normalized
    # axb[1].bar(x-0.25, par_mat[1,1,:]/par_norm[1,1,:]*100, width=0.5)
    # axb[1].bar(x+0.25, par_mat[1,2,:]/par_norm[1,2,:]*100, width=0.5)
    axb[1].bar(x-0.25, par_mat[1,1,:], width=0.5, yerr=par_mat[3,1,:], ecolor='k', capsize=5)
    axb[1].bar(x+0.25, par_mat[1,2,:], width=0.5, yerr=par_mat[3,2,:], ecolor='k', capsize=5)
    # axb[1].set_ylabel('%')
    axb[1].set_xticks(ticks=x)
    axb[1].set_xticklabels(X)
    axb[1].set_title('B')
    axb[1].set_ylim(ylim_b)
    axb[1].set_axisbelow(True)
    axb[1].grid(True, axis='y', linestyle=':')
    figb.suptitle('PC{}'.format(wound))
    plt.tight_layout()
    
if True:  # mua  
    figb, axb = plt.subplots(1,1, figsize=(6,4), num=66)
    x = np.array([0, 1.1, 2.2])
    X = list(f'week{x}' for x in range(3))
    # normalized
    # axb.bar(x-0.25, mua_mat[0,1,:]/mua_norm[0,1,:]*100, width=0.5)
    # axb.bar(x+0.25, mua_mat[0,2,:]/mua_norm[0,2,:]*100, width=0.5)
    axb.bar(x-0.25, mua_mat[0,1,:], width=0.5, yerr=mua_mat[1,1,:], ecolor='k', capsize=5)
    axb.bar(x+0.25, mua_mat[0,2,:], width=0.5, yerr=mua_mat[1,2,:], ecolor='k', capsize=5)
    # axb.set_yscale('log')
    axb.set_xticks(ticks=x)
    axb.set_xticklabels(X)
    axb.legend(['f1', 'f2'])
    axb.set_title(r'$\mu_a (530nm)$')
    axb.set_ylabel(r'mm$^{-1}$')
    # axb.set_ylabel('%')
    axb.set_ylim(ylim_mua)
    axb.set_axisbelow(True)
    axb.grid(True, axis='y', linestyle=':')
    figb.suptitle('PC{}'.format(wound))
    plt.tight_layout()