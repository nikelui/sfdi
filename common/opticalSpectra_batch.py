# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:09:58 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

import numpy as np

from matplotlib import pyplot as plt
from sfdi.crop import crop

def opticalSpectra(op_fit_maps,par,names,save=False):
    """Plot mean and standard deviation of whole optical properties map (one for file)"""
    
    colours=['b','r',(0,1,0),'k',(0,1,1),(1,0,1),(0.9,0.9,0)] # colour scheme to draw lines
    if len(op_fit_maps) > 7:
        colours = colours*2 # we assume no more than 7 lines to plot
    
    opt_ave = []
    opt_std = []
    for opt_map in op_fit_maps: # loop through data
        opt_ave.append(np.mean(opt_map,axis=(0,1)))
        opt_std.append(np.std(opt_map,axis=(0,1)))
            
    fig,ax = plt.subplots(1,2,figsize=(9,4))
    
    for i in range(len(opt_ave)):
        ax[0].errorbar(par['wv'],opt_ave[i][:,0],yerr=opt_std[i][:,0],fmt='D',
          linestyle='solid',capsize=5,markersize=3,label=names[i])
    ax[0].set_title('Absorption coefficient ($\mu_A$)')
    ax[0].grid(True,which='both',linestyle=':')
    ax[0].set_xlabel('wavelength (nm)')
    ax[0].set_ylim(-0.01,0.01)
    #ax[0].legend()

    for i in range(len(opt_ave)):
        ax[1].errorbar(par['wv'],opt_ave[i][:,1],yerr=opt_std[i][:,1],fmt='D',
          linestyle='solid',capsize=5,markersize=3,label=names[i])
    ax[1].set_title('Scattering coefficient ($\mu_S´$)')
    ax[1].grid(True,which='both',linestyle=':')
    ax[1].set_xlabel('wavelength (nm)')
    ax[1].set_ylim(0,10)
    ax[1].legend()

    plt.tight_layout()
    
    if save:
        plt.savefig('figure%d.png'%save, )
    
    return opt_ave,opt_std


if __name__ == '__main__':
    opt_ave,opt_std = opticalSpectra(op_fit_maps,par,names)