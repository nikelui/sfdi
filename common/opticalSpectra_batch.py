# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:09:58 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

import numpy as np
import numpy.ma as mask

from matplotlib import pyplot as plt
from sfdi.crop import crop

def mad(x,scale=1.4826,axis=None):
    """Median assoulte difference (since Scipy does not implement it anymore).
    This is preferred as a measure of dispersion, since it is more robust to
    outliers. The scale factor is to make it comparable with standard deviation
    (for normal distribution)"""
    med = np.nanmedian(x,axis=axis)
    return np.nanmedian(np.abs(x-med),axis=axis)*scale

def opticalSpectra(op_fit_maps,par,names,save=False,outliers=False):
    """Plot mean and standard deviation of whole optical properties map (one for file)"""
    
    colours=['orange','red','darkred',
             'skyblue','royalblue','darkblue',
             'lightgreen','limegreen','darkgreen'] # colour scheme to draw lines
        
    opt_ave = []
    opt_std = []
    for opt_map in op_fit_maps: # loop through data
        # Convert to maskedArray for convenience
        opt_map = mask.masked_array(opt_map) # should already be set to nomask
        ## Remove outliers
        if outliers:
            ## New approach: create a masked array (to not alter original data)
            # First: calculate the thresholds using median and MAD:
            med = np.nanmedian(opt_map,axis=(0,1))
            MAD = mad(opt_map,axis=(0,1))
            # loop over wavelength and OP axis
            for wv in range(len(MAD)):
                for i in range(len(MAD[0])):
                    # mask if the distance from the median value is higher than 15*MAD
                    idx = np.where(abs(opt_map[:,:,wv,i] - med[wv,i]) > 15*MAD[wv,i])
                    for x in range(len(idx[0])):
                        opt_map[idx[0],idx[1],wv,i] = mask.masked
        
        opt_ave.append(np.nanmean(opt_map,axis=(0,1)))
        opt_std.append(np.nanstd(opt_map,axis=(0,1)))
            
    fig,ax = plt.subplots(1,2,figsize=(9,4))
    
    for i in range(len(opt_ave)):
        ax[0].errorbar(par['wv'],opt_ave[i][:,0],yerr=opt_std[i][:,0],fmt='D',
          linestyle='solid',capsize=5,markersize=3,label=names[i],color=colours[i%9])
    ax[0].set_title('Absorption coefficient ($\mu_A$)')
    ax[0].grid(True,which='both',linestyle=':')
    ax[0].set_xlabel('wavelength (nm)')
    #ax[0].set_ylim(-0.01,0.01)
    #ax[0].legend()

    for i in range(len(opt_ave)):
        ax[1].errorbar(par['wv'],opt_ave[i][:,1],yerr=opt_std[i][:,1],fmt='D',
          linestyle='solid',capsize=5,markersize=3,label=names[i],color=colours[i%9])
    ax[1].set_title('Scattering coefficient ($\mu_S´$)')
    ax[1].grid(True,which='both',linestyle=':')
    ax[1].set_xlabel('wavelength (nm)')
    #ax[1].set_ylim(0,10)
    ax[1].legend()

    plt.tight_layout()
    
    if save:
        plt.savefig('figure%d.png'%save, )
    
    return opt_ave,opt_std


if __name__ == '__main__':
    opt_ave,opt_std = opticalSpectra(op_fit_maps,par,names,outliers=True)