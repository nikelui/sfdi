# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:09:58 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from sfdi.crop import crop

def opticalSpectra(Im,op_fit_maps,par,save=False):
    """Select multiple ROI on the reflectance map and calculate an average of the optical properties to plot.
    """
    
    cv.namedWindow('Select ROIs',cv.WINDOW_NORMAL)
    cv.resizeWindow('Select ROIs',(Im.shape[1],Im.shape[0]))
    ROIs = cv.selectROIs('Select ROIs',Im,False,False) # press enter to save the ROI, esc to terminate
    cv.destroyAllWindows()
    
    colours=['b','r',(0,1,0),'k',(0,1,1),(1,0,1),(0.9,0.9,0)] # colour scheme to draw lines
    if len(ROIs) > 7:
        colours = colours*2 # we assume no more than 14 ROIs
    
    opt_ave = np.zeros((len(ROIs),len(par['wv']),2),dtype=float)
    opt_std = np.zeros((len(ROIs),len(par['wv']),2),dtype=float)
    for i in range(len(ROIs)): # calculate average and std inside ROIs
        opt_ave[i,:,:] = np.mean(crop(op_fit_maps,ROIs[i,:]//par['binsize']),axis=(0,1))
        opt_std[i,:,:] = np.std(crop(op_fit_maps,ROIs[i,:]//par['binsize']),axis=(0,1))
    
    fig,ax = plt.subplots(2,2,figsize=(10,6.5))
    
    im = ax[0,0].imshow(op_fit_maps[:,:,0,0],cmap='magma')
    ax[0,0].set_title('Absorption coefficient ($\mu_A$)')
    ax[0,0].get_xaxis().set_visible(False)
    ax[0,0].get_yaxis().set_visible(False)
    fig.colorbar(im,ax=ax[0,0])
    
    im = ax[0,1].imshow(op_fit_maps[:,:,0,1],cmap='magma')
    ax[0,1].set_title('Scattering coefficient ($\mu_S´$)')
    ax[0,1].get_xaxis().set_visible(False)
    ax[0,1].get_yaxis().set_visible(False)
    fig.colorbar(im,ax=ax[0,1])
    
    b = par['binsize'] # to correctly rescale the rectangles
    
    for i in range(len(ROIs)):
        lin = 'solid'
        if i >= 7:
            lin = 'dashed'
        
        rect = Rectangle((ROIs[i,0]//b,ROIs[i,1]//b),ROIs[i,2]//b,ROIs[i,3]//b,fill=False,
                         edgecolor=colours[i],facecolor=None,linewidth=2,linestyle=lin)
        ax[0,0].add_patch(rect)
        
        rect = Rectangle((ROIs[i,0]//b,ROIs[i,1]//b),ROIs[i,2]//b,ROIs[i,3]//b,fill=False,
                         edgecolor=colours[i],facecolor=None,linewidth=2,linestyle=lin)
        ax[0,1].add_patch(rect)
        
        
        ax[1,0].errorbar(par['wv'],opt_ave[i,:,0],yerr=opt_std[i,:,0],
              fmt='D',linestyle=lin,color=colours[i],capsize=5,markersize=3)
        ax[1,0].set_xlabel('wavelength (nm)')
        ax[1,0].grid(True,which='both',linestyle=':')
        
        ax[1,1].errorbar(par['wv'],opt_ave[i,:,1],yerr=opt_std[i,:,1],
              fmt='D',linestyle=lin,color=colours[i],capsize=5,markersize=3)
        ax[1,1].set_xlabel('wavelength (nm)')
        ax[1,1].grid(True,which='both',linestyle=':')

    plt.tight_layout()
    
    if save:
        plt.savefig('figure%d.png'%save, )
    
    return opt_ave,opt_std


if __name__ == '__main__':
    cropped = crop(cal_R[:,:,0,0],ROI)
    opt_ave,opt_std = opticalSpectra(cropped,op_fit_maps,par)