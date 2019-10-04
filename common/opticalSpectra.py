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
from matplotlib.widgets import RadioButtons
import matplotlib.gridspec as gridspec
from sfdi.crop import crop

def opticalSpectra(Im,op_fit_maps,par,save=False):
    """Select multiple ROI on the reflectance map and calculate an average of the optical properties to plot.
    """
    ## Put callbacks here
    def call_mua(label):
        label_dict = {'B':0,'GB':1,'G':2,'GR':3,'R':4}
        data_a = op_fit_maps[:,:,label_dict[label],0]
        data_s = op_fit_maps[:,:,label_dict[label],1]
        im1.set_data(data_a)
        if(np.max(op_fit_maps[:,:,:,0]) > 0.5):
            im1.set_clim(vmax=0.5)
        vmax = np.max(op_fit_maps[:,:,:,0])
        if vmax > 0.5:
            vmax = 0.4 # If there are some outlier values, limit the range
        im2.set_data(data_s)
        plt.draw()
    
#    def call_mus(label):
#        label_dict = {'B':0,'GB':1,'G':2,'GR':3,'R':4}
#        data = op_fit_maps[:,:,label_dict[label],1]
#        im2.set_data(data)
#        plt.draw()
    
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
    
    # manually define axis
    fig = plt.figure(constrained_layout=True,figsize=(10,6.5))
    spec = gridspec(ncols=3,nrows=2,width_ratios=[1,0.5,1])
    
    ax1 = fig.add_subplot(spec[0,0])
    ax2 = fig.add_subplot(spec[0,1])
    ax3 = fig.add_subplot(spec[0,2])
    ax4 = fig.add_subplot(spec[1,0])
    ax5 = fig.add_subplot(spec[1,1])
    ax6 = fig.add_subplot(spec[1,2])
    
    ax = np.array([[ax1,ax2,ax3],[ax4,ax5,ax6]]) # this way the old scheme is kept
    
    #fig,ax = plt.subplots(2,3,figsize=(10,6.5))
    
    vmax = np.max(op_fit_maps[:,:,:,0])
    if vmax > 0.5:
        vmax = 0.4 # If there are some outlier values, limit the range    
    im1 = ax[0,0].imshow(op_fit_maps[:,:,0,0],cmap='magma',vmin=0,vmax=vmax)
    ax[0,0].set_title('Absorption coefficient ($\mu_A$)')
    ax[0,0].get_xaxis().set_visible(False)
    ax[0,0].get_yaxis().set_visible(False)
    fig.colorbar(im1,ax=ax[0,0])
    
    radio1 = RadioButtons(ax[0,1], labels=('B','GB','G','GR','R'))
    radio1.on_clicked(call_mua)
    
    vmax = np.max(op_fit_maps[:,:,:,1])
    if vmax > 10:
        vmax = 10 # If there are some outlier values, limit the range    
    im2 = ax[0,2].imshow(op_fit_maps[:,:,0,1],cmap='magma',vmin=0,vmax=vmax)
    ax[0,2].set_title('Scattering coefficient ($\mu_S´$)')
    ax[0,2].get_xaxis().set_visible(False)
    ax[0,2].get_yaxis().set_visible(False)
    fig.colorbar(im2,ax=ax[0,2])

    ax[1,1].axis('off')
    
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
        
        ax[1,2].errorbar(par['wv'],opt_ave[i,:,1],yerr=opt_std[i,:,1],
              fmt='D',linestyle=lin,color=colours[i],capsize=5,markersize=3)
        ax[1,2].set_xlabel('wavelength (nm)')
        ax[1,2].grid(True,which='both',linestyle=':')
        
    #plt.tight_layout()
    
    if save:
        plt.savefig('figure%d.png'%save, )
    
    return opt_ave,opt_std,radio1


if __name__ == '__main__':
    cropped = crop(cal_R[:,:,0,0],ROI)
    opt_ave,opt_std,radio = opticalSpectra(cropped,op_fit_maps,par)