# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:31:35 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
#import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

def stackPlot_old(data,cmap='viridis'):
    """Plot Reflectance data in a tabular form (rows: wavelengths, columns: spatial frequencies)"""
    dim = data.shape
    
    #mpl.rcParams['savefig.pad_inches'] = 0
    (fig,ax) = plt.subplots(dim[2],dim[3])
    
    plt.tight_layout()
    fig.set_size_inches(640/64,480*5/4/64)
    fig.subplots_adjust(wspace=0, hspace=0,right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.05, 0.02, 0.9])
    plt.autoscale(tight=True)
    
    M = np.max(data)
    
    for i in range(dim[2]):
        for j in range(dim[3]):
            im = ax[i,j].imshow(data[:,:,i,j]/M,cmap=cmap,vmin=0,vmax=1)
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
    fig.colorbar(im,cax=cbar_ax)

def stackPlot(data,cmap='viridis'):
    """Plot Reflectance data in a tabular form (rows: wavelengths, columns: spatial frequencies)"""
    dim = data.shape
    temp = np.zeros((dim[0]*dim[2],dim[1]*dim[3]),dtype='float')
    
    for i in range(dim[2]):
        for j in range(dim[3]):
            temp[i*dim[0]:(i+1)*dim[0],j*dim[1]:(j+1)*dim[1]] = data[:,:,i,j]
    plt.figure(num=100,figsize=(7.5,6))
    plt.imshow(temp,cmap=cmap,vmin=0,vmax=1)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    plt.colorbar()
    
    plt.tight_layout()
    

if __name__ == '__main__':
    stackPlot(cal_R,'viridis')