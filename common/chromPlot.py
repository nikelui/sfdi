# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:44:51 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
from matplotlib import pyplot as plt
import numpy as np


def chromPlot(chrom_map,name,par):
    """"A function to plot the chromophores distribution map"""
    # TODO: fix doc
    
    titles = ['',r'HbO$_2$','Hb',r'H$_2$O','lipid','melanin'] # chromophores names. the first is empty to
                                                              # respect the naming convention
    titles = [titles[i] for i in par['chrom_used']] # Only keep used chromophores
    
    # If no chromophore are present, do nothing
    if len(par['chrom_used']) == 0:
        return
    
    ## Simple case: only one chromophore
    if len(par['chrom_used']) == 1:
        fig,ax = plt.subplots(1,1)
        fig.set_size_inches(5,4)
        plt.suptitle(name,fontsize=12) # Title of the entire figure
        mp = ax.imshow(chrom_map,cmap='viridis')
        ax.set_title(titles[0],fontsize=10)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.colorbar(mp,ax=ax)
        plt.tight_layout()    
    
    ## If chromophores are 2 or more, plot in a grid (two columns)
    # These are the dimensions of the subplots
    else:
        nrows = ((chrom_map.shape[-1]-1)//2) + 1
        if chrom_map.shape[-1] == 1:
            ncols = 1
        else:
            ncols = 2
    
        fig,ax = plt.subplots(nrows,ncols)
        if ax.ndim == 1:
            ax = ax.reshape((1,-1)) # force 2D array
        fig.set_size_inches(4.5*ncols,2.5*nrows)
        plt.suptitle(name,fontsize=12) # Title of the entire figure
        
        for i in range(chrom_map.shape[-1]):
            # Subplots indices are in 2D
            j = i//2
            k = i%2
            
            mp = ax[j,k].imshow(chrom_map[:,:,i],cmap='viridis')
            ax[j,k].set_title(titles[i],fontsize=10)
            ax[j,k].get_xaxis().set_visible(False)
            ax[j,k].get_yaxis().set_visible(False)
            fig.colorbar(mp,ax=ax[j,k])
        
        if (chrom_map.shape[-1] % 2 == 1 and chrom_map.shape[-1] > 1):
            ax[-1,-1].axis('off') # Delete empty axis if chrom. number is odd and > 1   
        plt.tight_layout()


if __name__ == '__main__':  
#    for cm,name in zip(chrom_map,names):
#        chromPlot(cm,name,par)
    chromPlot(chrom_map,name.split('/')[-1],par)