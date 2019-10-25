# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:44:51 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import numpy.ma as mask


def mad(x,scale=1.4826,axis=None):
    """Median assoulte difference (since Scipy does not implement it anymore).
    This is preferred as a measure of dispersion, since it is more robust to
    outliers. The scale factor is to make it comparable with standard deviation
    (for normal distribution)"""
    med = np.nanmedian(x,axis=axis)
    return np.nanmedian(np.abs(x-med),axis=axis)*scale

def colourbar(mappable):
    """Improved colorbar function. Fits well to the axis dimension."""
    if (mappable.colorbar is not None):
        mappable.colorbar.remove()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

def chromPlot(chrom_map,name,par,outliers=True):
    """"A function to plot the chromophores distribution map.
@Inputs
    - chrom_map: chromophores map, given by chromFit
    - name: filename of the phantom, to put in the figure title
    - par: dictionary containing processing parameters
@Return
    - chrom_map: will be a MaskedArray with the outlier values masked
"""
    if outliers:
        # Convert to MaskedArray for convenience
        chrom_map = mask.masked_array(chrom_map)
    
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
        
        if outliers:
            # Mask outliers
            # First: calculate the thresholds using median and MAD:
            med = np.nanmedian(chrom_map,axis=(0,1))
            MAD = mad(chrom_map,axis=(0,1))
            # mask if the distance from the median value is higher than 15*MAD
            idx = np.where(abs(chrom_map - med) > 15*MAD)
            for x in range(len(idx[0])):
                chrom_map[idx[0],idx[1]] = mask.masked
        
        mp = ax.imshow(chrom_map,cmap='viridis')
        ax.set_title(titles[0],fontsize=10)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        current_cmap = cm.get_cmap('viridis')
        current_cmap.set_bad(color='red') # masked values colour
        
        #fig.colorbar(mp,ax=ax)
        colourbar(mp)
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
        fig.set_size_inches(4*ncols,2.5*nrows)
        plt.suptitle(name,fontsize=12) # Title of the entire figure
        
        for i in range(chrom_map.shape[-1]):
            # Subplots indices are in 2D
            j = i//2
            k = i%2
            
            if outliers:
                # Mask outliers
                # First: calculate the thresholds using median and MAD:
                med = np.nanmedian(chrom_map[:,:,i],axis=(0,1))
                MAD = mad(chrom_map[:,:,i],axis=(0,1))
                # mask if the distance from the median value is higher than 15*MAD
                idx = np.where(abs(chrom_map[:,:,i] - med) > 15*MAD)
                for x in range(len(idx[0])):
                    chrom_map[idx[0],idx[1],i] = mask.masked
            
            mp = ax[j,k].imshow(chrom_map[:,:,i],cmap='viridis')
            ax[j,k].set_title(titles[i],fontsize=10)
            ax[j,k].get_xaxis().set_visible(False)
            ax[j,k].get_yaxis().set_visible(False)
            
            current_cmap = cm.get_cmap('viridis')
            current_cmap.set_bad(color='red') # masked values colour
            
            #fig.colorbar(mp,ax=ax[j,k])
            colourbar(mp)
        
        if (chrom_map.shape[-1] % 2 == 1 and chrom_map.shape[-1] > 1):
            ax[-1,-1].axis('off') # Delete empty axis if chrom. number is odd and > 1   
        plt.tight_layout()      
    return chrom_map


if __name__ == '__main__':  
#    for cm,name in zip(chrom_map,names):
#        chromPlot(cm,name,par)
    #chromPlot(chrom_map,name.split('/')[-1],par)
    chrom_map = chromPlot(chrom_map,'aaa',par)
