# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:44:51 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
from matplotlib import pyplot as plt


def chromPlot(chrom_map,name,par):
    """"A function to plot the chromophores distribution map"""
    # TODO: fix doc
    
    titles = ['',r'HbO$_2$','Hb',r'H$_2$O','lipid','melanin'] # chromophores names. the first is empty to
                                                              # respect the naming convention
    titles = [titles[i] for i in par['chrom_used']] # Only keep used chromophores
    
    nrows = ((chrom_map.shape[-1]-1)//2) + 1
    if chrom_map.shape[-1] == 1:
        ncols = 1
    else:
        ncols = 2
    
    fig,ax = plt.subplots(nrows,ncols)
    fig.set_size_inches(5*ncols,3*nrows)
    plt.suptitle(name,fontsize=12)
    
    for i in range(chrom_map.shape[-1]):
        
        j = i//2
        k = i%2
        
        mp = ax[j,k].imshow(chrom_map[:,:,i],cmap='magma')
        ax[j,k].set_title(titles[i],fontsize=10)
        fig.colorbar(mp,ax=ax[j,k])

    plt.tight_layout()
if __name__ == '__main__':
    
    for cm,name in zip(chrom_map,names):
        chromPlot(cm,name,par)