# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:53:46 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Improved colorbar function (better fit to axes dimensions)
"""
from mpl_toolkits.axes_grid1 import make_axes_locatable

def colourbar(mappable):
    """
    Parameters
    ----------
    mappable : matplotlib plot object

    Returns
    -------
    matplotlib colorbar object
    """
    if (mappable.colorbar is not None):
        mappable.colorbar.remove()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)
