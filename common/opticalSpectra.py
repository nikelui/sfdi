# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:09:58 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

import cv2 as cv
import numpy as np
import numpy.ma as mask

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RadioButtons, AxesWidget
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sfdi.crop import crop


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
    

class MyRadioButtons(RadioButtons):
    """Custom radio button class from stackoverflow [https://stackoverflow.com/a/55102639]"""
    def __init__(self, ax, labels, active=0, activecolor='blue', size=49,
                 orientation="vertical", **kwargs):
        """
        Add radio buttons to an `~.axes.Axes`.
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The axes to add the buttons to.
        labels : list of str
            The button labels.
        active : int
            The index of the initially selected button.
        activecolor : color
            The color of the selected button.
        size : float
            Size of the radio buttons
        orientation : str
            The orientation of the buttons: 'vertical' (default), or 'horizontal'.
        Further parameters are passed on to `Legend`.
        """
        AxesWidget.__init__(self, ax)
        self.activecolor = activecolor
        axcolor = ax.get_facecolor()
        self.value_selected = None

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)

        circles = []
        for i, label in enumerate(labels):
            if i == active:
                self.value_selected = label
                facecolor = activecolor
            else:
                facecolor = axcolor
            p = ax.scatter([],[], s=size, marker="o", edgecolor='black',
                           facecolor=facecolor)
            circles.append(p)
        if orientation == "horizontal":
            kwargs.update(ncol=len(labels), mode="expand")
        kwargs.setdefault("frameon", False)    
        self.box = ax.legend(circles, labels, loc="center", **kwargs)
        self.labels = self.box.texts
        self.circles = self.box.legendHandles
        for c in self.circles:
            c.set_picker(5)
        self.cnt = 0
        self.observers = {}

        self.connect_event('pick_event', self._clicked)


    def _clicked(self, event):
        if (self.ignore(event) or event.mouseevent.button != 1 or
            event.mouseevent.inaxes != self.ax):
            return
        if event.artist in self.circles:
            self.set_active(self.circles.index(event.artist))



def opticalSpectra(Im,op_fit_maps,par,save=False,outliers=False):
    """Select ROIs and calculate average optical properties on each one.

@Inputs
    - Im: (cropped) Reflectance map to selet ROIs
    - op_fit_maps: (binned) map of optical properties, given by fitOps
    - par: dictionary containing processng parameters
    - save: if True, it will save the plot
    - outliers: if True, mask outlier values (detected using median and Median
                Absolute Deviation)
@Return
    - op_fit_maps: if outlier is True, it will be a MaskedArray, otherwise
                   return the original data
    - opt_ave: array containing average of optical properties in the ROIs
    - opt_std: array containing standard deviation of optical properties
               in the ROIs
    - radio1: need a reference to the widget in order to be interactive
@Instructions:
    - select multiple ROIs on the reflectance map (confirm each one with ENTER)
    - end selection with ESCAPE
    - [interactive]: use the radio buttons to show different wavelengths colormap    
"""
    ## Put callbacks here
    def call_mua(label):
#        label_dict = {'B':0,'GB':1,'G':2,'GR':3,'R':4}
#        data_a = op_fit_maps[:,:,label_dict[label],0]
#        data_s = op_fit_maps[:,:,label_dict[label],1]
#        im1.set_data(data_a)
#        im1.set_clim(vmin=0,vmax=np.nanmax(op_fit_maps[:,:,label_dict[label],0]))
#        cbar1 = colourbar(im1)
#
#        im2.set_data(data_s)
#        im2.set_clim(vmin=0,vmax=np.nanmax(op_fit_maps[:,:,label_dict[label],1]))
#        cbar2 = colourbar(im2)
#        plt.draw()
#        plt.tight_layout()
        ## new callback
        data_a = op_fit_maps[:,:,int(label),0]
        data_s = op_fit_maps[:,:,int(label),1]
        im1.set_data(data_a)
        im1.set_clim(vmin=0,vmax=np.nanmax(op_fit_maps[:,:,int(label),0]))
        cbar1 = colourbar(im1)

        im2.set_data(data_s)
        im2.set_clim(vmin=0,vmax=np.nanmax(op_fit_maps[:,:,int(label),1]))
        cbar2 = colourbar(im2)
        plt.draw()
        plt.tight_layout()
       
    cv.namedWindow('Select ROIs',cv.WINDOW_NORMAL)
    cv.resizeWindow('Select ROIs',(Im.shape[1],Im.shape[0]))
    ROIs = cv.selectROIs('Select ROIs',Im,False,False) # press enter to save the ROI, esc to terminate
    cv.destroyAllWindows()
    
    # Convert to maskedArray for convenience
    op_fit_maps = mask.masked_array(op_fit_maps) # should already be set to nomask
    
    ## Remove outliers
    if outliers:
        ## New approach: create a masked array (to not alter original data)
        # First: calculate the thresholds using median and MAD:
        med = np.nanmedian(op_fit_maps,axis=(0,1))
        MAD = mad(op_fit_maps,axis=(0,1))
        # loop over wavelength and OP axis
        for wv in range(len(MAD)):
            for i in range(len(MAD[0])):
                # mask if the distance from the median value is higher than 10*MAD
                idx = np.where(abs(op_fit_maps[:,:,wv,i] - med[wv,i]) > 15*MAD[wv,i])
                for x in range(len(idx[0])):
                    op_fit_maps[idx[0],idx[1],wv,i] = mask.masked
    
    colours=['b','r',(0,1,0),'k',(0,1,1),(1,0,1),(0.9,0.9,0)] # colour scheme to draw lines
    if len(ROIs) > 7:
        colours = colours*2 # we assume no more than 14 ROIs
    
    opt_ave = np.zeros((len(ROIs),len(par['wv']),2),dtype=float)
    opt_std = np.zeros((len(ROIs),len(par['wv']),2),dtype=float)

    # Before calculating average, put the 

    for i in range(len(ROIs)): # calculate average and std inside ROIs
        opt_ave[i,:,:] = np.nanmean(crop(op_fit_maps,ROIs[i,:]//par['binsize']),axis=(0,1))
        opt_std[i,:,:] = np.nanstd(crop(op_fit_maps,ROIs[i,:]//par['binsize']),axis=(0,1))
    
#    # New: manually define axis using gridspec
#    fig = plt.figure(constrained_layout=False,figsize=(10,6))
#    spec = gridspec.GridSpec(ncols=3,nrows=2,width_ratios=[1,0.3,1],figure=fig)
#    
#    ax1 = fig.add_subplot(spec[0,0])
#    ax2 = fig.add_subplot(spec[0,1])
#    ax3 = fig.add_subplot(spec[0,2])
#    ax4 = fig.add_subplot(spec[1,0])
#    ax5 = fig.add_subplot(spec[1,1])
#    ax6 = fig.add_subplot(spec[1,2])
#    
#    ax = np.array([[ax1,ax2,ax3],[ax4,ax5,ax6]]) # this way the old scheme is kept
    
    ## New layout
    fig = plt.figure(constrained_layout=False,figsize=(10,7))
    spec = gridspec.GridSpec(ncols=2,nrows=3,height_ratios=[0.2,1,1],figure=fig)
    
    # convert coordinates
    ax1 = fig.add_subplot(spec[1,0])  # 0,0 -> 1,0
    ax2 = fig.add_subplot(spec[0,:])  # 0,1 -> 0,:
    ax3 = fig.add_subplot(spec[1,1])  # 0,2 -> 1,1
    ax4 = fig.add_subplot(spec[2,0])  # 1,0 -> 2,0
    ax5 = fig.add_subplot(spec[1,1])  # 1,1 -> xx
    ax6 = fig.add_subplot(spec[2,1])  # 1,2 -> 2,1
    ax = np.array([[ax1,ax2,ax3],[ax4,None,ax6]]) # this way the old scheme is kept
    
    
    vmax = np.nanmax(op_fit_maps[:,:,:,0])
    im1 = ax[0,0].imshow(op_fit_maps[:,:,0,0],cmap='magma',vmin=0,vmax=vmax)
    ax[0,0].set_title('Absorption coefficient ($\mu_A$)')
    ax[0,0].get_xaxis().set_visible(False)
    ax[0,0].get_yaxis().set_visible(False)
    cbar1 = colourbar(im1)
    
    radio1 = MyRadioButtons(ax[0,1], labels=[str(x) for x in range(op_fit_maps.shape[2])], 
                            orientation='horizontal', title='wavelength', size=60)
    radio1.on_clicked(call_mua)
    
    vmax = np.nanmax(op_fit_maps[:,:,:,1])
    im2 = ax[0,2].imshow(op_fit_maps[:,:,0,1],cmap='magma',vmin=0,vmax=vmax)
    ax[0,2].set_title('Scattering coefficient ($\mu_S´$)')
    ax[0,2].get_xaxis().set_visible(False)
    ax[0,2].get_yaxis().set_visible(False)
    
    current_cmap = cm.get_cmap('magma')
    current_cmap.set_bad(color='cyan') # masked values colour
    cbar2 = colourbar(im2)

    #ax[1,1].axis('off')
    
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
        
    plt.tight_layout()
    
    if save:
        plt.savefig('figure%d.png'%save, bbox_inches=None)
    
    return op_fit_maps,opt_ave,opt_std,radio1


if __name__ == '__main__':
#    asd = np.load('../processing/test_data.npz')
#    op_fit_maps = asd['op_fit_maps']
#    cal_R = asd['cal_R']
#    ROI = asd['ROI']
#    
#    par = {'binsize':4, 'wv':[458,520,536,556,626]}
    cropped = crop(cal_R[0][:,:,0,0],ROI)
    op_fit_maps,opt_ave,opt_std,radio = opticalSpectra(cropped,op_fit_maps[0],par,outliers=True)