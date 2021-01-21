# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:15:00 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Custom dictionary class (inheriting from python dict) to store and analyze
the processed data.

class methods:
    - plot_op(key):
        Plots (mua, mus') of dataset <key> at all wavelengths. Each fx is a separate figure
    - plot_mus(key, [vmin, vmax]):
        Plots only mus' of dataset <key> at all fx in a single figure.
    - plot_par(key):
        Plots (A,B) fitted parameters of dataset <key> at all fx in a single figure
    - singleROI(key):
        Select a ROI on dataset <key> and plots mean and std (mua, mus') in the ROI
    - mask_on():
        Masks outlier values in (mua, mus')
    - mask_off():
        Delete mask on (mua, mus') if exists

data structure:
    data
    |-- dataset1
    |   |-- f0
    |   |   |-- Matlab header stuff (not relevant)
    |   |   |-- op_fit_maps (mua, mus')
    |   |   |-- par_map (A, B)
    |   |
    |   |-- f1
    |   |-- f2
    |
    |-- dataset2
    |   |-- f0
    |   |-- f1
    |   |-- f2
    etc...
"""
import os
import itertools
import numpy as np
import numpy.ma as ma
from scipy.io import loadmat
from scipy.optimize import curve_fit
import cv2 as cv
from sfdi.common.sfdi.crop import crop
from sfdi.common.stackPlot import stackPlot
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def colourbar(mappable, **kwargs):
    """Improved colorbar function. Fits well to the axis dimension."""
    if (mappable.colorbar is not None):
        mappable.colorbar.remove()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, **kwargs)

def mad(x,scale=1.4826,axis=None):
    """Median assoulte difference (since Scipy does not implement it anymore).
    This is preferred as a measure of dispersion, since it is more robust to
    outliers. The scale factor is to make it comparable with standard deviation
    (for normal distribution)"""
    med = np.nanmedian(x,axis=axis)
    return np.nanmedian(np.abs(x-med),axis=axis)*scale

def fit_fun(lamb, a, b):
    """Exponential function to fit data to"""
    return a * np.power(lamb, -b)


class dataDict(dict):
    """An helper class that expands the dict container"""
    def __init__(self,*arg,**kw):
       super(dataDict, self).__init__(*arg, **kw)
       parameters = kw.pop('parameters', {'wv':[458, 520, 536, 556, 626]})
       self.par = parameters
    
    def plot_op(self, key, **kwargs):
        """Plot optical properties of dataset <key>"""
        #TODO: improve docstring
        f_used = kwargs.pop('f', list(range(len(self[key]))))  # Default: use all frequencies
        f_used = [x for x in f_used if 0 <= x < len(self[key])]  # add additional check to index
        for _f, fx in enumerate(self[key].keys()):
            if _f in f_used:
                column, row = self[key][fx]['op_fit_maps'].shape[2:]
                fig, ax = plt.subplots(num=100+_f, nrows=row, ncols=column, figsize=(column*3, row*2))
                plt.suptitle('{}'.format(fx))
                for _i, _j in itertools.product(np.arange(row), np.arange(column)):
                    op_map = self[key][fx]['op_fit_maps'][:,:,_j,_i]
                    if _i == 0:
                        vmax = 0.5
                    else:
                        vmax = 5
                    im = ax[_i, _j].imshow(op_map, cmap='magma', vmax=vmax)
                    colourbar(im)
                    ax[_i, _j].axis('off')
                cmap = cm.get_cmap('magma')
                cmap.set_bad(color='cyan')
                plt.tight_layout()
            
    def plot_mus(self, key, vmin=0, vmax=4, **kwargs):
        """Plot optical properties of dataset <key>"""
        #TODO: improve docstring
        f_used = kwargs.pop('f', list(range(len(self[key]))))  # Default: use all frequencies
        f_used = [x for x in f_used if 0 <= x < len(self[key])]  # add additional check to index
        column = self[key]['f0']['op_fit_maps'].shape[2]
        row = len(f_used)
        fig, ax = plt.subplots(num=200, nrows=row, ncols=column, figsize=(column*3, row*2))
        ax = ax.reshape((row, column))  # just in case it is 1D
        # plt.suptitle(r"$\mu'_s'$")
        fx = list(self[key].keys())
        for _i, _j in itertools.product(np.arange(len(self[key])), np.arange(column)):
            if _i in f_used:
                _k = f_used.index(_i)  # change variable for simplicity
                mus = self[key][fx[_i]]['op_fit_maps'][:,:,_j,1]
                
                # import pdb; pdb.set_trace()
                
                im = ax[_k, _j].imshow(mus, cmap='magma', vmin=vmin, vmax=vmax)
                colourbar(im)
                ax[_k, _j].axes.xaxis.set_ticks([])
                ax[_k, _j].axes.yaxis.set_ticks([])
                if _j == 0:
                    ax[_k, _j].set_ylabel('{}'.format(fx[_i]))
        cmap = cm.get_cmap('magma')
        cmap.set_bad(color='cyan')
        plt.tight_layout()
    
    def plot_par(self, key, **kwargs):
        """Plot fitted A, B parameters of reduced scattering"""
        f_used = kwargs.pop('f', list(range(len(self[key]))))  # Default: use all frequencies
        f_used = [x for x in f_used if 0 <= x < len(self[key])]  # add additional check to index
        column = self[key]['f0']['par_map'].shape[-1]
        row = len(f_used)
        fig, ax = plt.subplots(num=200, nrows=row, ncols=column, figsize=(column*3, row*2))
        fx = list(self[key].keys())
        for _i in range(len(self[key])):
            first_column = True
            if _i in f_used:
                _k = f_used.index(_i)  # change variable for simplicity
                par = self[key][fx[_i]]['par_map']
                im = ax[_k, 0].imshow(par[:,:,0], cmap='seismic',
                                      norm=colors.LogNorm(vmin=1e-5, vmax=1e4))
                colourbar(im)
                im = ax[_k, 1].imshow(par[:,:,1], cmap='seismic', vmin=-5, vmax=5)
                colourbar(im)
                ax[_k, 0].axes.xaxis.set_ticks([])
                ax[_k, 0].axes.yaxis.set_ticks([])
                ax[_k, 1].axes.xaxis.set_ticks([])
                ax[_k, 1].axes.yaxis.set_ticks([])
                ax[_k, 0].set_ylabel('{}'.format(fx[_i]))
                if first_column == 0:
                    ax[_k, 0].set_title('A')
                    ax[_k, 1].set_title('B')
                    first_column = False
        cmap = cm.get_cmap('seismic')
        cmap.set_bad(color='limegreen')
        plt.tight_layout()
    
    def plot_cal(self, key, path=None):
        fpath = [x for x in os.listdir(path) if 'calR' in x and key in x]
        calR = loadmat('{}/{}'.format(path, fpath[0]))
        calR = calR['cal_R']
        stackPlot(calR, cmap='magma', num=400)
        del calR  # to save some memory
    
    def singleROI(self, key, **kwargs):
        """optional arguments:
 - norm: the index to normalize the mus plot to (default is None). Usually use 0 or -1
 - fit: plot the fitted mus (default is False)
        """
        zoom = kwargs.pop('zoom', 3)  # defaults to 3
        norm = kwargs.pop('norm', None)
        fit = kwargs.pop('fit', False)  # wether to plot the raw data or the fitted one
        f_used = kwargs.pop('f', list(range(len(self[key]))))  # Default: use all frequencies
        f_used = [x for x in f_used if 0 <= x < len(self[key])]  # add additional check to index
        im = self[key]['f0']['op_fit_maps'][:,:,0,0]  # reference image
        cv.namedWindow('select ROI', cv.WINDOW_NORMAL)
        cv.resizeWindow('select ROI', im.shape[1]*zoom, im.shape[0]*zoom)
        ROI = cv.selectROI('select ROI', im)
        cv.destroyAllWindows()
        # calculate average inside ROI
        op_ave = np.zeros((len(self[key]), self[key]['f0']['op_fit_maps'].shape[2],
                           self[key]['f0']['op_fit_maps'].shape[-1]),  dtype=float)
        op_std = np.zeros((len(self[key]), self[key]['f0']['op_fit_maps'].shape[2],
                           self[key]['f0']['op_fit_maps'].shape[-1]),  dtype=float)
        op_fit = np.zeros((len(self[key]), 100), dtype=float)
        depths = np.zeros((len(self[key]), self[key]['f0']['op_fit_maps'].shape[2]),
                          dtype=float)
        fx = list(self[key].keys())  # list of fx ranges
        for _i in range(len(self[key])):
            op_ave[_i, :, :] = np.nanmean(crop(self[key][fx[_i]]['op_fit_maps'], ROI), axis=(0,1))
            op_std[_i, :, :] = np.nanstd(crop(self[key][fx[_i]]['op_fit_maps'], ROI), axis=(0,1))
            depths[_i,:] = self.depth(op_ave[_i,:,0], op_ave[_i,:,1], np.mean(self.par[fx[_i]]))
            if fit:
                try:
                    (A, B), _ = curve_fit(fit_fun, self.par['wv'], op_ave[_i,:,1], p0=[100,1],
                                          method='trf', loss='soft_l1', max_nfev=2000)
                    op_fit[_i,:] = fit_fun(np.linspace(self.par['wv'][0], self.par['wv'][-1], 100), A, B)
                except RuntimeError:
                    continue
            if norm is not None:  # Normalize to reference wv
                op_ave[_i, :, 1] /= op_ave[_i, norm, 1]
                op_std[_i, :, 1] /= op_ave[_i, norm, 1]
                if fit:
                    op_fit[_i,:] /= op_fit[_i, norm]
        # Here plot the data points
        fig, ax = plt.subplots(num=300, nrows=1, ncols=3, figsize=(12, 4))
        if fit:
            for _j in range(len(f_used)):
                _i = f_used[_j]  # variable change, just for simplicity
                ax[0].errorbar(self.par['wv'], op_ave[_i,:,0],fmt='o', yerr=op_std[_i,:,0], linestyle='-',
                               linewidth=2, capsize=5, label=fx[_i], color='C{}'.format(_j))
                ax[1].errorbar(self.par['wv'], op_ave[_i,:,1],fmt='o', yerr=op_std[_i,:,1], linestyle='none',
                               capsize=5, label=fx[_i], color='C{}'.format(_j))
                ax[1].plot(np.linspace(self.par['wv'][0], self.par['wv'][-1], 100), op_fit[_i,:], linestyle='-',
                           linewidth=2, color='C{}'.format(_j))
                ax[2].plot(self.par['wv'], depths[_i], 'o', linestyle='-', color='C{}'.format(_j),
                           label=fx[_i])
        else:
            for _j in range(len(f_used)):
                _i = f_used[_j]
                ax[0].errorbar(self.par['wv'], op_ave[_i,:,0],fmt='o', yerr=op_std[_i,:,0], linestyle='-',
                               linewidth=2, capsize=5, label=fx[_i], color='C{}'.format(_j))
                ax[1].errorbar(self.par['wv'], op_ave[_i,:,1],fmt='o', yerr=op_std[_i,:,1], linestyle='-',
                               linewidth=2, capsize=5, label=fx[_i], color='C{}'.format(_j))
                ax[2].plot(self.par['wv'], depths[_i], 'o', linestyle='-', color='C{}'.format(_j),
                           label=fx[_i])
        ax[0].grid(True, linestyle=':')
        ax[0].set_xlabel('nm')
        ax[0].set_ylabel(r'mm$^{-1}$')
        ax[0].set_title(r'$\mu_a$')
        ax[0].legend(loc=0)
        ax[1].grid(True, linestyle=':')
        ax[1].set_xlabel('nm')
        ax[1].set_ylabel(r'mm$^{-1}$')
        ax[1].set_title(r"$\mu'_s$")
        ax[1].legend(loc=0)
        ax[2].grid(True, linestyle=':')
        ax[2].set_xlabel('nm')
        ax[2].set_ylabel('mm')
        ax[2].set_title("penetration depth")
        ax[2].legend(loc=0)
        cmap = cm.get_cmap('magma')
        cmap.set_bad(color='cyan')
        plt.tight_layout()
        return op_ave, op_std
        
    def mask_on(self):
        for dataset in self:
            for fx in self[dataset]:
                op_map = self[dataset][fx]['op_fit_maps']
                mask = np.zeros(op_map.shape, dtype=bool)  # initialize
                for _i, _j in itertools.product(range(op_map.shape[-2]), range(op_map.shape[-1])):
                    # set mask to True if pixel values are outliers
                    mask[:,:,_i,_j] = np.logical_or(op_map[:,:,_i,_j] >= 15*mad(op_map[:,:,_i,_j]),
                                                    op_map[:,:,_i,_j] >= 20)  # hard limit
#                    print('{}, {}: {}'.format(_i, _j, mad(op_map[:,:,_i,_j])))  # DEBUG
                self[dataset][fx]['op_fit_maps'] = ma.masked_array(
                          data=op_map, mask=mask, fill_value=np.nan)
    
    def mask_off(self):
        for dataset in self:
            for fx in self[dataset]:
                self[dataset][fx]['op_fit_maps'] = self[dataset][fx]['op_fit_maps'].data
    
    def depth(self, mua, mus, fx):
        """Function to calculate effective penetration depth based on diffusion approximation
    - mua, mus: vectors (1 x wv)
    - fx: average fx in range"""
        mut = mua + mus
        mueff = np.sqrt(np.abs(3 * mua * mut))
        mueff1 = np.sqrt(mueff**2 + fx**2)
        d = 1/mueff1
        return d