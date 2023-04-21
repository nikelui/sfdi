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
import cv2 as cv
import numpy as np
import numpy.ma as ma
from scipy.io import loadmat
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.cm as cm
import addcopyfighandler

from sfdi.analysis.depthMC import depthMC
from sfdi.processing.crop import crop
from sfdi.processing.stackPlot import stackPlot
from sfdi.common.colourbar import colourbar

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

def get_mpl_colormap(cmap_name):
    """Get matplotlib colormat to use with opencv.
    https://stackoverflow.com/a/52501371"""
    cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]
    return color_range.reshape(256, 1, 3)


class dataDict(dict):
    """An helper class that expands the dict container"""
    def __init__(self,*arg,**kw):
       super(dataDict, self).__init__(*arg, **kw)
       parameters = kw.pop('parameters', {'wv':[458, 520, 536, 556, 626]})
       self.par = parameters
    
    def list_data(self):
        """List the name of all the datasets and the number of spatial frequencies ranges"""
        print('Datasets:')
        for key in [x for x in self.keys() if not x=='parameters']:
            print('- {} [{} fx]'.format(key, len(self[key])))
    
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
                    vmin = 0
                    if _i == 0:
                        vmax = 1
                    else:
                        vmax = 5
                    im = ax[_i, _j].imshow(op_map, cmap='magma', vmax=vmax, vmin=vmin)
                    colourbar(im)
                    ax[_i, _j].axis('off')
                cmap = cm.get_cmap('magma')
                cmap.set_bad(color='cyan')
                plt.tight_layout()
    
    def plot_op_sfds(self, key, **kwargs):
        """Plot optical properties of dataset <key> (SFDS)"""
        f_used = kwargs.pop('f', list(range(len(self[key]))))  # Default: use all frequencies
        f_used = [x for x in f_used if 0 <= x < len(self[key])]  # add additional check to index
        norm = kwargs.pop('norm', None)
        fit = kwargs.pop('fit', None)  # wether to plot the raw data or the fitted one
        fig, ax = plt.subplots(num=100, nrows=1, ncols=2, figsize=(10,4))
        for _f, fx in enumerate(self[key].keys()):
            if _f in f_used:
                fitted = fit_fun(self.par['wv_sfds'], self[key][fx]['sfds']['par'][0],
                                     self[key][fx]['sfds']['par'][1])
                ax[0].plot(self.par['wv_sfds'], self[key][fx]['sfds']['op_fit'][:,0],
                               linewidth=2, label=f'{fx}', color='C{}'.format(_f))
                if fit == 'single' and norm is None:
                    ax[1].plot(self.par['wv_sfds'], fitted, linewidth=2, label=f'{fx}',
                               color='C{}'.format(_f))
                elif fit == 'single' and norm is not None:
                    ax[1].plot(self.par['wv_sfds'], fitted / fitted[norm],
                               linewidth=2, label=f'{fx}', color='C{}'.format(_f))
                elif fit == 'double' and norm is None:
                    ax[1].plot(self.par['wv_sfds'], self[key][fx]['sfds']['op_fit'][:,1],
                               linewidth=2, label=f'{fx}', color='C{}'.format(_f))
                    ax[1].plot(self.par['wv_sfds'], fitted, linewidth=2,
                               color='k'.format(_f), linestyle=':')
                elif fit == 'double' and norm is not None:
                    ax[1].plot(self.par['wv_sfds'], self[key][fx]['sfds']['op_fit'][:,1],
                               linewidth=2, label=f'{fx}', color='C{}'.format(_f))
                    ax[1].plot(self.par['wv_sfds'], fitted / fitted[norm], linewidth=2,
                               color='k'.format(_f), linestyle=':')
                else:
                    ax[1].plot(self.par['wv_sfds'], self[key][fx]['sfds']['op_fit'][:,1],
                               linewidth=2, label=f'{fx}', color='C{}'.format(_f))
        plt.suptitle('{}'.format(key))
        ax[0].set_title(r"$\mu_a$")
        ax[0].set_xlabel('wv (nm)')
        ax[0].set_ylabel(r'mm$^{-1}$')
        ax[0].grid(True, linestyle=':')
        ax[0].set_xlim([450, 750])
        ax[1].set_title(r"$\mu'_s$")
        ax[1].set_xlabel('wv (nm)')
        ax[1].set_ylabel(r'mm$^{-1}$')
        ax[1].grid(True, linestyle=':')
        ax[1].set_xlim([450, 750])
        ax[1].legend(loc=0)
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
        for _i, _j in itertools.product(np.arange(len(self[key])), np.arange(column)):
            if _i in f_used:
                _k = f_used.index(_i)  # change variable for simplicity
                mus = self[key][f'f{_i}']['op_fit_maps'][:,:,_j,1]
                
                # import pdb; pdb.set_trace()  # DEBUG
                
                im = ax[_k, _j].imshow(mus, cmap='viridis', vmin=vmin, vmax=vmax)
                colourbar(im)
                ax[_k, _j].axes.xaxis.set_ticks([])
                ax[_k, _j].axes.yaxis.set_ticks([])
                if _j == 0:
                    ax[_k, _j].set_ylabel('{}'.format(f'f{_i}'))
        cmap = cm.get_cmap('magma')
        cmap.set_bad(color='cyan')
        plt.tight_layout()
    
    def plot_par(self, key, **kwargs):
        """Plot fitted A, B parameters of reduced scattering"""
        f_used = kwargs.pop('f', list(range(len(self[key]))))  # Default: use all frequencies
        f_used = [x for x in f_used if 0 <= x < len(self[key])]  # add additional check to index
        bmax = kwargs.pop('bmax', 2)
        amax = kwargs.pop('amax', 1e5)
        column = self[key]['f0']['par_map'].shape[-1]
        row = len(f_used)
        fig, ax = plt.subplots(num=300, nrows=row, ncols=column, figsize=(column*3, row*2))
        for _i in range(len(self[key])):
            first_column = True
            if _i in f_used:
                _k = f_used.index(_i)  # change variable for simplicity
                par = self[key][f'f{_i}']['par_map']
                im = ax[_k, 0].imshow(par[:,:,0], cmap='magma',
                                      norm=colors.LogNorm(vmin=1e-4, vmax=amax))
                colourbar(im)
                im = ax[_k, 1].imshow(par[:,:,1], cmap='viridis', vmin=0, vmax=bmax)
                colourbar(im)
                ax[_k, 0].axes.xaxis.set_ticks([])
                ax[_k, 0].axes.yaxis.set_ticks([])
                ax[_k, 1].axes.xaxis.set_ticks([])
                ax[_k, 1].axes.yaxis.set_ticks([])
                ax[_k, 0].set_ylabel('{}'.format(f'f{_i}'))
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
        I = kwargs.pop('I', 10)
        zoom = kwargs.pop('zoom', 3)  # defaults to 3
        norm = kwargs.pop('norm', None)
        fit = kwargs.pop('fit', None)  # wether to plot the raw data or the fitted one
        f_used = kwargs.pop('f', list(range(len(self[key]))))  # Default: use all frequencies
        f_used = [x for x in f_used if 0 <= x < len(self[key])]  # add additional check to index
        #TODO: actually implement this. Might need to convert parameters to np.array
        wv_used = kwargs.pop('wv', list(range(len(self.par['wv']))))
        im = self[key]['f0']['op_fit_maps'][:,:,-1,0]*I  # reference image
        im = cv.applyColorMap(im.astype('uint8'), get_mpl_colormap('magma'))  # apply matplotlib colormap
        z = np.arange(0, 4, 0.001)  # 1um resolution
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
        op_fit_double = np.zeros((len(self[key]), 100, 2), dtype=float)
        depths = np.zeros((len(self[key]), self[key]['f0']['op_fit_maps'].shape[2]),
                          dtype=float)  # depth based on diffusion approx.
        depths_std = np.zeros((len(self[key]), 2, self[key]['f0']['op_fit_maps'].shape[2]),
                          dtype=float)
        depth_phi = np.zeros((len(self[key]), self[key]['f0']['op_fit_maps'].shape[2]),
                            dtype=float)  # depth based on phi^2
        depth_MC = np.zeros((len(self[key]), self[key]['f0']['op_fit_maps'].shape[2]),
                            dtype=float)  # depth based on Monte Carlo
        fluence = np.zeros((len(self[key]), self[key]['f0']['op_fit_maps'].shape[2],
                            len(z)), dtype=float)
        par_ave = np.zeros((len(self[key]), 2), dtype=float)
        par_std = np.zeros((len(self[key]), 2), dtype=float)
        
        # fx = list(self[key].keys())  # list of fx ranges

        for _i in f_used:
            # mean and std of mua, mus over ROI
            op_ave[_i, :, :] = np.nanmean(crop(self[key][f'f{_i}']['op_fit_maps'], ROI), axis=(0,1))
            op_std[_i, :, :] = np.nanstd(crop(self[key][f'f{_i}']['op_fit_maps'], ROI), axis=(0,1))
            #  depths calculated with diffusion approximation
            depths[_i,:] = self.depth(op_ave[_i,:,0], op_ave[_i,:,1], np.mean(self.par[f'f{_i}']))
            depths_std[_i,0,:] = self.depth(op_ave[_i,:,0], op_ave[_i,:,1], self.par[f'f{_i}'][-1])
            depths_std[_i,1,:] = self.depth(op_ave[_i,:,0], op_ave[_i,:,1], self.par[f'f{_i}'][0])
            depths_std[_i,:,:] = np.absolute(depths_std[_i,:,:] - depths[_i,np.newaxis,:])  # relative depth
            # fluence, from diffusion approximation
            phi = self.phi(op_ave[_i,:,0], op_ave[_i,:,1], np.mean(self.par[f'f{_i}']), z)
            fluence[_i,:,:] = phi.T
            # calculate depth from fluence^2
            for _j, line in enumerate(phi.T):
                # import pdb; pdb.set_trace()
                idx = np.argwhere(line**2 <= np.max(line**2)/np.e)[0][0]
                depth_phi[_i, _j] = z[idx]  # where phi < (1/e * phi)
            # calculate depth based on Monte Carlo table
            temp = depthMC(op_ave[_i,:,0], op_ave[_i,:,1], np.mean(self.par[f'f{_i}']))
            depth_MC[_i, :] = temp[4,:,:]  # index '4'-> assumes 90% of photons

            if fit:
                try:
                    if fit == 'single':
                        (A, B), _ = curve_fit(fit_fun, self.par['wv'][:], op_ave[_i,:,1], p0=[100,1],
                                              method='trf', loss='soft_l1', max_nfev=2000)
                        par_ave[_i,:] = (A, B)
                        
                        #### New approach?
                        # (A, B) = np.nanmean(crop(self[key][f'f{_i}']['par_map'], ROI), axis=(0,1))
                        # par_ave[_i,:] = np.nanmean(crop(self[key][f'f{_i}']['par_map'], ROI), axis=(0,1))
                        # par_std[_i,:] = np.nanstd(crop(self[key][f'f{_i}']['par_map'], ROI), axis=(0,1))
                        ####
                        
                        op_fit[_i,:] = fit_fun(np.linspace(self.par['wv'][0], self.par['wv'][-1], 100), A, B)
                        # import pdb; pdb.set_trace()  # DEBUG start
                        print('{}\nA: {:.2f}, B: {:.4f}\nd: {:.4f}, df: {:.3f}'.format(
                                    f'f{_i}', A, B, np.mean(depths[_i,:]), np.mean(depth_phi[_i,:])) +
                                    ', dmc: {:.4f}'.format(np.mean(depth_MC[_i,:])))  # DEBUG
                    elif fit == 'double':
                        (A1, B1), _ = curve_fit(fit_fun, self.par['wv'][:3], op_ave[_i,:3,1], p0=[100,1],
                                              method='trf', loss='soft_l1', max_nfev=2000)
                        (A2, B2), _ = curve_fit(fit_fun, self.par['wv'][3:], op_ave[_i,3:,1], p0=[100,1],
                                              method='trf', loss='soft_l1', max_nfev=2000)
                        print('{}\nA1: {:.2f}, A2: {:.2f}\nB1: {:.4f}, B2: {:.4f}'.format(
                            f'f{_i}', A1, A2, B1, B2))  # DEBUG
                        op_fit_double[_i,:,0] = fit_fun(np.linspace(self.par['wv'][0], self.par['wv'][-1], 100), A1, B1)
                        op_fit_double[_i,:,1] = fit_fun(np.linspace(self.par['wv'][0], self.par['wv'][-1], 100), A2, B2)
                except RuntimeError:
                    continue
            if norm is not None:  # Normalize to reference wv
                op_ave[_i, :, 1] /= op_ave[_i, norm, 1]
                op_std[_i, :, 1] /= op_ave[_i, norm, 1]
                if fit == 'single':
                    op_fit[_i,:] /= op_fit[_i, norm]
                elif fit =='double':
                    op_fit_double[_i,:,0] /= op_fit_double[_i,norm,0]
                    op_fit_double[_i,:,1] /= op_fit_double[_i,norm,1]
        # Here plot the data points
        fig, ax = plt.subplots(num=500, nrows=1, ncols=3, figsize=(15, 4))
        if fit:
            if fit == 'single':
                for _j in range(len(f_used)):
                    _i = f_used[_j]  # variable change, just for simplicity
                    ax[0].errorbar(self.par['wv'], op_ave[_i,:,0],fmt='o', yerr=op_std[_i,:,0], linestyle='-',
                                   linewidth=2, capsize=5, label=f'f{_i}', color='C{}'.format(_j))
                    ax[1].errorbar(self.par['wv'], op_ave[_i,:,1],fmt='o', yerr=op_std[_i,:,1], linestyle='none',
                                   capsize=5, label=f'f{_i}', color='C{}'.format(_j))
                    ax[1].plot(np.linspace(self.par['wv'][0], self.par['wv'][-1], 100), op_fit[_i,:], linestyle='-',
                               linewidth=2, color='C{}'.format(_j))
                    ax[2].errorbar(self.par['wv'], depths[_i], fmt='o', yerr=depths_std[_i,:,:], linestyle='-',
                                   linewidth=2, capsize=5, label=f'f{_i}', color='C{}'.format(_j)) 
            elif fit == 'double':
                for _j in range(len(f_used)):
                    _i = f_used[_j]  # variable change, just for simplicity
                    ax[0].errorbar(self.par['wv'], op_ave[_i,:,0],fmt='o', yerr=op_std[_i,:,0], linestyle='-',
                                   linewidth=2, capsize=5, label=f'f{_i}', color='C{}'.format(_j))
                    ax[1].errorbar(self.par['wv'], op_ave[_i,:,1],fmt='o', yerr=op_std[_i,:,1], linestyle='none',
                                   capsize=5, label=f'f{_i}', color='C{}'.format(_j))
                    ax[1].plot(np.linspace(self.par['wv'][0], self.par['wv'][-1], 100), op_fit_double[_i,:,0], linestyle='--',
                               linewidth=2, color='C{}'.format(_j))
                    ax[1].plot(np.linspace(self.par['wv'][0], self.par['wv'][-1], 100), op_fit_double[_i,:,1], linestyle=':',
                               linewidth=2, color='C{}'.format(_j))
                    
                    ax[2].errorbar(self.par['wv'], depths[_i], fmt='o', yerr=depths_std[_i,:,:], linestyle='-',
                                   linewidth=2, capsize=5, label=f'f{_i}', color='C{}'.format(_j)) 
        else:
            for _j in range(len(f_used)):
                _i = f_used[_j]
                ax[0].errorbar(self.par['wv'], op_ave[_i,:,0],fmt='o', yerr=op_std[_i,:,0], linestyle='-',
                               linewidth=2, capsize=5, label=f'f{_i}', color='C{}'.format(_j))
                ax[1].errorbar(self.par['wv'], op_ave[_i,:,1],fmt='o', yerr=op_std[_i,:,1], linestyle='-',
                               linewidth=2, capsize=5, label=f'f{_i}', color='C{}'.format(_j))
                ax[2].errorbar(self.par['wv'], depths[_i], fmt='o', yerr=depths_std[_i,:,:], linestyle='-',
                               linewidth=2, capsize=5, label=f'f{_i}', color='C{}'.format(_j))     
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
        ax[2].set_ylim([0, None])
        cmap = cm.get_cmap('magma')
        cmap.set_bad(color='cyan')
        plt.tight_layout()
        
        ret_value = {'ROI':ROI, 'op_ave': op_ave, 'op_std': op_std, 'depths': depths, 'depths_std': depths_std,
                     'depth_phi': depth_phi, 'par_ave': par_ave, 'par_std': par_std, 'fluence':fluence,
                     'depth_MC':depth_MC}
        return ret_value
    
    def multiROI(self, key, **kwargs):
        I = kwargs.pop('I', 1)
        zoom = kwargs.pop('zoom', 3)  # defaults to 3
        what = kwargs.pop('what', 'mus')
        wv = kwargs.pop('wv', 2)
        vmax = kwargs.pop('vmax', 3)
        vmin = kwargs.pop('vmin', 1)
        f_used = kwargs.pop('f', list(range(len(self[key]))))  # Default: use all frequencies
        f_used = [x for x in f_used if 0 <= x < len(self[key])]  # add additional check to index
        wv_used = kwargs.pop('wv', list(range(len(self.par['wv']))))
        im = self[key]['f0']['op_fit_maps'][:,:,0,0]*I  # reference image
        im = cv.applyColorMap(im.astype('uint8'), get_mpl_colormap('magma'))  # apply matplotlib colormap
        cv.namedWindow('select ROI', cv.WINDOW_NORMAL)
        cv.resizeWindow('select ROI', im.shape[1]*zoom, im.shape[0]*zoom)
        ROI = cv.selectROIs('select ROI', im)
        cv.destroyAllWindows()
        
        # fx = list(self[key].keys())  # list of fx ranges
        fig, ax = plt.subplots(num=600, nrows=2, ncols=len(f_used), figsize=(15, 6))
        rect_colors = ['cyan', 'lime', 'blue', 'magenta']
        for _i,_j in enumerate(f_used):
            if what == 'mus':
                #TODO: implement fit to power law
                im = ax[0,_i].imshow(self[key][f'f{_j}']['op_fit_maps'][:,:,wv,1],
                                     vmin=vmin, vmax=vmax, cmap='magma')
                cb = colourbar(im)
                for _r, roi in enumerate(ROI):
                    rect = patches.Rectangle((roi[0], roi[1]), roi[2], roi[3], linewidth=1,
                                             facecolor='none', edgecolor=rect_colors[_r])
                    ax[0,_i].add_patch(rect)
                    ax[1,_i].plot(self.par['wv'], np.mean(crop(self[key][f'f{_j}']['op_fit_maps'][:,:,:,1], roi), axis=(0,1)),
                                  '-d', linewidth=1.5, color=rect_colors[_r])
                ax[0,_i].axis('off')
                ax[1,_i].grid(True, linestyle=':')
                ax[1,_i].set_ylim([vmin, vmax])
        plt.tight_layout()

            #TODO: implement this for a,b params
            # elif what == 'par':
            #     im = ax[0,_i].imshow(self[key][fx[_j]]['par_map'][:,:,1], vmax=2)
            #     cb = colourbar(im)
            #     for _r, roi in enumerate(ROI):
            #         rect = patches.Rectangle((roi[0], roi[1]), roi[2], roi[3], linewidth=1.5,
            #                                  facecolor='none', edgecolor=rect_colors[_r])
            #         ax[0,_i].add_patch(rect)
            #         ax[1,_i].plot(self.wv, np.mean(crop(self[key][fx[_j]]['par_map'][:,:,1], roi), axis=(0,1)),
            #                       'd', linewidth=1.5, color=rect_colors[_r])
            #     ax[0,_i].grid(True, linestyle=':')
            #     ax[1,_i].grid(True, linestyle=':')
            
    
    def mask_on(self):
        for dataset in [x for x in self if not x == 'parameters']:
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
        for dataset in [x for x in self if not x == 'parameters']:
            for fx in self[dataset]:
                self[dataset][fx]['op_fit_maps'] = self[dataset][fx]['op_fit_maps'].data
    
    def depth(self, mua, mus, fx):
        """Function to calculate effective penetration depth based on diffusion approximation
    - mua, mus: vectors (1 x wv)
    - fx: average fx in range"""
        mut = mua + mus
        mueff = np.sqrt(np.abs(3 * mua * mut))
        mueff1 = np.sqrt(mueff**2 + (2*np.pi*fx)**2)
        d = 1/mueff1
        return d
    
    def phi(self, mua, mus, fx, z):
        """Function to calculate effective penetration depth from the fluence formula as 1/e
    - mua, mus: vectors (1 x wv)
    - fx: average fx in range
    - z: depth (array 1 x N)"""
        mut = mua + mus
        mueff = np.sqrt(np.abs(3 * mua * mut))
        mueff1 = np.sqrt(mueff**2 + (2*np.pi*fx)**2)
        a1 = mus / mut  # albedo
        # effective reflection coefficient. Assume n = 1.4
        Reff = 0.0636*1.4 + 0.668 + 0.71/1.4 - 1.44/1.4**2
        A = (1 - Reff)/(2*(1 + Reff))  # coefficient
        C = (-3*a1*(1 + 3*A))/((mueff1**2/mut**2 - 1) * (mueff1/mut + 3*A))
        phi = 3*a1 / (mueff1**2 / mut**2 - 1) * np.exp(-mut[np.newaxis,:] * z[:,np.newaxis]) +\
              C * np.exp(-mueff1[np.newaxis,:] * z[:,np.newaxis])    
        return phi
        