# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 10:49:15 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Script to fit mus' to a power law of the kind A * lambda^(-b), select a ROI and
compare the variation at different fx
"""
import os, re
import datetime
import pickle
import itertools
import numpy as np
import cv2 as cv
from scipy.io import loadmat  # new standard: work with Matlab files for compatibility
from scipy.optimize import curve_fit
from sfdi.common.sfdi.getPath import getPath
from sfdi.common.sfdi.readParams3 import readParams
from sfdi.common.sfdi.crop import crop
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


class dataDict(dict):
    """An helper class that expands the dict container"""
    def __init__(self,*arg,**kw):
       super(dataDict, self).__init__(*arg, **kw)
    
    def plot_op(self, key, **kwargs):
        """Plot optical properties of dataset <key>"""
        #TODO: improve docstring
        for _f, fx in enumerate(self[key].keys()):
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
            plt.tight_layout()
            
    def plot_mus(self, key, vmin=0, vmax=4, **kwargs):
        """Plot optical properties of dataset <key>"""
        #TODO: improve docstring
        column = self[key]['f0']['op_fit_maps'].shape[2]
        row = len(self[key])
        fig, ax = plt.subplots(num=200, nrows=row, ncols=column, figsize=(column*3, row*2))
        plt.suptitle(r"$\mu'_s'$")
        fx = list(self[key].keys())
        for _i, _j in itertools.product(np.arange(row), np.arange(column)):
            mus = self[key][fx[_i]]['op_fit_maps'][:,:,_j,1]
            im = ax[_i, _j].imshow(mus, cmap='magma', vmin=vmin, vmax=vmax)
            colourbar(im)
            ax[_i, _j].axes.xaxis.set_ticks([])
            ax[_i, _j].axes.yaxis.set_ticks([])
            if _j == 0:
                ax[_i, _j].set_ylabel('{}'.format(fx[_i]))
        plt.tight_layout()
    
    def plot_par(self, key, **kwargs):
        """Plot fitted A, B parameters of reduced scattering"""
        column = self[key]['f0']['par_map'].shape[-1]
        row = len(self[key])
        fig, ax = plt.subplots(num=200, nrows=row, ncols=column, figsize=(column*3, row*2))
        fx = list(self[key].keys())
        for _i in range(row):
            par = self[key][fx[_i]]['par_map']
            im = ax[_i, 0].imshow(par[:,:,0], cmap='seismic',
                                  norm=colors.LogNorm(vmin=1e-5, vmax=1e4))
            colourbar(im)
            im = ax[_i, 1].imshow(par[:,:,1], cmap='seismic', vmin=-5, vmax=5)
            colourbar(im)
            ax[_i, 0].axes.xaxis.set_ticks([])
            ax[_i, 0].axes.yaxis.set_ticks([])
            ax[_i, 1].axes.xaxis.set_ticks([])
            ax[_i, 1].axes.yaxis.set_ticks([])
            ax[_i, 0].set_ylabel('{}'.format(fx[_i]))
            if _i == 0:
                ax[_i, 0].set_title('A')
                ax[_i, 1].set_title('B')
            
        plt.tight_layout()
    
    def singleROI(self, key, **kwargs):
        zoom = kwargs.pop('zoom', 3)  # defaults to 3
        im = self[key]['f0']['op_fit_maps'][:,:,0,0]  # reference image
        cv.namedWindow('select ROI', cv.WINDOW_NORMAL)
        cv.resizeWindow('select ROI', im.shape[1]*zoom, im.shape[0]*zoom)
        ROI = cv.selectROI('select ROI', im)
        cv.destroyAllWindows()
        # calculate average inside ROI
        op_ave = np.zeros((len(self[key]), self[key]['f0'].shape[2],
                           self[key]['f0'].shape[-1]),  dtype=float)
        fx = list(self[key].keys())  # list of fx ranges
        for _i in range(len(self[key])):
            op_ave[_i, :, :] = np.mean(crop(self[key][fx[_i]], ROI), axis=(0,1))

def save_obj(obj, name, path):
    """Utility function to save python objects using pickle module"""
    with open(path + '/obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, path):
    """Utility function to load python objects using pickle module"""
    with open(path + '/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    

def fit_fun(lamb, a, b):
    """Exponential function to fit data to"""
    return a * np.power(lamb, -b)


def mad(x,scale=1.4826,axis=None):
    """Median assoulte difference (since Scipy does not implement it anymore).
    This is preferred as a measure of dispersion, since it is more robust to
    outliers. The scale factor is to make it comparable with standard deviation
    (for normal distribution)"""
    med = np.nanmedian(x,axis=axis)
    return np.nanmedian(np.abs(x-med),axis=axis)*scale


def colourbar(mappable, **kwargs):
    """Improved colorbar function. Fits well to the axis dimension."""
    if (mappable.colorbar is not None):
        mappable.colorbar.remove()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, **kwargs)

#%%

wv = np.array([458, 520, 536, 556, 626])  # wavelengts (nm). Import from params?

regex = re.compile('.*f\d\.mat')  # regular expression
data_path = getPath('Select data path')
files = [x for x in os.listdir(data_path) if re.match(regex, x)]

datasets = set(x.split('_')[1] for x in files)  # sets have unique values
data = dataDict()
# load the data into a dictionary
# data structure:
#    data (dict)
#    |
#    |-- dataset1
#    |   |-- f0
#    |   |   |-- op_fit_maps (mua, mus')
#    |   |   |-- par_map (A, B)
#    |   |
#    |   |-- f1
#    |   |-- f2
#    |
#    |-- dataset2
#    |   |-- f0
#    |   |-- f1
#    |   |-- f2
#    ...
start = datetime.datetime.now()
for _d, dataset in enumerate(datasets, start=1):
    data[dataset] = {}  # need to initialize it
    temp = [x for x in files if dataset in x]   # get filenames
    freqs = [x.split('_')[-1][:-4] for x in temp]  # get frequency range
    for file,fx in zip(temp, freqs):
        data[dataset][fx] = loadmat('{}/{}'.format(data_path, file))
        # here fit the data
        print('Fitting dataset {}_{}...[{} of {}]'.format(dataset, fx, _d, len(datasets)))
        op_map = data[dataset][fx]['op_fit_maps']  # for convenience
        p_map = np.zeros((op_map.shape[0], op_map.shape[1], 2), dtype=float)  #initialize
        for _i in range(op_map.shape[0]):
            for _j in range(op_map.shape[1]):
                try:
                    temp, _ = curve_fit(fit_fun, wv, op_map[_i,_j,:,1], p0=[10, 1],
                                        method='trf', loss='soft_l1', max_nfev=2000)
                except RuntimeError:
                    continue
                p_map[_i, _j, :] = temp
        data[dataset][fx]['par_map'] = p_map
end = datetime.datetime.now()
print('Elapsed time: {}'.format(str(end-start).split('.')[0]))


# test to save data
if not os.path.isdir('{}/obj'.format(data_path)):
    os.makedirs('{}/obj'.format(data_path))
save_obj(data, 'dictionary', data_path)

bb = load_obj('dictionary', data_path)
