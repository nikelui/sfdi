# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:12:13 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import os, sys
import time
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.optimize import lsq_linear
# from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches

from sfdi.processing.chromPlot import chromPlot
from sfdi.processing.crop import crop
from sfdi.common.getPath import getPath
from sfdi.common.colourbar import colourbar

def target_fun(x, chrom, mua):
    """Function to minimize for non-linear fitting
    - x [Nx1]: unknown chromophore concentrations
    - chrom [WVxN]: chromophore spectrum
    - mua[WVx1]: absorption spectrum
"""
    return (chrom @ x) - mua

# Decorate chromPlot function to add RGB color image
def decorate(fun):
    def wrapper(*args, **kwargs):
        N = kwargs['dataset']
        # print('title = {}'.format(*args[1]))
        # print('kwargs = {}, '.format(kwargs.items()))
        _ = fun(args[0][N], args[1][N], args[2], kwargs['outliers'])
        # Set vmin / vmax
        axes = plt.gcf().get_axes()
        for _i in range(len(axes)//2):
            axes[_i].get_images()[0].set_clim(0, 0.02)
            if _i == len(axes)//2 - 1:
                axes[_i].get_images()[0].set_clim(-50, 50)
                axes[_i].get_images()[0].set_cmap('seismic')
        fig = plt.figure(300)
        ax = fig.add_subplot(111)
        ax.imshow(kwargs['Im'][N])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.tight_layout()
    return wrapper
chrom_plot = decorate(chromPlot)


# load chromophores data
# 0: wavelength, 1: HbO2, 2: Hb, 3: H2O, 4: Lipid, 5: melanin, 6: MetHb
cfile = open('C:/Users/luibe59/Documents/sfdi/common/models/visnir_chrom_400_1100nm with MetHb.txt', 'r')
chrom = np.genfromtxt(cfile, comments='%')
chrom_used = [1, 2, 5, 6]

WV5 = np.array([458, 520, 536, 556, 626])  # wavelengts (nm)

# select data path
path = getPath('Select data path')

# Load data from .npz
op_fit_maps = []
files = [x for x in os.listdir(path) if x.endswith('f0.npz') and 'calR' not in x]
files.sort()
titles = [y.split('_')[1] for y in files]  # file name
for file in files:
    data = np.load(f'{path}/{file}')
    op_fit_maps.append(data['op_fit_maps'])

# Load calibrated reflectance
calRs = []
files = [x for x in os.listdir(path) if x.endswith('calR.npz')]
if len(files) == 0:
    files = [x for x in os.listdir(path) if x.endswith('f0.npz')]
files.sort()

for file in files:
    data = np.load(f'{path}/{file}')
    calR = crop(data['cal_R'], data['ROI'])
    calR = np.squeeze(calR[:,:,(8,4,0),0])
    calRs.append(calR)
   
# sys.exit()

## Linear fitting
data = np.genfromtxt('C:/Users/luibe59/Documents/sfdi/common/overlaps_calibrated.csv',
                       delimiter=',') # load overlaps spectrum
spec = data[(9,6,5,4,1),:] # Keep 5 channels in order from BB to RR
wv = data[0,:] # wavelength axis
# need to interpolate to the same wv as the spectral bands
f = interp1d(chrom[:,0], chrom[:,chrom_used], kind='linear',
              axis=0, fill_value='extrapolate')
chrom = f(wv).T # chromophores used, full spectrum [380-720]nm
E = np.zeros((len(WV5), len(chrom_used))) # initialize


# TODO: Nested for loop, very inefficient. Try to optimize
start = time.time()
linear = False
chrom_maps = []
for o, op_map in enumerate(op_fit_maps):
    print('Fitting data set {}'.format(o))
    for i,band in enumerate(spec):
        for j,cr in enumerate(chrom):
            E[i,j] = np.sum(cr*band) / np.sum(band)
    
    if (len(chrom_used) == 1):
        E = E[:,np.newaxis] # This way is forced to be a column matrix
    
    if linear:
        # some linear algebra: inv_E = (E' * E)^-1 * E'
        #inv_E = np.array((E.T @ E).I @ E.T) # try with @ operator (matrix multiplication)
        inv_E = np.linalg.pinv(E) # pseudo inverse of E. It is the same as the old code 
        if op_map.ndim == 4: # in SFDI is a 4D array
            # Here use reshape() to multiply properly across the last two dimensions
            chrom_map = np.reshape(inv_E,(1,1,inv_E.shape[0],inv_E.shape[1])) @ op_map[:,:,:,0:1]
        else:
            chrom_map = inv_E @ op_map[:,0,None] # assuming they are in 2D
    else:
        chrom_map = np.zeros((op_map.shape[0], op_map.shape[1], len(chrom_used)+1), dtype=float)
        for _i in range(op_map.shape[0]):
            for _j in range(op_map.shape[1]):
                temp = lsq_linear(E, op_map[_i,_j,:,0], bounds=(0,np.inf), max_iter=100, method='bvls')
                chrom_map[_i, _j, :-1] = temp.x  # solution
                res = np.sum(temp.fun) / np.sum(op_map[_i,_j,:,0]) * 100  # residuals (in %)
                chrom_map[_i, _j, -1] = res
                # print('{} ({}, {}): niter={}'.format(o,_i,_j,temp.nit))  # DEBUG
        # print('### END ###')
                
    chrom_maps.append(np.squeeze(chrom_map))
end = time.time()
print('Elapsed time: {}'.format(end-start))

par = {'chrom_used': chrom_used}
chrom_plot(chrom_maps, titles, par, outliers=False, dataset=9, Im=calRs)

# Select a ROI to plot optical properties and Hb spectrum
if True:
    dataset=9
    binning = 8
    cv.namedWindow('Select ROI', cv.WINDOW_NORMAL)
    cv.resizeWindow('Select ROI', chrom_maps[dataset].shape[1] * 5, chrom_maps[dataset].shape[0] * 5)
    ROI = cv.selectROI('Select ROI', op_fit_maps[dataset][:,:,-1,0]*4)
    cv.destroyAllWindows()
    
    # fraction of cromophores in ROI
    fractions = np.mean(crop(chrom_maps[dataset][:,:,:-1], ROI), axis=(0,1))
    # average absorption in ROI
    mua = np.mean(crop(op_fit_maps[dataset][:,:,:,0], ROI), axis=(0,1))
    
    plt.figure(111)
    plt.plot(wv, np.sum(fractions[:,np.newaxis] * chrom, axis=0), '-b',
             linewidth=1.5, label='chrom_spectrum' ) # sum of chromophores in ROI
    plt.plot(WV5, mua, 'or', markerfacecolor='none', label=r'measured $\mu_a$')
    plt.plot(WV5, np.sum((E*fractions),axis=1), '^', markerfacecolor='none',
             markeredgecolor='limegreen', label='emulated chromophores')
    plt.xlabel('wavelength (nm)')
    plt.xlim([450,650])
    # plt.ylim([0,0.4])
    plt.legend(loc=1)
    plt.grid(True, which='both', linestyle=':')
    
    
## New plot
labels = ['', r'HbO$_2$','Hb',r'H$_2$O','lipid','melanin', 'MetHb']
labels = [labels[_i] for _i in chrom_used]
fig, ax = plt.subplots(2,4, num=1000, figsize=(16,6))
for _i in range(chrom_maps[dataset].shape[-1] - 1):
    im = ax[0,_i].imshow(chrom_maps[dataset][:,:,_i], cmap='magma', vmax=0.005)
    ax[0,_i].set_title(labels[_i])
    ax[0,_i].axis('off')
    cb = colourbar(im)
if len(chrom_used) < 4:
    for _i in range(chrom_maps[dataset].shape[-1] - 1, 4):
        ax[0,_i].set_axis_off()

im = ax[1,0].imshow(chrom_maps[dataset][:,:,-1], cmap='seismic', vmin=-50, vmax=50)
cb = colourbar(im)
cb.set_ticks([-50,-25,0,25,50])
cb.ax.set_xlabel(r'%')
cb.ax.xaxis.set_label_position('top')
ax[1,0].set_title('residuals')
ax[1,0].set_axis_off()

ax[1,1].imshow(calRs[dataset])
ax[1,1].set_axis_off()
rect = patches.Rectangle((ROI[0]*binning, ROI[1]*binning), ROI[2]*binning,
                         ROI[3]*binning, linewidth=1.5, linestyle='--',
                         edgecolor='limegreen', facecolor='none')
ax[1,1].add_patch(rect)

ax[1,2].plot(wv, np.sum(fractions[:,np.newaxis] * chrom, axis=0), '-b',
         linewidth=1.5, label='chrom_spectrum' ) # sum of chromophores in ROI
ax[1,2].plot(WV5, mua, 'or', markerfacecolor='none', label=r'measured $\mu_a$')
ax[1,2].plot(WV5, np.sum((E*fractions),axis=1), '^', markerfacecolor='none',
         markeredgecolor='limegreen', label='emulated chromophores')
ax[1,2].set_xlabel('wavelength (nm)')
ax[1,2].set_xlim([450, 650])
ax[1,2].set_ylim([0, 0.5])
ax[1,2].legend(loc=1)
ax[1,2].grid(True, which='both', linestyle=':')

text = '\n'.join('{:8}: {:.6f}'.format(a,b) for a,b in zip(labels, fractions))
ax[1,-1].text(0,0.1, text, family='monospace', size='large')
ax[1,-1].axis('off')

plt.tight_layout()
