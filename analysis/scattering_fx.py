# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 10:49:15 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Script to fit mus' to a power law of the kind A * lambda^(-b), select a ROI and
compare the variation at different fx
"""
import os, sys, re
from datetime import datetime
import pickle
# import json
import numpy as np
from scipy.io import loadmat  # new s tandard: work with Matlab files for compatibility
from scipy.optimize import curve_fit

from sfdi.common.getPath import getPath
from sfdi.analysis.dataDict import dataDict  # moved class to other file
from sfdi.common.readParams import readParams
from sfdi.common import models

# support functions
def save_obj(obj, name, path):
    """Utility function to save python objects using pickle module"""
    with open('{}/obj/{}.pkl'.format(path, name), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, path):
    """Utility function to load python objects using pickle module"""
    with open('{}/obj/{}.pkl'.format(path, name), 'rb') as f:
        return pickle.load(f)

def fit_fun(lamb, a, b):
    """Power law function to fit data to"""
    return a * np.power(lamb, -b)

data_path = getPath('Select data path')
par = readParams('{}/processing_parameters.ini'.format(data_path))  # optional
if 'wv' in par.keys():
    wv = par['wv']
else:
    wv = np.array([458, 520, 536, 556, 626])  # wavelengts (nm). Import from params?
regex = re.compile('.*f\d*\.mat')  # regular expression for optical properties
regex2 = re.compile('.*calR.mat')  # regular expression for calibrated reflectance
regex3 = re.compile('SFDS.*\.mat')  # regular expression for SFDS data
regex4 = re.compile('f\d')  # get frequency range with regex

# If the dataset has already been processed, load it
if '-load' in sys.argv and os.path.exists('{}/obj/dataset.pkl'.format(data_path)):
    data = load_obj('dataset', data_path)
    # data.par = par  # This should be already saved in the pickle
# If you need to process / modify it. NOTE: the old set will be overwritten
else:
    files = [x for x in os.listdir(data_path) if re.match(regex, x)]
    datasets = set(x.split('_')[-3] for x in files)  # sets have unique values
    
    sfds_path = [x for x in os.listdir(data_path) if re.match(regex3, x)]  # should be only one
    if sfds_path:  # need a check, because it might not exist
        sfds = loadmat('{}/{}'.format(data_path,sfds_path[0]))
        par['wv_sfds'] = np.squeeze(sfds['wv'])
    # datasets = set(x for x in sfds.keys() if 'AlO' in x or 'TiO' in x)  # DEBUG (sfds-only)
    data = dataDict()
    data.par = par
    # load the SFDI data into a custom dictionary
    start = datetime.now()  # calculate execution time
    for _d, dataset in enumerate(datasets, start=1):
        data[dataset] = {}  # need to initialize it
        temp = [x for x in files if dataset in x]   # get filenames
        # freqs = [x.split('_')[-1][:-4] for x in temp]  # get frequency range
        
        freqs = [x for x in par.keys() if re.match(regex4, x)]
        for file,fx in zip(temp, freqs):
            data[dataset][fx] = loadmat('{}/{}'.format(data_path, file))
            # here fit the data
            print('Fitting dataset {}_{}...[{} of {}]'.format(dataset, fx, _d, len(datasets)))
            # SFDI data
            op_map = data[dataset][fx]['op_fit_maps']  # for convenience
            p_map = np.zeros((op_map.shape[0], op_map.shape[1], 2), dtype=float)  # initialize
            for _i in range(op_map.shape[0]):
                for _j in range(op_map.shape[1]):
                    try:
                        temp, _ = curve_fit(fit_fun, wv, op_map[_i,_j,:,1], p0=[10, 1],
                                            method='trf', loss='soft_l1', max_nfev=2000)
                    except RuntimeError:
                        continue
                    p_map[_i, _j, :] = temp
            data[dataset][fx]['par_map'] = p_map
        # SFDS data
        if sfds_path and dataset in sfds.keys():
            for fx in freqs:
                data[dataset][fx] = {}  # DEBUG - sfds only
                data[dataset][fx]['sfds'] = {}
                data[dataset][fx]['sfds']['op_fit'] = sfds[dataset][:,freqs.index(fx),:]
                temp, _ = curve_fit(fit_fun, par['wv_sfds'], data[dataset][fx]['sfds']['op_fit'][:,1],
                                    p0=[10, 1], method='trf', loss='soft_l1', max_nfev=2000)
                data[dataset][fx]['sfds']['par'] = temp
            
    end = datetime.now()
    print('Elapsed time: {}'.format(str(end-start)))
    # save fitted dataset to file for easier access
    if not os.path.isdir('{}/obj'.format(data_path)):
        os.makedirs('{}/obj'.format(data_path))
    save_obj(data, 'dataset', data_path)


# Post- processing
# data.mask_on()  # mask outliers
# data.plot_cal('AlO05ml', data_path)
# data.plot_op_sfds('TiO20ml', f=[0,1,2,3,4])
# ret = data.singleROI('TiObase', norm=-1, fit='single', f=[0,1,2,3,4])
ret = data.singleROI('SC1', norm=None, fit='single', f=[0,1,2,3], I=2e3)
# ret = data.singleROI('wound2', norm=None, fit="single", f=[0,1,2,3,4], I=2e3, zoom=3)
# ret['par_ave'] = ret['par_ave'].T
# ret['par_std'] = ret['par_std'].T
# temp = np.stack([ret['op_ave'][:,:,0],ret['op_std'][:,:,0],
#                  ret['op_ave'][:,:,1],ret['op_std'][:,:,1]], axis=2)  # for ease of copying
# temp2 = np.stack([ret['op_ave'][:,2,0],ret['op_std'][:,2,0]])
# data.plot_op('wound4', f=[0])
# data.plot_cal('wound4', data_path)
#%% To save data
from sfdi.processing.crop import crop
from scipy.io import savemat, loadmat
import os
# ROI = ret['ROI']
# keys = ['AlObaseTop', 'TiObaseTop', 'TiO05ml', 'TiO10ml', 'TiO15ml', 'TiO20ml', 'TiO30ml']
# keys = ['SC1', 'SC2', 'SC3', 'K1', 'K2', 'CTL1', 'CTL2', 'CTL3', 'CTL4']
keys = ['K1', 'K2', 'SC1', 'SC2', 'CTL1', 'CTL2']
FX = ['f{}'.format(x) for x in range(5)]
out_mean = {}
out_std = {}

for key in keys:
    ret = data.singleROI(key, norm=None, fit='single', f=[0,1,2,3], I=2e3)
    ROI = ret['ROI']
    out_mean[key] = {}
    out_std[key] = {}
    temp = np.zeros((len(FX), 5, 2))
    stemp = np.zeros((len(FX), 5, 2))
    for _f,F in enumerate(FX):
        temp[_f,:,:] = np.mean(crop(data[key][F]['op_fit_maps'], ROI), axis=(0,1))
        stemp[_f,:,:] = np.std(crop(data[key][F]['op_fit_maps'], ROI), axis=(0,1))
    out_mean[key] = temp
    out_std[key] = stemp

out_path = '{}/test/'.format(data_path)

if not os.path.exists(out_path):
    os.makedirs(out_path)

# savemat('{}exVivo_mean.mat'.format(out_path), out_mean)
# savemat('{}exVivo_std.mat'.format(out_path), out_std)
# asd = loadmat('{}batch3_mean.mat'.format(out_path), struct_as_record=True, squeeze_me=True)

#%% Add SDFS only
if False:
    sfds_path = [x for x in os.listdir(data_path) if re.match(regex3, x)]  # should be only one
    par = readParams('{}/processing_parameters.ini'.format(data_path))  # optional
    if sfds_path:  # need a check, because it might not exist
        sfds = loadmat('{}/{}'.format(data_path,sfds_path[0]))
        par['wv_sfds'] = np.squeeze(sfds['wv'])
        data.par = par
    datasets = list(x for x in sfds.keys() if 'AlO' in x or 'TiO' in x)  # DEBUG (sfds-only)
    datasets.sort()
    # do an average if multiple SFDS measurements are present
    if True:  # change dimensions as appropriate
        datasets = [[x for x in datasets if x.endswith('_1')],
                    [x for x in datasets if x.endswith('_2')]]
    temp = {}
    for _d in range(len(datasets[0])):
        temp[datasets[0][_d][:-2]] = np.mean(np.array([sfds[datasets[0][_d]],sfds[datasets[1][_d]]]), axis=0)
    sfds = temp
    
    freqs = [x for x in par.keys() if re.match(regex4, x)]
    for dataset in sfds.keys():
        for fx in freqs:
            # data[dataset][fx] = {}  # DEBUG - sfds only. WARNING, this will delete SFDI data if not careful
            data[dataset][fx]['sfds'] = {}
            data[dataset][fx]['sfds']['op_fit'] = sfds[dataset][:,freqs.index(fx),:]
            temp, _ = curve_fit(fit_fun, par['wv_sfds'], data[dataset][fx]['sfds']['op_fit'][:,1],
                                p0=[10, 1], method='trf', loss='soft_l1', max_nfev=2000)
            data[dataset][fx]['sfds']['par'] = temp
        

#%% plots of fluence
if False:
    from matplotlib import pyplot as plt
    from matplotlib import cm
    import addcopyfighandler
    
    dz = 0.01  # resolution
    asd = loadmat(f'{data_path}/SFDS_8fx.mat')
    fx = np.array([np.mean(par['fx'][i:i+4]) for i in range(len(par['fx'])-3)])
    z = np.arange(0, 10, dz)
    lamb = 500  # nm
    WV = np.where(asd['wv'][:,0] >= lamb)[0][0]
    F = 0  # spatial frequency to plot
    
    phi_diff = {}  # diffusion
    phi_diffusion = {}  # diffusion, Seo
    phi_deltaP1 = {}  # delta-P1, Vasen modified
    phi_dp1 = {}  # delta-P1, Seo original
    
    keys = [x for x in asd.keys() if 'TiO' in x or 'AlObaseTop' in x]
    # keys.remove('TiObaseBottom')
    keys.sort()
    
    col = cm.get_cmap('Blues', len(keys))
    
    fig1, ax1 = plt.subplots(1,1, num=1, figsize=(6,4))
    fig2, ax2 = plt.subplots(1,1, num=2, figsize=(6,4))
    fig3, ax3 = plt.subplots(1,1, num=3, figsize=(6,4))
    fig4, ax4 = plt.subplots(1,1, num=4, figsize=(6,4))
    
    for _i, key in enumerate(keys):
        if _i == 0:
            color = '#FF0000'
        elif _i == len(keys)-1:
            color = '#00FF00'
        else:
            color = col(_i)
        
        phi_diff[key] = models.phi_diff(z, asd[key][:,:,0].T, asd[key][:,:,1].T/0.2, fx)  # diffusion
        phi_diffusion[key] = models.phi_diffusion(z, asd[key][:,:,0].T, asd[key][:,:,1].T/0.2, fx)  # diffusion, Seo
        phi_deltaP1[key] = models.phi_deltaP1(z, asd[key][:,:,0].T, asd[key][:,:,1].T/0.2, fx)  # d-p1, Luigi
        phi_dp1[key] = models.phi_dP1(z, asd[key][:,:,0].T, asd[key][:,:,1].T/0.2, fx)  # d-p1, Seo
        
        ax1.plot(z, phi_diff[key][F,WV,:], color=color, label=key)
        ax2.plot(z, phi_dp1[key][F,WV,:], color=color, label=key)
        ax3.plot(z, phi_deltaP1[key][F,WV,:], color=color, label=key)
        ax4.plot(z, phi_diffusion[key][F,WV,:], color=color, label=key)
    
    ax1.legend()
    ax1.set_xlabel('mm')
    ax1.set_ylabel(r'$\varphi$', fontsize=14)
    ax1.set_title('Diffusion')
    ax1.grid(True, linestyle=':')
    ax1.set_xlim([0,6])
    ax2.legend()
    ax2.set_xlabel('mm')
    ax2.set_ylabel(r'$\varphi$', fontsize=14)
    ax2.set_title(r'$\delta-P1$ - Seo thesis')
    ax2.grid(True, linestyle=':')
    ax2.set_xlim([0,6])
    ax3.legend()
    ax3.set_xlabel('mm')
    ax3.set_ylabel(r'$\varphi$', fontsize=14)
    ax3.set_title(r'$\delta-P1$ - Vasen modified')
    ax3.grid(True, linestyle=':')
    ax3.set_xlim([0,6])
    ax4.legend()
    ax4.set_xlabel('mm')
    ax4.set_ylabel(r'$\varphi$', fontsize=14)
    ax4.set_title('Diffusion - Seo')
    ax4.grid(True, linestyle=':')
    ax4.set_xlim([0,6])
    
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    
    #%% iterate and fit for alpha, beta - diffusion
    from scipy.optimize import minimize
    from scipy.optimize import Bounds
    
    keys = [x for x in data.keys() if 'TiO' in x]
    # keys.remove('TiObaseBottom')
    # keys.remove('TiObaseTop')
    keys.sort()
    F = 0  # spatial frequency
    lam = 0  # Tikhonov regularization
    
    phi_TiO = phi_diff['TiObaseTop'][F,WV,:]  # for now @ 500nm
    phi_AlO = phi_diff['AlObaseTop'][F,WV,:]
    phi_meas = [phi_diff[key][F,WV,:] for key in keys]
    x = np.zeros((len(keys), 2))
    
    # function to minimize: least square error
    min_fun = lambda x, phi_top, phi_bottom, phi_meas: np.sum(
        (models.phi_2lc_2(x, phi_bottom, phi_top) - phi_meas)**2) + np.sum((lam*x)**2)
    
    for _i, meas in enumerate(phi_meas):
        res = minimize(min_fun,  # target function
                       np.array([.8, .8]),  # initial guess of alpha
                       args=(phi_TiO, phi_AlO, meas),
                       method='Nelder-Mead',
                       bounds=Bounds(lb=0, ub=1),
                       options= {'maxiter':500})
        x[_i,:] = res.x
    
    #%% plots of fluence - 2layer model
    # from matplotlib import pyplot as plt
    # from matplotlib import cm
    # import addcopyfighandler
    
    # dz = 0.01  # resolution
    # asd = loadmat(f'{data_path}/SFDS_8fx.mat')
    # fx = np.array([np.mean(par['fx'][i:i+4]) for i in range(len(par['fx'])-3)])
    # z = np.arange(0, 10, dz)
    # lamb = 500  # nm
    # WV = np.where(asd['wv'][:,0] >= lamb)[0][0]
    
    # # alpha = np.arange(.9,0,-0.2)
    # alpha = x
    
    # phi_2ldiff = {}  # diffusion
    # phi_2ldeltaP1 = {}  # delta-P1, Vasen modified
    # phi_2ldp1 = {}  # delta-P1, Seo original
    
    # keys = ['AlObaseTop', '0.1', '0.3', '0.5', '0.7', '0.9', 'TiObaseTop']
    # col = cm.get_cmap('Blues', len(keys))
    
    # fig1, ax1 = plt.subplots(1,1, num=1, figsize=(6,4))
    # fig2, ax2 = plt.subplots(1,1, num=2, figsize=(6,4))
    # fig3, ax3 = plt.subplots(1,1, num=3, figsize=(6,4))
    
    # phi_2ldiff['AlObaseTop'] = models.phi_diff(z, asd['AlObaseTop'][:,:,0].T, 
    #                                           asd['AlObaseTop'][:,:,1].T/0.2, fx)  # diffusion
    # phi_2ldeltaP1['AlObaseTop'] = models.phi_deltaP1(z, asd['AlObaseTop'][:,:,0].T,
    #                                                 asd['AlObaseTop'][:,:,1].T/0.2, fx)  # d-p1, Luigi
    # phi_2ldp1['AlObaseTop'] = models.phi_dP1(z, asd['AlObaseTop'][:,:,0].T,
    #                                         asd['AlObaseTop'][:,:,1].T/0.2, fx)  # d-p1, Seo
    
    # phi_2ldiff['TiObaseTop'] = models.phi_diff(z, asd['TiObaseTop'][:,:,0].T, 
    #                                           asd['TiObaseTop'][:,:,1].T/0.2, fx)  # diffusion
    # phi_2ldeltaP1['TiObaseTop'] = models.phi_deltaP1(z, asd['TiObaseTop'][:,:,0].T,
    #                                                 asd['TiObaseTop'][:,:,1].T/0.2, fx)  # d-p1, Luigi
    # phi_2ldp1['TiObaseTop'] = models.phi_dP1(z, asd['TiObaseTop'][:,:,0].T,
    #                                         asd['TiObaseTop'][:,:,1].T/0.2, fx)  # d-p1, Seo
    # for _i, key in enumerate(keys):
    #     if _i == 0:
    #         color = '#FF0000'
    #         ax1.plot(z, phi_2ldiff[key][0,WV,:], color=color, label=key)
    #         ax2.plot(z, phi_2ldp1[key][0,WV,:], color=color, label=key)
    #         ax3.plot(z, phi_2ldeltaP1[key][0,WV,:], color=color, label=key)
    #     elif _i == len(keys)-1:
    #         color = '#00FF00'
    #         ax1.plot(z, phi_2ldiff[key][0,WV,:], color=color, label=key)
    #         ax2.plot(z, phi_2ldp1[key][0,WV,:], color=color, label=key)
    #         ax3.plot(z, phi_2ldeltaP1[key][0,WV,:], color=color, label=key)
    #     else:
    #         color = col(_i)
    #         phi_2ldiff[key] = models.phi_2lc_2(alpha[_i-1], phi_2ldiff['AlObaseTop'], phi_2ldiff['TiObaseTop'])  # diffusion
    #         phi_2ldeltaP1[key] = models.phi_2lc_2(alpha[_i-1], phi_2ldeltaP1['AlObaseTop'], phi_2ldeltaP1['TiObaseTop'])  # d-p1, Luigi
    #         phi_2ldp1[key] = models.phi_2lc_2(alpha[_i-1], phi_2ldp1['AlObaseTop'], phi_2ldp1['TiObaseTop'])  # d-p1, Seo
        
    #         ax1.plot(z, phi_2ldiff[key][0,WV,:], color=color,
    #                  label=r'$\alpha$={:.3f}, $\beta$={:.3f}'.format(alpha[_i-1,0], alpha[_i-1,1]))
    #         ax2.plot(z, phi_2ldp1[key][0,WV,:], color=color,
    #                  label=r'$\alpha$={:.3f}, $\beta$={:.3f}'.format(alpha[_i-1,0], alpha[_i-1,1]))
    #         ax3.plot(z, phi_2ldeltaP1[key][0,WV,:], color=color,
    #                  label=r'$\alpha$={:.3f}, $\beta$={:.3f}'.format(alpha[_i-1,0], alpha[_i-1,1]))
    
    # ax1.legend()
    # ax1.set_xlabel('mm')
    # ax1.set_ylabel(r'$\varphi$', fontsize=14)
    # ax1.set_title('Diffusion')
    # ax1.grid(True, linestyle=':')
    # ax1.set_xlim([0, 6])
    # ax2.legend()
    # ax2.set_xlabel('mm')
    # ax2.set_ylabel(r'$\varphi$', fontsize=14)
    # ax2.set_title(r'$\delta-P1$ - Seo thesis')
    # ax2.grid(True, linestyle=':')
    # ax2.set_xlim([0, 6])
    # ax3.legend()
    # ax3.set_xlabel('mm')
    # ax3.set_ylabel(r'$\varphi$', fontsize=14)
    # ax3.set_title(r'$\delta-P1$ - Vasen modified')
    # ax3.grid(True, linestyle=':')
    # ax3.set_xlim([0, 6])
    
    # fig1.tight_layout()
    # fig2.tight_layout()
    # fig3.tight_layout()
    
    
    #%% plotting
    # from matplotlib import pyplot as plt
    # import addcopyfighandler
    
    # cal_path = [x for x in os.listdir(data_path) if 'calR' in x and 'TiObase' in x]
    # calR =loadmat(f'{data_path}/{cal_path[0]}')
    # calR = calR['cal_R']
    # H,W = calR.shape[:2]
    # Rd = np.nanmean(calR[H//2-10:H//2+10,W//2-10:W//2+10,:,:], axis=(0,1))
    # fx = np.arange(0, 0.51, 0.05)
    # # fx = np.array([np.mean(x) for x in [par[f'f{y}'] for y in range(8)]])
    # # wv_used = np.array([0,3,4,5,8])
    # plt.figure(22, figsize=(7,4))
    # plt.plot(fx, Rd[:,:].T)
    # plt.legend([r'{:d} nm'.format(x) for x in par['wv']])
    # plt.grid(True, linestyle=':')
    # plt.xlabel(r'Spatial frequency (mm$^{{-1}}$')
    # plt.xlim([0,0.5])
    # plt.title('Calibrated reflectance')
    # plt.tight_layout()
    
    # if False:
        
    #     from sfdi.common.phantoms import __path__ as ph_path
    #     ref = np.genfromtxt('{}/TS2.txt'.format(ph_path._path[0]))  # reference
    #     plt.figure(figsize=(10,4))
    #     labels = ['f0','f1','f2','f3','f4','f5','f6','f7']
    #     for _j in range(ret['op_ave'].shape[0]-3):
    #         plt.subplot(1,2,1)
    #         plt.errorbar(wv, ret['op_ave'][_j,:,0], yerr=ret['op_std'][_j,:,0], fmt='s', capsize=5,
    #                      linestyle='solid', linewidth=2, label=labels[_j])
    #         plt.grid(True, linestyle=':')
        
    #         plt.subplot(1,2,2)
    #         plt.errorbar(wv, ret['op_ave'][_j,:,1], yerr=ret['op_std'][_j,:,1], fmt='s', capsize=5,
    #                      linestyle='solid', linewidth=2, label=labels[_j])
    #         plt.grid(True, linestyle=':')
        
    #     plt.subplot(1,2,1)
    #     plt.plot(ref[:4,0], ref[:4,1], '*k', linestyle='--', label='reference', linewidth=2, zorder=100, markersize=10)
    #     plt.title(r'$\mu_a$')
    #     plt.xlabel('nm')
    #     plt.legend()
    #     plt.subplot(1,2,2)
    #     plt.plot(ref[:4,0], ref[:4,2], '*k', linestyle='--', label='reference', linewidth=2, zorder=100, markersize=10)
    #     plt.title(r"$\mu'_s$")
    #     plt.xlabel('nm')
    #     plt.tight_layout()
    
    #%%
    if False:
        for key in ['TiObase', 'TiO05ml', 'TiO10ml', 'TiO15ml', 'TiO20ml', 'TiO30ml', 'AlObase']:
            print(key)
            for fx in ['f0', 'f1', 'f2', 'f3', 'f4']:
                print('{} -> A: {:.2f}\tB:{:.4f}'.format(fx, data[key][fx]['sfds']['par'][0], data[key][fx]['sfds']['par'][1]))
