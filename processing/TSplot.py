# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:08:07 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

files = ['C:/Users/luibe59/Documents/AcquisitionCode/5wv_capture/dist_iris18/op18.npz',
         'C:/Users/luibe59/Documents/AcquisitionCode/5wv_capture/dist_iris28/op28.npz',
         'C:/Users/luibe59/Documents/AcquisitionCode/5wv_capture/dist_iris40/op40.npz',
         'C:/Users/luibe59/Documents/AcquisitionCode/5wv_capture/dist_iris60/op60.npz']

dists = list(range(5,-6,-1))
op_wv = [458,520,536,556,626]

labels = [str(x)+'nm' for x in op_wv]
colors = ['blue','cyan','limegreen','magenta','red']
titles = ['aperture=1.8','aperture=2.8','aperture=4.0','aperture=6.0']

# Reference optical properties for TS2
TS2 = np.genfromtxt('../common/phantoms/TS2.txt')
TS2 = TS2[0:5,0:3]
mua =  interp1d(TS2[:,0],TS2[:,1],kind='cubic',fill_value='extrapolate')
mus =  interp1d(TS2[:,0],TS2[:,2],kind='cubic',fill_value='extrapolate')
ref_ave = np.stack((mua(op_wv),mus(op_wv)),axis=1)

fig,ax = plt.subplots(num=1,figsize=(10,6),ncols=2,nrows=2)
fig.suptitle(r'$\mu_A$',fontsize=18)
for i,f in enumerate(files):
    temp = np.load(f)
    op_ave = temp['op_ave']    
    error = (op_ave - ref_ave) / ref_ave * 100
    
    for j,w in enumerate(op_wv):
        ax[i//2,i%2].plot(dists,error[:,j,0],'*',color=colors[j],linestyle='solid',label=labels[j])
    ax[i//2,i%2].set_title(titles[i])
    if(i>1):
        ax[i//2,i%2].set_xlabel('distance(mm)')
    ax[i//2,i%2].set_ylabel('error(%)')
    ax[i//2,i%2].grid(True,linestyle=':')
    ax[i//2,i%2].set_ylim([-30,20])
    if (i==0):
        ax[i//2,i%2].legend(ncol=2)
plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.095)

fig,ax = plt.subplots(num=2,figsize=(10,6),ncols=2,nrows=2)
fig.suptitle(r'$\mu_S$',fontsize=18)
for i,f in enumerate(files):
    temp = np.load(f)
    op_ave = temp['op_ave']    
    error = (op_ave - ref_ave) / ref_ave * 100
    
    for j,w in enumerate(op_wv):
        ax[i//2,i%2].plot(dists,error[:,j,1],'*',color=colors[j],linestyle='solid',label=labels[j])
    ax[i//2,i%2].set_title(titles[i])
    if(i>1):
        ax[i//2,i%2].set_xlabel('distance(mm)')
    ax[i//2,i%2].set_ylabel('error(%)')
    ax[i//2,i%2].grid(True,linestyle=':')
    ax[i//2,i%2].set_ylim([-10,10])
    if (i==0):
        ax[i//2,i%2].legend(ncol=2)
plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.095)