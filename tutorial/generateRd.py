# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 16:25:53 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Script to generate demo SFDI datasets using diffusion approximation.
"""
import os
from time import time
import PIL
import numpy as np
from itertools import product
from Rd import diffuseReflectance
from matplotlib import pyplot as plt

##### Parameters #####

# Note: to pass multiple wavelength, mua, mus and n should be arrays with one
#       value for each wavelength. For compatibility, pass an array with a
#       single value for single wavelength.

# Optical properties, tissue 1
mua1 = [0.1, 0.1]  # mm^-1
mus1 = [5, 4]  # mm^-1
n1 = [1.4, 1.4]  # index of refraction

# Optical properties, tissue 2
mua2 = [0.5, 0.5]  # mm^-1
mus2 = [15, 10]  # mm^-1
n2 = [1.4, 1.4]  # index of refraction

# Spatial frequencies (output)
fx = [0, 0.1, 0.2, 0.3]  # mm^-1

# Wavelengths (corresponding to mua,mus,n values)
wv = [500, 550]  # nm

# Output images parameters
width = 640  # pixel
height = 480  # pixel
w = 100  # mm, physical dimension

nPhase = 3 # phases for demodulation

exptime = 50  # ms, dummy exposure time

basepath = './demo'  # base output path
dataname = 'test'  # dataset pathname
phantomname = 'calibration'  # dummy calibration pantom pathname
basename = 'im'  # images base name

##### END parameters #####

Reflectance = np.zeros((height, width, len(wv), len(fx)), dtype=float)  # Initialize Rd output
Phantom = np.ones((height, width, len(wv), len(fx)), dtype=float)  # Dummy calibration phantom
# Use the mask to "draw" shapes with two different tissues
mask = np.zeros((height, width), dtype=bool)  # default is all zero

#### Modify your boolean mask here to insert second tissue ####
mask[height//3:2*height//3,width//3:2*width//3] = True  # Test

## indexing
idx1 = np.where(mask == False)  # Tissue 1
idx2 = np.where(mask == True)  # Tissue 2

# Diffuse reflectance
Reflectance[idx1[0],idx1[1],:] = diffuseReflectance(mua1, mus1, n1, fx)
Reflectance[idx2[0],idx2[1],:] = diffuseReflectance(mua2, mus2, n2, fx)

# Plot images (debug)
if False:
    vmax = np.max(Reflectance)  # for consistent colorbar
    for i,f in enumerate(fx):
        plt.figure(i)
        plt.imshow(Reflectance[:,:,0,i], cmap='magma', vmax=vmax, vmin=0)
        plt.colorbar()
        plt.tight_layout()

# Sinusoidal patterns
sinpattern = np.zeros((height, width, len(wv), len(fx), nPhase), dtype=float)  # initialize
for i,(f,p) in enumerate(product(fx, range(nPhase))):
    # 1D to 2D indexing
    j = i // nPhase  # fx
    k = i % nPhase  # phase
    pattern1D = 0.5*np.cos(np.arange(width)/width * 2*np.pi * f * w + 2/nPhase*np.pi*p) + 0.5
    pattern2D = np.tile(pattern1D, [height,1])
    sinpattern[:,:,:,j,k] = pattern2D[:,:,np.newaxis]

# plot sinpatterns (debug)
if False:
    for i,j,k in product(range(len(wv)),range(len(fx)),range(nPhase)):
        plt.figure(100)
        plt.imshow(sinpattern[:,:,i,j,k], vmin=0, vmax=1, cmap='bone')
        if i == 0 and j == 0 and k == 0:
            plt.colorbar()
        plt.tight_layout()
        plt.pause(0.5)

# simulate sinusoidal patterns
out_Rd = Reflectance[:,:,:,:,np.newaxis] * sinpattern
out_Phantom = Phantom[:,:,:,:,np.newaxis] * sinpattern

# create save path if does not exist
tstamp = int(time())  # timestamp
datapath = f'{basepath}/{tstamp}_{dataname}_{exptime}ms'
os.makedirs(datapath, exist_ok=True)
phantompath = f'{basepath}/{tstamp}_{phantomname}_{exptime}ms'
os.makedirs(phantompath, exist_ok=True)

# plot output (debug) / save images
if True:
    for i,j,k in product(range(len(wv)),range(len(fx)),range(nPhase)):
        im = PIL.Image.fromarray(out_Rd[:,:,i,j,k] * 255)
        if im.mode == 'F':
            im = im.convert('RGB')
        im.save(f'{datapath}/{basename}_{i}{j}{k}.bmp')
        
        im = PIL.Image.fromarray(out_Phantom[:,:,i,j,k] * 255)
        if im.mode == 'F':
            im = im.convert('RGB')
        im.save(f'{phantompath}/{basename}_{i}{j}{k}.bmp')
        # plt.imsave(f'{datapath}/{basename}_{i}{j}{k}.bmp', out_Rd[:,:,i,j,k])
        # plt.imsave(f'{phantompath}/{basename}_{i}{j}{k}.bmp', out_Rd[:,:,i,j,k])
        if False:  # Debug plots
            plt.figure(100)
            plt.imshow(out_Rd[:,:,i,j,k], vmin=0, vmax=1, cmap='bone')
            if i == 0 and j == 0 and k == 0:
                plt.colorbar()
            plt.tight_layout()
            plt.pause(0.5)