# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 16:25:53 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Script to generate demo SFDI datasets using diffusion approximation.
"""
import numpy as np
from Rd import diffuseReflectance
from matplotlib import pyplot as plt

## Parameters
# Note: to pass multiple wavelength, mua, mus and n should be arrays with one
#       value for each wavelength. For compatibility, pass an array with a
#       single value for single wavelength.

# Optical properties, tissue 1
mua1 = [0.1]  # mm^-1
mus1 = [5]  # mm^-1
n1 = [1.4]  # index of refraction

# Optical properties, tissue 2
mua2 = [0.5]  # mm^-1
mus2 = [15]  # mm^-1
n2 = [1.4]  # index of refraction

# Spatial frequencies (output)
fx = [0, 0.1, 0.2, 0.3]  # mm^-1

# For now, single wavelength
wv = [500]  # nm

# Output images parameters
width = 640  # pixel
height = 480  # pixel

output = np.zeros((height, width, len(fx)), dtype=float)  # Initialize Rd output

mask = np.zeros((height, width), dtype=bool)  # default is all zero

#### Modify your boolean mask here to insert second tissue ####
mask[height//3:2*height//3,width//3:2*width//3] = True  # Test

## indexing
idx1 = np.where(mask == False)  # Tissue 1
idx2 = np.where(mask == True)  # Tissue 2

# Diffuse reflectance
output[idx1[0],idx1[1],:] = diffuseReflectance(mua1, mus1, n1, fx)
output[idx2[0],idx2[1],:] = diffuseReflectance(mua2, mus2, n2, fx)

# Plot images
vmax = np.max(output)  # for consistent colorbar
for i,f in enumerate(fx):
    plt.figure(i)
    plt.imshow(output[:,:,i], cmap='magma', vmax=vmax, vmin=0)
    plt.colorbar()
    plt.tight_layout()
    