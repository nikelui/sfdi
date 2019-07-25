# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:37:05 2019

@author: luibe59
"""
import numpy as np

def sinPattern(xRes,yRes,w,f,p,B=255,correction=np.array([]),channels='rgb'):
    """Function to generate a 2D sinusoidal pattern in the x direction.
@Inputs:
- xRes -> projector horizontal resolution
- yRes -> projector vertical resolution
- w -> width of projected screen (in mm)
- f -> spatial frequency of sinusoid (cycles per mm)
- p -> phase delay of sinusoid (for demodulation use multiples of 2/3 pi)
- B -> brightness of the pattern (should be B <= 255 to avoid saturation)
- correction -> a 256 element numpy array with gamma correction for non linear output
- channels -> A string containing the color channels to give in output
@Return:
- I -> Grayscale
- Ir -> RED image
- Ig -> GREEN image
- Ib -> BLUE image
Grayscale is a 2D float array of size (yRes,xRes) with values normalized to [0,1]
Color images are 3D float arrays of size (yRes,xRes,3) with values normalized to [0,1]
"""
    x = np.linspace(0,xRes-1,xRes) # x coordinates from 0 to xRes
    y = (0.5*np.cos(x/xRes * 2*np.pi * f * w + p) + 0.5) * B # 1D perfect sinusoid
    
    if not correction.size == 0:
        idx = np.array([int(g) for g in np.ceil(y)]) # indicize
        y = correction[idx] / correction[B]*B /255 # use look-up values to correct gamma and normalize to [0,1]
    else:
        y = y / 255 # Normalize to [0,1]

    ## 2D Pattern
    I = np.tile(y,[yRes,1]) # 2D grayscale pattern
    O = np.zeros([yRes,xRes])
    Ir = Ig = Ib = []
    if 'r' in channels:
        Ir = np.stack((I,O,O),axis=2) # RED channel
    if 'g' in channels:
        Ig = np.stack((O,I,O),axis=2) # GREEN channel
    if 'b' in channels:
        Ib = np.stack((O,O,I),axis=2) # BLUE channel
    
    return(I,Ir,Ig,Ib)