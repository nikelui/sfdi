# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:37:05 2019

@author: luibe59
"""
import numpy as np

def sinPattern(xRes,yRes,width,f,p,B=255,correction=np.array([]),channels='rgb',diagonal=False):
    """Function to generate a 2D sinusoidal pattern in the x direction.
@Inputs:
- xRes -> projector horizontal resolution
- yRes -> projector vertical resolution
- width -> width of projected screen (in mm)
- f -> spatial frequency of sinusoid (cycles per mm)
- p -> phase delay of sinusoid (for demodulation use multiples of 2/3 pi)
- B -> brightness of the pattern (should be B <= 255 to avoid saturation)
- correction -> a 256 element numpy array with gamma correction for non linear output
- channels -> A string containing the color channels to give in output
- diagonal -> A boolean flag to add support for diagonal pixels DMD. If true, alternate
              lines are shifted by 1 pixel in a zig-zag pattern.
@Return:
- I -> Grayscale
- Ir -> RED image
- Ig -> GREEN image
- Ib -> BLUE image
Grayscale is a 2D float array of size (yRes,xRes) with values normalized to [0,1]
Color images are 3D float arrays of size (yRes,xRes,3) with values normalized to [0,1]
"""
    x = np.arange(xRes) # x coordinates from 0 to xRes
    if diagonal:
        y1 = (0.5*np.cos(x/xRes * 2*np.pi * f * width + p) + 0.5) * B # 1D perfect sinusoid
        y2 = (0.5*np.cos((x-1)/xRes * 2*np.pi * f * width + p) + 0.5) * B # 1D perfect sinusoid
    else:
        y1 = (0.5*np.cos(x/xRes * 2*np.pi * f * width + p) + 0.5) * B # 1D perfect sinusoid
        y2 = (0.5*np.cos(x/xRes * 2*np.pi * f * width + p) + 0.5) * B # 1D perfect sinusoid
    
    if not correction.size == 0:
        idx = np.array([int(g) for g in np.ceil(y1)]) # indicize
        y1 = correction[idx] / correction[B]*B /255 # use look-up values to correct gamma and normalize to [0,1]
        idx = np.array([int(g) for g in np.ceil(y2)]) # indicize
        y2 = correction[idx] / correction[B]*B /255 # use look-up values to correct gamma and normalize to [0,1]
    else:
        y1 = y1 / 255 # Normalize to [0,1]
        y2 = y2 / 255

    ## 2D Pattern
    if diagonal:
        temp = np.stack((y1, y2), axis=0)  # first create a 2-line alternating pattern
        I = np.tile(temp, [yRes//2, 1])   # then stack it to form an image
        if yRes % 2 != 0:  # in case there are an odd number of lines
            I = np.vstack((I, y1))
    else:
        I = np.tile(y1,[yRes,1]) # 2D grayscale pattern
    O = np.zeros([yRes,xRes])
    Ir = Ig = Ib = []
    if 'r' in channels:
        Ir = np.stack((I,O,O),axis=2) # RED channel
    if 'g' in channels:
        Ig = np.stack((O,I,O),axis=2) # GREEN channel
    if 'b' in channels:
        Ib = np.stack((O,O,I),axis=2) # BLUE channel
    
    return(I,Ir,Ig,Ib)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    
    pattern,_,_,_ = sinPattern(720, 481, 110, 0.4, 2/3*np.pi, channels='',diagonal=False)
    pattern *= 255
    # pattern = np.ceil(pattern/4).astype('uint8')
    plt.suptitle(r'fx = 0.4mm$^{-1}$')
    plt.subplot(2,1,1)
    plt.imshow(pattern[0:30,0:100].astype('uint8'), cmap='gray')
    plt.title('8-bit')
    plt.subplot(2,1,2)
    pattern /= 4
    plt.imshow(pattern[0:30,0:100].astype('uint8'), cmap='gray')
    plt.title('6-bit')
    plt.tight_layout()
    
    
    #%% demodulation test
    pattern1,_,_,_ = sinPattern(720, 481, 110, 0.1, 0, channels='',diagonal=False)
    pattern2,_,_,_ = sinPattern(720, 481, 110, 0.1, 2/3*np.pi, channels='',diagonal=False)
    pattern3,_,_,_ = sinPattern(720, 481, 110, 0.1, 4/3*np.pi, channels='',diagonal=False)
    pattern1 *= 255/1
    pattern2 *= 255/1
    pattern3 *= 255/1
    dem = 2/3* np.sqrt(pattern1**2 + pattern2**2 + pattern3**2)
    
    
    plt.figure(11)
    plt.subplot(2,2,1)
    plt.imshow(pattern1[:50,:50].astype('uint8'), cmap='gray')
    plt.title('phase 0')
    plt.subplot(2,2,2)
    plt.imshow(pattern2[:50,:50].astype('uint8'), cmap='gray')
    plt.title('phase 1')
    plt.subplot(2,2,3)
    plt.imshow(pattern3[:50,:50].astype('uint8'), cmap='gray')
    plt.title('phase 2')
    plt.subplot(2,2,4)
    plt.imshow(dem[:50,:50].astype('uint8'), cmap='gray')
    plt.title('demodulate')
    plt.tight_layout()