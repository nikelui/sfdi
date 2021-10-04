# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:38:21 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

import numpy as np

def rebin(data, binsize):
    """Function to efficiently bin a large data set with the reshape method (avoid for loops).
If the dimensions are not multiples of nbin, it will crop the data."""
    data = data[:(data.shape[0] // binsize) * binsize,:(data.shape[1] // binsize) * binsize] # Crop the data
    old_shape = data.shape
    
    new_shape = (data.shape[0]//binsize, data.shape[1]//binsize) # binned shape
    # shape in higher dimension
    shape = [new_shape[0],data.shape[0]//new_shape[0],new_shape[1],data.shape[1]//new_shape[1]]
    
    if len(old_shape) > 2:
        for i in range(len(old_shape)-2):
            shape.append(old_shape[i+2])
    
    temp = data.reshape(tuple(shape)).mean(3).mean(1)
    
    return temp
    
    
if __name__ == '__main__':
    import cv2 as cv
    I1 = cv.imread('C:/Users/luibe59/Desktop/g1.jpg',cv.IMREAD_COLOR)
    I2 = cv.imread('C:/Users/luibe59/Desktop/g2.jpg',cv.IMREAD_COLOR)
    I3 = cv.imread('C:/Users/luibe59/Desktop/g3.jpg',cv.IMREAD_COLOR)
    
    Im = np.stack([I1,I2,I3],axis=3)
    
    Imc = rebin(Im,6)
    
    cv.namedWindow('one',cv.WINDOW_NORMAL)
    cv.namedWindow('two',cv.WINDOW_NORMAL)
    cv.namedWindow('three',cv.WINDOW_NORMAL)

    cv.resizeWindow('one',900,900)
    cv.resizeWindow('two',900,900)
    cv.resizeWindow('three',900,900)

    cv.imshow('one',Imc[:,:,:,0].astype('uint8'))
    cv.imshow('two',Imc[:,:,:,1].astype('uint8'))
    cv.imshow('three',Imc[:,:,:,2].astype('uint8'))
    


    cv.waitKey(0)
    cv.destroyAllWindows()