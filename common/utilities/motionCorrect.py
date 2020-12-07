# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 13:34:56 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

A function to perform motion correction algorithms on demodulated SFDI data
"""
import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
sys.path.append('../')  # common folder
from stackPlot import stackPlot

def motionCorrect(AC, par, edge='laplacian', con=1, gauss=(0,0), debug=False):
    """A function to perform motion correction

The function will apply motion correction algorithms to the data set.
It is best to use it before fitting for optical properties. It might produce
better results if applied to raw data (instead of calibrated reflectance)

@input:
    - AC: demodulated data set
    - par: processing parameters
    - edge: edge detection algorithm to use ('sobel', 'laplacian'[default])
    - con: coefficient to increase contrast [default = 1]
    - gauss: gaussian filter params (ksize, stdev) [default = (0,0)]
    - debug: debugging flag [default = False]

@output:
    - AC_aligned: re-aligned data-set
"""   
    ## Some tests: edge detection
    edges = np.zeros(AC.shape, dtype=float)
    for _i in range(AC.shape[2]):
        for _j in range(AC.shape[3]):
            image = AC[:,:,_i,_j]
            if gauss != (0,0):
                image = cv.GaussianBlur(image,(gauss[0],gauss[0]),gauss[1])
            if edge == 'sobel':
                dx = cv.Sobel(image / np.max(image[20:-20,20:-20], axis=(0,1)),
                              cv.CV_64FC1, ksize=3, dx=1, dy=0)
                dy = cv.Sobel(image / np.max(image[20:-20,20:-20], axis=(0,1)),
                              cv.CV_64FC1, ksize=3, dx=0, dy=1)
                edges[:,:,_i,_j] = cv.addWeighted(src1=np.abs(dx), src2=np.abs(dy),
                                                  alpha=0.5, beta=0.5, gamma=0)
            elif edge == 'laplacian':
                edges[:,:,_i,_j] = cv.Laplacian(image / np.max(image[20:-20,20:-20],
                                 axis=(0,1)), cv.CV_32FC1, ksize=5)
            # NOTE: the edges are normalized to the max value, in a central region
            #       cropped 20px from the borders (to avoid edge artifacts)
            edges[:,:,_i,_j] *= con  # adjust contrast
    if debug:
        stackPlot(edges[:,:,par['wv_used'],:], cmap='Greys_r', num=102)
    
    #TODO: adjust this code to work with dataset
    ### FindTransformEEC
    warp_mode = cv.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Specify the number of iterations.
    number_of_iterations = 1000
    # Specify the threshold of the increment in the correlation 
    # coefficient between two iterations
    termination_eps = 1e-6
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 
                number_of_iterations, termination_eps)
    # Run the ECC algorithm. The results are stored in warp_matrix.
    im1 = edges[:,:,0,0].astype('float32')  # reference image
    AC_aligned = np.zeros(AC.shape, dtype=float)
    AC_aligned[:,:,0,0] = AC[:,:,0,0]  # the first is used as reference
    
    if debug:
        ff = open('{}transform.txt'.format(par['savefile']), 'w')
    
    for _i in par['wv_used']:  # only align used wavelengths for speed
        for _j in range(AC.shape[3]):
            if _i == 0 and _j == 0:
                pass
            else:
                im2 = edges[:,:,_i,_j].astype('float32')
                (cc, warp_matrix) = cv.findTransformECC(im1, im2, warp_matrix, 
                                                 warp_mode, criteria)
                if debug:  # print out to file
                    print('Frequency: {:4f}  Wavelength: {:d}\nMatrix:'.format(par['freqs'][_j],
                          par['wv'][_i]), file=ff)
                    for line in warp_matrix:
                        print('{:>7.4f} {:>7.4f} {:>5.1f}'.format(*line), file=ff)
                    print('', file=ff)
#                    print('Matrix:\n{:4f}\n'.format(np.matrix(warp_matrix)), file=ff)
                ## Apply transform
                # Get the target size from the desired image
                target_shape = im1.shape
                AC_aligned[:,:,_i,_j] = cv.warpAffine(
                                          AC[:,:,_i,_j], warp_matrix, 
                                          (target_shape[1], target_shape[0]), 
                                          flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,
                                          borderMode=cv.BORDER_CONSTANT, 
                                          borderValue=0)
    if debug:
        ff.close()  # close file
        # DEBUG plots
        #import pdb; pdb.set_trace()
        stackPlot(AC, num=200)
        stackPlot(AC_aligned, num=201)
        fig, ax = plt.subplots(1,2, num=301)
        im = np.stack([AC[:,:,8,0] / np.max(AC[:,:,8,0]),  # BLUE channel
                       AC[:,:,4,0] / np.max(AC[:,:,4,0]),  # GREEN channel
                       AC[:,:,0,0] / np.max(AC[:,:,0,0])   # RED channel
                ], axis=-1)
        ax[0].imshow(im)
        
        im_aligned = np.stack([AC_aligned[:,:,8,0] / np.max(AC_aligned[:,:,8,0]),  # BLUE channel
                               AC_aligned[:,:,4,0] / np.max(AC_aligned[:,:,4,0]),  # GREEN channel
                               AC_aligned[:,:,0,0] / np.max(AC_aligned[:,:,0,0])   # RED channel
            ], axis=-1)
        ax[1].imshow(im_aligned)

    return AC_aligned


if __name__ == '__main__':
    from sfdi.readParams3 import readParams
    from rawDataLoad import rawDataLoad
    from calibrate import calibrate
    
    par = readParams('../../processing/parameters.ini')

    if len(par['freq_used']) == 0: # use all frequencies if empty
        par['freq_used'] = list(np.arange(len(par['freqs'])))
    
    if len(par['wv_used']) == 0: # use all wavelengths if empty
        par['wv_used'] = list(np.arange(len(par['wv'])))
  
    ## Load tissue data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
    AC,name = rawDataLoad(par, 'Select tissue data folder')
    ## Load calibration phantom data. Note: if ker > 1 in the parameters, it will apply a Gaussian smoothing
#    ACph,_ = rawDataLoad(par, 'Select calibration phantom data folder')
#    ## Calibration step
#    cal_R = calibrate(AC, ACph, par)
    
    AC_aligned = motionCorrect(AC, par, edge='sobel', con=2, gauss=(7,5), debug=True)
#    C_aligned = motionCorrect(cal_R, par, edge='sobel', con=1)
