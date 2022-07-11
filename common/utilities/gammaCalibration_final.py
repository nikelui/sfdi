# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 13:56:39 2022

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

More generalized gamma calibration script
"""
import time
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import cv2 as cv

from sfdi.common.readParams import readParams
# from sfdi.camera.xiCam import xiCam as Camera
from sfdi.camera.DummyCam import DummyCam as Camera
from sfdi.acquisition.setWindow import setWindow
from sfdi.processing.crop import crop
from sfdi.acquisition import __path__ as par_path

## Read parameters from .ini file
par = readParams('{}/parameters.ini'.format(par_path[0]))

ROI = np.array([])  # write ROI here for calculation (x,y,width,heigth)

cam = Camera(num=0, fps=30)  # set-up camera
cam.setExposure(50)

levels = np.arange(0,256,5)
setWindow('pattern', size=(par['xres'],par['yres']),pos=(par['w'], 0))  # Set-up window on second monitor

if False:
    raw_data = np.array([])
    for level in levels:
        Im = np.ones((par['yres'], par['xres']), dtype='uint8') * level
        cv.imshow('pattern', Im)
        time.sleep(0.5)
        frame = cam.preview()
        np.append(raw_data, np.mean(crop(frame, ROI),axis=[0,1]))
        
    
else:
    # For checking saturation, ROI
    Im = np.ones((par['yres'], par['xres']), dtype='uint8')
    cv.imshow('pattern', Im)
    cv.waitKey(0)
    frame = cam.capture()
    roi = cv.selectROI('ROI', frame)
    print('ROI: {}'.format(roi))

cv.destroyAllWindows()