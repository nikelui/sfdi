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
from sfdi.camera.xiCam import XiCam as Camera
# from sfdi.camera.DummyCam import DummyCam as Camera
from sfdi.acquisition.setWindow import setWindow
from sfdi.processing.crop import crop
from sfdi.acquisition import __path__ as par_path
import addcopyfighandler

## Read parameters from .ini file
par = readParams('{}/parameters.ini'.format(par_path[0]))

ROI = np.array([788, 175, 234, 127])  # write ROI here for calculation (x,y,width,heigth)

cam = Camera(num=0, fps=30)  # set-up camera
cam.setExposure(900)

levels = np.arange(0,256,5)
setWindow('pattern', size=(par['xres'],par['yres']),pos=(par['w'], 0))  # Set-up window on second monitor

if True:
    raw_data = np.array([])
    for level in levels:
        Im = np.ones((par['yres'], par['xres']), dtype='uint8') * level
        Im = 255 - Im  # Invert image
        cv.imshow('pattern', Im)
        cv.waitKey(1)
        time.sleep(1)
        # import pdb; pdb.set_trace()
        frame = cam.preview()
        raw_data = np.append(raw_data, np.mean(crop(frame, ROI),axis=(0,1)))
    np.save('XXX_gamma.npy', raw_data)  # change filename
        
    
else:
    # For checking saturation, ROI
    Im = np.ones((par['yres'], par['xres']), dtype='uint8') * 255
    Im = 255 - Im  # Invert image
    cv.imshow('pattern', Im)
    
    cv.waitKey(1)
    frame = cam.preview()
    roi = cv.selectROI('ROI', frame)
    print('ROI: {}'.format(roi))

cv.destroyAllWindows()
cam.close()

#%%
# TODO: plot gamma and calculate inverse
raw_data = np.load('XXX_gamma.npy')

raw_data -= np.min(raw_data)  # remove offset
raw_data *= (255 / np.max(raw_data))  # re-scale 0-255
levels_out = np.arange(0,256)
z = np.polyfit(levels, raw_data, deg=6)
f = np.poly1d(z)

levels_input = f(levels_out)
# plt.plot(levels, raw_data, '*r')
# plt.plot(levels_out, levels_input)
# plt.grid(True, linestyle=':')
# plt.xlabel('INPUT')
# plt.ylabel('OUTPUT')
# plt.legend(['raw_data', 'fitted curve'])

# invert the curve
# ff = interp1d(levels_input, levels_out, kind='linear',
#               assume_sorted=True, fill_value='extrapolate')
zz = np.polyfit(raw_data, levels, deg=8)
ff = np.poly1d(zz)
gamma_correction = ff(levels_out)
gamma_correction -= np.min(gamma_correction)
gamma_correction *= (255 / np.max(gamma_correction))

plt.figure()
# plt.plot(raw_data, levels, '*r')
plt.plot(levels_out, levels_out, ':k')
plt.plot(levels_out, gamma_correction, '-g')
plt.plot(levels_out, levels_input, '-b')
plt.grid(True, linestyle=':')
plt.xlabel('INPUT')
plt.ylabel('OUTPUT')
plt.legend(['raw_data', 'fitted curve'])

# plt.figure()
# plt.plot(levels_out, gamma_correction, '-b')
# plt.grid(True, linestyle=':')
# plt.xlabel('INPUT')
# plt.ylabel('OUTPUT')

np.savetxt('gammaCorrection_keynote.csv', gamma_correction, fmt='%.5f', delimiter=',',
            newline=',')