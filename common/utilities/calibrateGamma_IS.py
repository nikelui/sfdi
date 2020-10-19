# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:48:50 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"Spatial approach: create a linear gradient, capture it and extract the profile
"""

import datetime,time,sys
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import cv2 as cv
sys.path.append('../') # 'common' folder
from sfdi.readParams3 import readParams
from sfdi.setWindow import setWindow
from sfdi.crop import crop
from sfdi.IS.IS import ImagingSource


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


if True:
    par = readParams('parameters.ini') # Read calibration parameters
    
    ######## First part: acquire data ########
    cam = ImagingSource(num=0,res=par['res'],fps=par['fps'])
    cam.setExposure(8.3)
    win = setWindow('light',size=(par['xres'],par['yres']),pos=(par['w'],0)) # create window
    curve = np.zeros((10,26,3), dtype=float)
    cap = setWindow('capture',size=(640,480),pos=(0,0)) # create window
    
    snap = cam.capture()
    ROI = cv.selectROI('ROI', snap)
    
    ## loop
    for _n in range(3):
        for _i in range(10): # do some average
            temp = []
            for B in range(0,256,10):
#                Im = np.ones((par['xres'],par['yres'],3), dtype='uint8') * B
                Im = np.zeros((par['xres'],par['yres'],3), dtype='uint8')
                Im[:,:,_n] = np.ones((par['xres'],par['yres']), dtype='uint8') * B
                cv.imshow('light', Im)
                cv.waitKey(1)
                time.sleep(0.5)
                frame = cam.capture()
                frame = cam.capture()
                I = np.mean(crop(frame, ROI))
                temp.insert(-1,I)
                cv.imshow('capture', frame)
                cv.waitKey(1)
                time.sleep(0.2)
            curve[_i,:,_n] = np.array(temp)
    cam.close()
    
    now = datetime.datetime.now()
    np.save('gamma_{:.0f}.npy'.format(datetime.datetime.timestamp(now)), curve)
    cv.destroyAllWindows()

else:
    curve = np.load('gamma.npy')
    curve = np.mean(curve, axis=0)

x = np.array(range(0,256,10))
curve = np.array(curve) - curve[0]
curve[10:] = curve[10]
curve = curve / max(curve) * 255

xx = np.arange(0,256,1)
pp = np.polyfit(x, curve, deg=10)
qq = np.polyfit(curve, x, deg=10)

p = np.poly1d(pp)
q = np.poly1d(qq)

gamma = (p(xx) - p(xx)[0]) / np.max(p(xx) - p(xx)[0]) * 255
gamma[125:] = gamma[125]
f = interp1d(gamma, xx, kind='linear', fill_value='extrapolate')
gamma_c = f(xx)
gamma_c[-1] = 255

plt.plot(xx, gamma, '-b', label='gamma')
plt.plot(x, curve, '*c')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True, linestyle=':')
plt.title('Gamma - Aaxa')
plt.plot(xx, gamma_c, '-', color='orange')
plt.plot(curve, x, '*r')


out = q(xx) - q(xx)[0]
out = out / max(out) * 255

f = interp1d(out, xx, kind='linear')
out2 = f(xx)

## Save on file
if False:
    header = ['Calibration data','date and time: '+datetime.datetime.now().strftime("%d/%m/%y-%H:%M"),
              'exposure time: %.1fms'%8.3 + '[%.1f]'%8.3] # this way is easier to extract with regex
    np.savetxt('gammaCorrection_aaxa.csv', gamma_c.reshape(1,-1), header='\n'.join(header), delimiter=',', fmt='%f')
#    np.savetxt('gammaCorrection_aaxa2.csv', out.reshape(1,-1), header='\n'.join(header), delimiter=',', fmt='%f')
