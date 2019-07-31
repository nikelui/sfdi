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
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work
import PyCapture2 as pc
from mycsv import csvwrite
from sfdi.readParams2 import readParams
import sfdi

par = readParams('parameters.cfg') # Read calibration parameters

expt = float(input('Insert exposure time (ms): '))

######## First part: acquire data ########
cam = sfdi.setCamera_pg(num=0,res=par['res'],fps=par['fps']) # set-up camera
win = sfdi.setWindow('light',size=(par['xRes'],par['yRes']),pos=(par['W'],0)) # create window

# Set the exposure time
try:
    prop = pc.Property(pc.PROPERTY_TYPE.SHUTTER,absControl=True,absValue=expt)
    cam.setProperty(prop) # DONE
except pc.Fc2error as fc2Err:
    print('Error setting exposure time: %s' % fc2Err)
    
## create gradient image

Im = np.zeros((par['yRes'],par['xRes'],3),dtype='uint8') # initialize reference image
bins = 1#par['xRes'] // 256 # number of pixels per level

pad = int((par['xRes'] - 256*bins) / 2) # padding

# create 1D gradient
x = np.zeros((par['xRes']),dtype='uint8')
x[:pad] = 0
x[pad:256*bins+pad] = np.repeat(np.arange(0,256),bins)
x[256*bins+pad:] = 255

Im[:,:,0] = x # 2D gradient
Im[:,:,1] = x # 2D gradient
Im[:,:,2] = x # 2D gradient

Im[:,:pad,:] = (0,255,0)
Im[:,256*bins+pad:,:] = (0,255,0)

cv.imshow('light',Im)
cv.waitKey(1)
time.sleep(5)

frame = sfdi.camCapt_pg(cam,1,False)
cam.disconnect()

ROI = cv.selectROI('ROI',frame)
cv.destroyAllWindows()

profile = np.mean(sfdi.crop(frame,ROI),axis=(0,2))
plt.plot(profile)
plt.grid(True)

#### Second part: interpolate ####
xax = np.linspace(0,255,len(profile)) # rescale horizontal axis

profile -= min(profile) # remove offset
profile *= (255 / max(profile)) # normalize between 0-255

x = np.arange(0,256) # In coordinates
z = np.polyfit(xax,profile,deg=7) # polynomial fit of order N (must be high enough)
f = np.poly1d(z)
Input = f(x) # direct curve
plt.figure()
plt.plot(x,Input)
plt.plot([0,255],[0,255],':k')
plt.grid(True)

ff = interp1d(Input,x,kind='linear',assume_sorted=True,fill_value='extrapolate') # swap x and y
Output = ff(x) # inverse curve
Output -= Output[0] # remove offset
Output *= (255 / max(Output))

## Debug
plt.figure()
plt.plot([0,255],[0,255],':k')
plt.plot(x,Output,'k')

header = ['Calibration data','date and time: '+datetime.datetime.now().strftime("%d/%m/%y-%H:%M"),
          'exposure time: %.1fms'%expt + '[%.1f]'%expt] # this way is easier to extract with regex

csvwrite('gammaCorrection_smartBeam_space.csv',Output.reshape(1,-1),header)