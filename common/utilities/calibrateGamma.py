# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:25:41 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Script to calibrate for the gamma correction of projector, using a PointGrey camera.
You should point the projector at a reflection target, or similar reflecting surface
"""
import datetime,time,sys
from matplotlib import pyplot as plt
#from scipy.interpolate import interp1d
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

## First part: acquire data
cam = sfdi.setCamera_pg(num=0,res=par['res'],fps=par['fps']) # set-up camera
win = sfdi.setWindow('light',size=(par['xRes'],par['yRes']),pos=(par['W'],0)) # create window

# Set the exposure time
try:
    prop = pc.Property(pc.PROPERTY_TYPE.SHUTTER,absControl=True,absValue=expt)
    cam.setProperty(prop) # DONE
except pc.Fc2error as fc2Err:
    print('Error setting exposure time: %s' % fc2Err)

Im = np.zeros((par['yRes'],par['yRes']),dtype='uint8') # initialize reference image
Im[:,:] = 255 # white image
exposures = np.arange(0,256,par['lstep']) # Light intensity levels

cv.imshow('light',Im) # Project figure
cv.waitKey(1)
time.sleep(0.2) # a small pause
frame = sfdi.camCapt_pg(cam,nframes=3,save=False) # capture one frame

ROI = cv.selectROI('ROI',frame) # select the ROI where to process the data
cv.destroyWindow('ROI')


# TODO: need a method to select multiple exposure times
raw_out = []

# loop over intensity levels
for level in exposures:
    Im[:,:] = level # change brightness
    cv.imshow('light',Im) # show on screen
    cv.waitKey(1)
    time.sleep(0.3) # a small pause
    frame = sfdi.camCapt_pg(cam,5,False) # capture image
    time.sleep(0.3) # a small pause
    
    raw_out.append(np.mean(sfdi.crop(frame,ROI),axis=(0,1))) # calculate average over ROI and append

cam.disconnect()
## Saving raw data
header = ['Raw calibration data','date and time: '+datetime.datetime.now().strftime("%y/%m/%d-%H:%M"),
          'exposure time: %.1fms'%expt + '[%1f]'%expt, # this way is easier to extract with regex
          'row1: exposure levels, row2-4: output levels for blue, green and red']
data = np.concatenate((exposures.reshape(1,-1),np.array(raw_out).T),axis=0)
plt.plot(data[0,:],data[1:,:].T) # debug
csvwrite('smart_beam_new_raw.csv',data,header)
