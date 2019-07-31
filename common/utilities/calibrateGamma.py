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

Im = np.zeros((par['yRes'],par['xRes']),dtype='uint8') # initialize reference image
Im[:,:] = 255 # white image
exposures = np.arange(0,256,par['lstep']) # Light intensity levels
#if not(255 in exposures):
#    exposures = np.append(exposures,255) # make sure that the end points are present

cv.imshow('light',Im) # Project figure
cv.waitKey(1)
time.sleep(0.2) # a small pause
frame = sfdi.camCapt_pg(cam,nframes=1,save=False) # capture one frame

ROI = cv.selectROI('ROI',frame) # select the ROI where to process the data
cv.destroyWindow('ROI')


# TODO: need a method to select multiple exposure times
raw_out = []

# loop over intensity levels
first = True
for level in exposures:
    Im[:,:] = level # change brightness
    cv.imshow('light',Im) # show on screen
    cv.waitKey(1)
    if first:
        time.sleep(1) # a short pause
        first=False
    frame = sfdi.camCapt_pg(cam,20,False) # capture image, average over 10
    #time.sleep(0.3) # a small pause
    print('Acquiring level: ',level)
    
    raw_out.append(np.mean(sfdi.crop(frame,ROI),axis=(0,1))) # calculate average over ROI and append

cam.disconnect()
cv.destroyAllWindows()
## Saving raw data
header = ['Raw calibration data','date and time: '+datetime.datetime.now().strftime("%d/%m/%y-%H:%M"),
          'exposure time: %.1fms'%expt + '[%.1f]'%expt, # this way is easier to extract with regex
          'row1: exposure levels, row2-4: output levels for blue, green and red']
data = np.concatenate((exposures.reshape(1,-1),np.array(raw_out).T),axis=0)

## Just some plots for debugging
plt.figure(1)
plt.plot(data[0,:],data[1,:],'-b',label='blue')
plt.plot(data[0,:],data[2,:],'-g',label='green')
plt.plot(data[0,:],data[3,:],'-r',label='red')
plt.grid(True,which='both',linestyle=':')

### compare the three channels
#ratiobg = data[1,-1]/data[2,-1]
#ratiorg = data[3,-1]/data[2,-1]
#
#plt.figure(2)
#plt.plot(data[0,:],data[1,:]/ratiobg,'-b',label='blue')
#plt.plot(data[0,:],data[2,:],'-g',label='green')
#plt.plot(data[0,:],data[3,:]/ratiorg,'-r',label='red')
#plt.grid(True,which='both',linestyle=':')

#csvwrite('smart_beam_new_raw.csv',data,header)

###### Second part, fit the data to get inverse function #######
# The three channel have the same curve (except for the different gain), so we take an average
#direct_curve = np.mean(data[1:,:],axis=0)
direct_curve = data[3,:]
direct_curve -= direct_curve[0] # remove offset
direct_curve *= (255 / max(direct_curve)) # normalize between 0 and 255

x = np.arange(0,256) # In coordinates
z = np.polyfit(data[0,:],direct_curve,deg=10) # polynomial fit of order N (must be high enough)
f = np.poly1d(z)
Input = f(x) # direct curve
## Debug
plt.figure()
plt.plot(data[0,:],direct_curve,'*r')
plt.plot(x,Input,'k')
plt.plot([0,255],[0,255],':k')

ff = interp1d(Input,x,kind='linear',assume_sorted=True,fill_value='extrapolate') # swap x and y
Output = ff(x) # inverse curve
Output -= Output[0] # remove offset
Output *= (255 / max(Output))

## Debug
plt.figure()
plt.plot(direct_curve,data[0,:],'*r')
plt.plot([0,255],[0,255],':k')
plt.plot(x,Output,'k')

csvwrite('gammaCorrection_smartBeam_new.csv',Output.reshape(1,-1),header=[])