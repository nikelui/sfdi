# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:39:08 2019

@author: Luigi Belcastro

#### edit1: new naming scheme ####

name_xyz.bmp

x: wavelength(0=B, 1=GB, 2=G, 3=GR, 4=R)
y: spatial frequency (0 to nFreq)
z: phase (0,1,2)

#### edit2: reorder spatial frequencies in name ####

The original name convention had them ordered by spatial period (0, 10mm, 20mm, 30mm...)
The new naming convention has that reversed (except from 0), so they are ordered
by spatial frequency (0, 1/30, 1/20, 1/10...)

#### edit3: reorganized the acquisition procedure ####

Now captured images are kept in memory and saved at the end, in order to
speed up acquisition time

#### edit4: changed output path structure ####

Each full acquisition dataset is saved inside a folder named with a timestamp,
this way multiple time-points can be processed easily.

"""

import sys, os
import cv2 as cv

sys.path.append('../common') # Add the common folder to path
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work
import sfdi
from sfdi.readParams2 import readParams
from mycsv import csvread

## Read parameters from .cfg file
par = readParams('./parameters.cfg') # .cfg file should be in the same directory
## Load gamma correction array
correction,_ = csvread(par['cPath'],True)

## Check if out folder exist, if not, create it
if not os.path.exists(par['outPath']):
    os.makedirs(par['outPath'])

### Setting up camera ###
cam = sfdi.setCamera_pg(num=0,res=par['res'],fps=par['fps']) # Set-up Camera
#TODO: automatically detect screen size
sfdi.setWindow('pattern',size=(par['xRes'],par['yRes']),pos=(par['W'],0)) # Set-up window on second monitor
#TODO: new GUI, with extra functionality
sfdi.expGUI_cvui(cam,par,'pattern',correction) # Start GUI

cam.disconnect()
cv.destroyAllWindows()
