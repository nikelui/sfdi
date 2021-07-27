# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:39:08 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

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

import os, sys
import cv2 as cv

sys.path.append('../common') # Add the common folder to path
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work
from sfdi.setWindow import setWindow
from sfdi.expGUI_cvui import expGUI_cvui
#from sfdi.expGUI_cvui_IS import expGUI_cvui
#from sfdi.IS.IS import ImagingSource
from sfdi.camera.pointGrey import PointGrey
from sfdi.setCamera_pg import setCamera_pg
from sfdi.readParams3 import readParams
from mycsv import csvread
#import numpy as np

## Read parameters from .cfg file
par = readParams('./parameters.ini') # .cfg file should be in the same directory
## Load gamma correction array
correction,_ = csvread(par['cpath'],True)

## Check if out folder exist, if not, create it
if not os.path.exists(par['outpath']):
    os.makedirs(par['outpath'])

### Setting up camera ###
#cam = setCamera_pg(num=0,res=par['res'],fps=par['fps']) # Set-up Camera
cam = PointGrey(num=0, res=par['res'], fps=par['res'])  # set-up camera
#cam = ImagingSource(num=0, res=par['res'], fps=par['fps'])
#TODO: automatically detect screen size
setWindow('pattern',size=(par['xres'],par['yres']),pos=(par['w'],0)) # Set-up window on second monitor
#TODO: new GUI, with extra functionality
expGUI_cvui(cam,par,'pattern',correction) # Start GUI

cam.close()
#cam.disconnect()
cv.destroyAllWindows()
