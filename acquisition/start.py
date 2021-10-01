# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:39:08 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

#### edit1: new naming scheme ####

name_xyz.bmp

x: wavelength (0 to n.Wavelengths)
y: spatial frequency (0 to nFreq)
z: phase (0 to nPhase)

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
# NOTE: the folder containing sfdi must be in your PYTHONPATH
import os
import cv2 as cv
from sfdi.acquisition.setWindow import setWindow
from sfdi.acquisition.expGUI_cvui import expGUI_cvui
from sfdi.camera.pointGrey import PointGrey as Camera
from sfdi.common.readParams import readParams
from numpy import genfromtxt

## Read parameters from .cfg file
par = readParams('./parameters.ini') # .cfg file should be in the same directory
## Load gamma correction array
if par['cpath']:
    par['gamma'] = genfromtxt(par['cpath'], delimiter=',')
else:
    par['gamma'] = None


## Check if out folder exist, if not, create it
if not os.path.exists(par['outpath']):
    os.makedirs(par['outpath'])

### Setting up camera ###
cam = Camera(num=0, res=par['res'], fps=par['res'])  # set-up camera
#TODO: automatically detect screen size
setWindow('pattern', size=(par['xres'],par['yres']),pos=(par['w'],0)) # Set-up window on second monitor
#TODO: new GUI, with extra functionality (remove opencv)
expGUI_cvui(cam, par, 'pattern', par['gamma'])  # Start GUI

# cam.close()
cv.destroyAllWindows()
