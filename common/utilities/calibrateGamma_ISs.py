# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:58:31 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
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


xpos = 1200
xlen = 512
ypos = 680

grad = np.zeros((1280,1920,3), dtype='uint8')
grad[:,0:xpos,:] = 255

grad[ypos:ypos+100, xpos:xpos+xlen, 0] = np.round(np.linspace(0,255,512))
grad[ypos+100:ypos+200, xpos:xpos+xlen, 1] = np.round(np.linspace(0,255,512))
grad[ypos+200:ypos+300, xpos:xpos+xlen, 2] = np.round(np.linspace(0,255,512))

cv.namedWindow('a',cv.WINDOW_NORMAL)
cv.moveWindow('a', 1920, 0)
cv.imshow('a', grad)