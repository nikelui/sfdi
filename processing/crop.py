# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:35:37 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

def crop(Im, ROI):
    """Simple function to crop images with ROI acquired in OpenCV.
- Im is an image in the form of a numpy array.
- ROI is a tuple that defines a rectangle: (x,y,width,heigth),
  with x,y the coordinates of top-left corner
"""
    I_ROI = Im[ROI[1]:ROI[1]+ROI[3],ROI[0]:ROI[0]+ROI[2]] # works both for color and grayscale
    return I_ROI
