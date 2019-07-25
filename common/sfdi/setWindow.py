# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:29:29 2019

@author: luibe59
"""
import cv2 as cv

def setWindow(name,size=(640,480),pos=(1920,0)):
    """Set Window object with some default values that can be changed.
Default create a window on second monitor (projector).
NEW: using WND_PROP_FULLSCREEN removes menu bar and taskbar."""
    W = pos[0]
    H = pos[1]
    width = size[0]
    height = size[1]
    
    cv.namedWindow(name,cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty(name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.resizeWindow(name,width,height)
    cv.moveWindow(name,W,H)
    
if (__name__ == '__main__'):
    setWindow('hello',(640,480),(300,100))
    cv.waitKey(0)
    cv.destroyAllWindows()