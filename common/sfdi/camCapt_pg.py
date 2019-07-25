# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:33:48 2019

@author: luibe59
"""
import numpy as np
import sys
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work
import PyCapture2 as pc
import cv2 as cv

def camCapt_pg(cam,nframes=1,save=False,filename='output.bmp'):
    """Camera capture with optional averaging over nframes [Point Grey version].

syntax: frame = camCapt_pg(cam,[nframes,save,filename])

- cap -> a PyCapture2 Camera object, properly connected to phyical camera
- nframes -> int, optional temporal averaging over n frames
- save -> bool, if True the captured frame is also saved on file "filename"
- filename -> string, can contain an absolute or relative path
"""
    # Initialize an internal variable to keep track of averaging
    try:
        (v,f) = cam.getVideoModeAndFrameRate()
    except pc.Fc2error as fc2Err:
        print('Error retrieving videoMode and frameRate: %s' % fc2Err)
        
    if v == pc.VIDEO_MODE.VM_1280x960Y8 or v == pc.VIDEO_MODE.VM_1280x960Y16:
        pic = np.zeros((960,1280,3),dtype='float')
    elif v == pc.VIDEO_MODE.VM_640x480Y8 or v == pc.VIDEO_MODE.VM_640x480Y16:
        pic = np.zeros((480,640,3),dtype='float')
    else:
        # Here acquire one image and use getRows(), getCols()
        try:
            cam.startCapture()
        except pc.Fc2error as fc2Err:
            print('Error starting acquisition: %s' % fc2Err)
        try:
            tmp = cam.retrieveBuffer()
        except pc.Fc2error as fc2Err:
            print('Error retrieving buffer: %s' % fc2Err)
        cam.stopCapture()
        pic = np.zeros((tmp.getRows(),tmp.getCols(),3),dtype='float')

    ## Begin acquisition
    try:
        cam.startCapture()
    except pc.Fc2error as fc2Err:
        print('Error starting acquisition: %s' % fc2Err)

    ## Add an extra acquisition for synchro?
    #im = cam.retrieveBuffer()
    for i in range(nframes):
        for i in range(10): # number of retries
            try:
                im = cam.retrieveBuffer()
                break # break if image was retrieved successfully
            except pc.Fc2error as fc2Err:
                print('Error retrieving buffer: %s' % fc2Err)
        im = im.convert(pc.PIXEL_FORMAT.BGR) # from RAW to color (BGR 8bit)
        data = im.getData() # a long array of data (python list)
        frame = np.reshape(data,(im.getRows(),im.getCols(),3)) # Reshape to 2D color image
        pic = pic + frame # sum the captured images
    frame = pic/nframes # average
    if save: # might rewrite this using PyCapture save()
        cv.imwrite(filename,frame.astype('uint8'))

    cam.stopCapture()
    return frame.astype('uint8')