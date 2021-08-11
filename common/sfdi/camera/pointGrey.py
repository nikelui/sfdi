# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:50:54 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import numpy as np
from cv2 import imwrite
try:
    import PyCapture2 as pc
except ImportError:
    import sys
    sys.path.append('C:/PythonX/Lib/site-packages/')
    import PyCapture2 as pc


class PointGrey:
    #TODO: improve doc
    """Set up and configure a PointGrey camera using proprietary drivers

syntax: cam = PointGrey(*args, **kwargs)

@input
- num -> an integer containing the camera index. Used to select from camera list obtained
         from GetDevices()
- res -> a string specifying the resolution for acquisition. Accepted values are '1280p' (1280x960)
         and '640p' (640x480) [default].
- fps -> framerate (float). Accepted values are XX, XX and XX. Anything else will default to XXfps
         [NOTE]: use 0fps to disable the framerate, for exposure times longer than 133ms.
- binning -> (Bool) only works with 640p resolution. If True it will rescale the image, if False
             it will crop the image in the center.

@methods
- capture(nframes, save, filename)
    @input
    - nframes -> int, optional temporal averaging over n frames [default = 1]
    - save -> bool, if True the captured frame is also saved to "filename"
    - filename -> string, can contain an absolute or relative path.
    @return
    - frame -> captured frame as numpy array

NOTE: for now, most camera configurations are hard-coded in this module
"""
    def __delattr__(self):
        # 
        super().__delattr__()
        self.close()
    
    
    def __init__(self, num=0, res='1280p', fps=15):
        self.nchannels = 3  # Number of color channels captured
        ## Set the default algorithm for color processing
        pc.setDefaultColorProcessing(pc.COLOR_PROCESSING.HQ_LINEAR)
        bus = pc.BusManager() # constructor
        try:
            if num < 10:
                uid = bus.getCameraFromIndex(num) # this retrieves the address of the camera to connect
            else:
                uid = bus.getCameraFromSerialNumber(num)
        except pc.Fc2error as fc2Err:
                print('Error retrieving device address: %s' % fc2Err)
                raise SystemExit(-1)
        self.cam = pc.Camera() # Camera object
        try:
            self.cam.connect(uid) # connect the camera
        except pc.Fc2error as fc2Err:
                print('Error connecting camera: %s' % fc2Err)
                raise SystemExit(-1)
        ## Here set the acquisition properties (look at the manual for supported values)
        if res == '1280p':
            vMode = pc.VIDEO_MODE.VM_1280x960Y8 # 1280x960 pixels, RAW 8bit
        elif res == '640p':
            vMode = pc.VIDEO_MODE.VM_640x480Y8 # 640x480 pixels, RAW 8bit
        else:
            vMode = pc.VIDEO_MODE.VM_640x480Y8 # default to 640x480 pixels, RAW 8bit
    
        if fps == 60:
            fRate = pc.FRAMERATE.FR_60 # 60fps. This limits the exposure to 16ms
        elif fps == 30:
            fRate = pc.FRAMERATE.FR_30 # 30fps. This limits the exposure to 33ms
        elif fps == 15:
            fRate = pc.FRAMERATE.FR_15 # 15fps. This limits the exposure to 66ms
        else:
            fRate = pc.FRAMERATE.FR_7_5 # Default to 7.5fps. This limits the exposure to 133ms
        
        try:
            self.cam.setVideoModeAndFrameRate(vMode,fRate)
        except pc.Fc2error as fc2Err:
             print('Error setting video mode and framerate: %s' % fc2Err)
        ## Here set camera properties (like disabling auto-exposure)
        ## Exposure time should be controlled in a GUI
        ## setProperty does not work properly if you don't pass a Property object
        prop = pc.Property(pc.PROPERTY_TYPE.BRIGHTNESS,present=True,absControl=False,valueA=0)
        self.cam.setProperty(prop) # DONE
        prop = pc.Property(pc.PROPERTY_TYPE.AUTO_EXPOSURE,absControl=True,autoManualMode=False,onOff=False)
        self.cam.setProperty(prop) # DONE
        prop = pc.Property(pc.PROPERTY_TYPE.GAMMA,absControl=True,absValue=1.0,onOff=True)
        self.cam.setProperty(prop) # DONE
        prop = pc.Property(pc.PROPERTY_TYPE.SHUTTER,absControl=True,absValue=50.0,autoManualMode=False)
        self.cam.setProperty(prop) # DONE
        prop = pc.Property(pc.PROPERTY_TYPE.GAIN,absControl=True,absValue=0.0,autoManualMode=False)
        self.cam.setProperty(prop) # DONE
        prop = pc.Property(pc.PROPERTY_TYPE.FRAME_RATE,absControl=True,absValue=fps,autoManualMode=False,onOff=True)
        if fps == 0:
            prop = pc.Property(pc.PROPERTY_TYPE.FRAME_RATE,onOff=False) # Disable fps, to get longer exposure time
            self.cam.setProperty(prop) # DONE
        prop = pc.Property(pc.PROPERTY_TYPE.WHITE_BALANCE,onOff=False)
        self.cam.setProperty(prop) # DONE
        self.cam.setConfiguration(
            numBuffers=10,
            grabMode=pc.GRAB_MODE.BUFFER_FRAMES)
        print('Camera connected. Remember to call cam.disconnect() in your script.')
    
    
    def capture(self, nframes=1, save=False, filename='output.bmp'):
        ## Try to auto-detect width and heigth to initialize data
        try:
            (v,f) = self.cam.getVideoModeAndFrameRate()
        except pc.Fc2error as fc2Err:
            print('Error retrieving videoMode and frameRate: %s' % fc2Err)
        if v == pc.VIDEO_MODE.VM_1280x960Y8 or v == pc.VIDEO_MODE.VM_1280x960Y16:
            pic = np.zeros((960,1280,3),dtype='float')
        elif v == pc.VIDEO_MODE.VM_640x480Y8 or v == pc.VIDEO_MODE.VM_640x480Y16:
            pic = np.zeros((480,640,3),dtype='float')
        else:
            # Here acquire one image and use getRows(), getCols()
            try:
                self.cam.startCapture()
            except pc.Fc2error as fc2Err:
                print('Error starting acquisition: %s' % fc2Err)
            try:
                tmp = self.cam.retrieveBuffer()
            except pc.Fc2error as fc2Err:
                print('Error retrieving buffer: %s' % fc2Err)
            self.cam.stopCapture()
            pic = np.zeros((tmp.getRows(),tmp.getCols(),3),dtype='float')
       
        ## Begin acquisition
        try:
            self.cam.startCapture()
        except pc.Fc2error as fc2Err:
            print('Error starting acquisition: %s' % fc2Err)

        for _i in range(nframes):
            for _i in range(10):  # number of retries
                try:
                    im = self.cam.retrieveBuffer()
                    break  # do not retry if image was retrieved successfully
                except pc.Fc2error as fc2Err:
                    print('Error retrieving buffer: %s' % fc2Err)
            im = im.convert(pc.PIXEL_FORMAT.BGR)  # from RAW to color (BGR 8bit)
            data = im.getData()  # a 1D array of data (python list)
            frame = np.reshape(data,(im.getRows(), im.getCols(), 3))  # Reshape to 2D color image
            pic = pic + frame  # sum the captured images
        frame = pic / nframes  # average
        if save:  # might rewrite this using PyCapture save()
            imwrite(filename, frame.astype('uint8'))
        self.cam.stopCapture()
    
        return frame.astype('uint8')


    def getResolution(self):
        """Function that returns a tuple with the camera resolution (heigth, widht)"""
        try:
           (vid,_) = self.cam.getVideoModeAndFrameRate()
        except pc.Fc2error as fc2Err:
            print('Error retrieving videoMode and frameRate: %s' % fc2Err)
        # Matrix to store pictures  
        if vid == pc.VIDEO_MODE.VM_1280x960Y8 or vid == pc.VIDEO_MODE.VM_1280x960Y16:
            return (960, 1280)
        elif vid == pc.VIDEO_MODE.VM_640x480Y8 or vid == pc.VIDEO_MODE.VM_640x480Y16:
            return (480, 640)
        
        
    def getExposure(self):
        try:
            prop = self.cam.getProperty(pc.PROPERTY_TYPE.SHUTTER)
            expT = float(prop.absValue)
        except pc.Fc2error as fc2Err:
            print('Error getting exposure property: %s' % fc2Err)
        return expT  # should be in ms
    
    
    def getExposureLim(self):
        # first check if framerate is off
        isOn = self.cam.getProperty(pc.PROPERTY_TYPE.FRAME_RATE)
        if isOn.onOff == True:
            try:
                prop = self.cam.getPropertyInfo(pc.PROPERTY_TYPE.SHUTTER)
                expMin = prop.absMin
                expMax = prop.absMax
                expStep = (expMax - expMin) / 543  # DEBUG: check if actual value
            except pc.Fc2error as fc2Err:
                print('Error getting exposure info: %s' % fc2Err)
        else:
            expMin = 0
            expMax = 500
            expStep = 0.1
        
        return (expMin, expMax, expStep)
    
    
    def setExposure(self, expT):
        try:
            prop = pc.Property(pc.PROPERTY_TYPE.SHUTTER, absControl=True,
                               absValue = expT, autoManualMode=False)
            self.cam.setProperty(prop) # DONE
        except pc.Fc2error as fc2Err:
            print('Error setting exposure time: %s' % fc2Err)
    
    
    def setFramerate(self, fps):
        try:
            vMode, _ = self.cam.getVideoModeAndFrameRate()
        except pc.Fc2error as fc2Err:
            print('Error getting video mode and framerate: %s' % fc2Err)
        if fps == 60:
            fRate = pc.FRAMERATE.FR_60 # 60fps. This limits the exposure to 16ms
        elif fps == 30:
            fRate = pc.FRAMERATE.FR_30 # 30fps. This limits the exposure to 33ms
        elif fps == 15:
            fRate = pc.FRAMERATE.FR_15 # 15fps. This limits the exposure to 66ms
        else:
            fRate = pc.FRAMERATE.FR_7_5 # Default to 7.5fps. This limits the exposure to 133ms
        
        if fps == 0:
            self.cam.setVideoModeAndFrameRate(vMode, fRate)
            prop = pc.Property(pc.PROPERTY_TYPE.FRAME_RATE,onOff=False) # Disable fps
            self.cam.setProperty(prop)
        else:
            self.cam.setVideoModeAndFrameRate(vMode, fRate)
            prop = pc.Property(pc.PROPERTY_TYPE.FRAME_RATE,absControl=True,
                               absValue=fps,autoManualMode=False,onOff=True)
            self.cam.setProperty(prop)


#    def stopCapture(self):
#        self.cam.stopCapture()
        
        
    def close(self):
        """Shut down camera"""
        # TODO: if self: run = True, cam.stopCapture()
        self.cam.disconnect()
        print('Camera successfully disconnected.')