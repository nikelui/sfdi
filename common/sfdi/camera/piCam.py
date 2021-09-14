# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 15:25:07 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import time
import numpy as np
from picamera import PiCamera  # only works in raspbian system
from PIL import Image

class PiCam:
    #TODO: improve doc
    """Set up and configure a PointGrey camera using proprietary drivers

syntax: cam = Picamera(*args, **kwargs)

@input
- num -> an integer containing the camera index. Used to select from camera list obtained
         from GetDevices()
- res -> a string specifying the resolution for acquisition. Accepted values are:
        - '3280p' (3280x2464)
        - '1640p' (1640x922)
        - '1280p' (1280x720) [default]
        - '640p' (640x480)
- fps -> framerate (float). [default=30fps]

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
        self.cam.close()
    
    
    def __init__(self, num=0, res='1280p', fps=30):
        self.nchannels = 3  # Number of color channels captured
        if res == '3280p':
            resolution = (3280,2464)
        elif res == '1640p':
            resolution = (1640,922)
        elif res == '640p':
            resolution = (640,480)
        else:  # default to 1280p
            resolution = (1280,720)
        self.cam = PiCamera(camera_num=num, resolution=resolution, framerate=fps)
        # PiCam.setlocation_preview(self)
        # Set fixed camera properties
        self.cam.ISO = 100
        self.cam.awb_mode = 'off'  # white balance off
        self.cam.awb_gains = (1.2,1.5)  # TODO: "calibrate" camera and find appropriate values
        self.cam.brightness = 50
        time.sleep(1)
        self.cam.exposure_mode = 'off'  # auto exposure off
        self.cam.shutter_speed = 33 *1000  # exposure time (in microseconds, remember to convert)      
        time.sleep(1)
        # PiCam.Setting_info(self)
        # PiCam.Histogram(self)
        
    
    def capture(self, nframes=1, save=False, filename='output.bmp'):
        ## Try to auto-detect width and heigth to initialize data
        width, height = self.cam.resolution
        pic = np.zeros((height, width, 3), dtype='float')  # need float to average
        ## Begin acquisition
        for _i in range(nframes):
            frame = np.zeros((height, width, 3), dtype='uint8')  # initialize
            self.cam.capture(frame, format='rgb')
            pic = pic + frame  # sum the captured images
        frame = pic / nframes  # average
        if save:  # might rewrite this using PIL
            Im = fromarray(frame.astype('uint8'))
            Im.save(filename)
        return frame.astype('uint8')


    def getResolution(self):
        """Function that returns a tuple with the camera resolution (height, width)"""
        width, height = self.cam.resolution
        return (height, width)        
        

    def getExposure(self):
        return self.cam.exposure_speed / 1000.  # in ms
    
    
    def getExposureLim(self):
        fps = self.cam.framerate
        if fps > 0:
            expMax = 1. / fps  * 1000  # in ms
        else:
            expMax = 500  # the actual limit is 6s, but will rarely need more than 200ms
        expMin = 0.2  # ms
        expStep = 0.2  # ms   
        return (expMin, expMax, expStep)
    
    
    def setExposure(self, expT):
        self.cam.shutter_speed = int(expT * 1000)  # convert to us
    
    
    def setFramerate(self, fps):
        #TODO: try / except
        self.cam.framerate = int(fps)
    
    def stoppreview(self):
        self.cam.stop_preview()
        
    def setstartpreview(self):
        self.cam.start_preview(fullscreen=False,window=(0,480,427,240))
    # def setlocation_preview(self):
    #     self.cam.stop_preview()
    #     self.cam.start_preview(fullscreen=False,window=(0,480,427,240)) 
    
    def Setting_info(self):
        print("SS: ",self.cam.shutter_speed)
        print("ISO: ",self.cam.iso)
        print("framerate: ",self.cam.framerate)
        print("resolution: ",self.cam.resolution)
        print("brightness: ",self.cam.brightness)
        print("contrast: ",self.cam.contrast)
        print("saturation: ",self.cam.saturation)
        print("exposure_speed: ",self.cam.exposure_speed)
        print("sharpness: ",self.cam.sharpness)
        print('awb gains: ',self.cam.awb_gains)
    
    # def Histogram(self):
    #     # Create the in-memory stream
    #     stream = io.BytesIO()   
    #     # Generates a 3D RGB array and stores it in rawCapture
    #     #raw_capture = PiRGBArray(self.cam, size=(1280, 720))
    #     self.cam.capture(stream, format="jpeg")
    #     # "Rewind" the stream to the beginning so we can read its content
    #     stream.seek(0)
    #     image = Image.open(stream)
    #     #frame = raw_capture.array
    #     R, G, B = image.split()
    #     print(R.histogram())
    #     R.histogram ()
    #     G.histogram () 
    #     B.histogram ()
    
    def close(self):
        """Shut down camera"""
        # TODO: if self: run = True, cam.stopCapture()
        self.cam.close()
        print('Camera successfully disconnected.')
