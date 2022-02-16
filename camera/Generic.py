# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:17:45 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import cv2 as cv


class cameraError(Exception):
    def __init__(self, message):
        self.message = message

class Generic:
    """A wrapper class for cameras. Subclass it for each new camera implementation.

As default, it tries to initialize a generic camera using OpenCV.
"""
    def __init__(self, num=0, width=640, heigth=480, **kwargs):
        self.cap = cv.VideoCapture(num)  # base opencv VideoCapture object
        #  some default values
        self.cap.set(cv.CAP_PROP_EXPOSURE,-5) # an arbitrary value of exposure
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, heigth)
        self.cap.set(cv.CAP_PROP_BRIGHTNESS, 0)
        self.cap.set(cv.CAP_PROP_GAIN, 0)
#       self.cap.set(cv.CAP_PROP_AUTO_WB, 0)
#       self.cap.set(cv.CAP_PROP_)

        for key, value in kwargs.items():  # set/override camera properties
            self.cap.set(key, value)
    
    def capture(self, exposure):
        """Capture one frame and return it as a 3D numpy array (heigth x width x color_channels)"""
        ret, frame = self.cap.read()
        ret, frame = self.cap.read()  # works better if acquires twice
        if ret == False:
            raise cameraError("Failed to retrieve buffer")
        else:
            
            return frame
    
    def getResolution(self):
        """Function that returns a tuple with the camera resolution (heigth, widht)"""
        pass
    
    def close(self):
        """Shut down camera"""
        pass
        