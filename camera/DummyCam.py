# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 10:52:24 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import numpy as np


class DummyCam:
    """A dummy camera object for debug purposes. It implements all the camera methods
    with placeholders. Make a copy of this class for each camera and implement the
    actual class methods.
    
    NOTE: in the __init__ section, you should set all the relevant camera initialization
    (e.g. disable auto-exposure, gain and white balance, set the framerate/resolution...)
    """
    def __init__(self, num=0, res='640p', fps=60.0, **kwargs):
        # Number of the camera color channels (1 for monochrome, 3 for RGB and N for multispectral)
        self.nchannels = 3
        # Camera horizontal and vertical resolution (pixel)
        self.xRes = 640
        self.yRes = 480
        
    def capture(self):
        """This function should acquire one frame from the camera and return it as
        a numpy array (uint8: values from 0-255)"""
        # The dummy class returns a linear gradient, just to be more interesting
        A = np.linspace(0, 255, self.xRes)
        B = np.tile(A, [self.yRes, 1])
        C = np.stack((B, B, B), axis=2)
        return C.astype('uint8')
    
    def getResolution(self):
        """Return the horizontal and vertical resolution of the camera (int: pixel)"""
        return (self.yRes, self.xRes)
    
    def getExposure(self):
        """Return the current set exposure time (float: milliseconds).
        NOTE: if the camera works with a different time unit, remember to convert."""
        return 1e2
    
    def getExposureLim(self):
        """Return the min and max values for exposure and the increase/decrease
        step (float: milliseconds)
        NOTE: if the camera works with a different time unit, remember to convert."""
        return (1, 100, 0.1)
    
    def setExposure(self, exp):
        """Set exposure time to exp (float: milliseconds)
        NOTE: if the camera works with a different time unit, remember to convert."""
        pass
    
    def setFramerate(self, fps):
        """Set the camera framerate (float: fps)"""
        pass
    
    def close(self):
        """Stop acquisition, disconnect the camera and clear the object from memory."""
        pass
    
if __name__ == '__main__':
    asd = DummyCam()
    frame = asd.capture()