# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 10:52:24 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import numpy as np


class DummyCam:
    """A dummy camera object for debug purposes. It implements all the camera methods with placeholders
    """
    def __init__(self, num=0, res='640p', fps=60.0, **kwargs):
        self.nchannels = 3
        self.xRes = 640
        self.yRes = 480
        
    def capture(self):
        # linear gradient, just to be more interesting
        A = np.linspace(0, 255, self.xRes)
        B = np.tile(A, [self.yRes, 1])
        C = np.stack((B, B, B), axis=2)
        return C.astype('uint8')
    
    def getResolution(self):
        return (self.yRes, self.xRes)
    
    def getExposure(self):
        return 1e2
    
    def getExposureLim(self):
        return (1, 100, 0.1)
    
    def setExposure(self, exp):
        pass
    
    def setFramerate(self, fps):
        pass
    
    def close(self):
        pass
    
if __name__ == '__main__':
    asd = DummyCam()
    frame = asd.capture()