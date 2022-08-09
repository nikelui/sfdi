# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 10:52:24 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import numpy as np
from ximea import xiapi
import PIL.Image  # trying not to use OpenCV
from sfdi.processing.crop import crop


class XiCam:
    """Set-up and configure a x
    imea camera using proprietary xiCam drivers."""
    # TODO: documentation
    
    def __delattr__(self):
        # 
        super().__delattr__()
        self.cam.close()
    
    def __init__(self, fps=60.0, **kwargs):
        # Number of the camera color channels (1 for monochrome, 3 for RGB and N for multispectral)
        self.nchannels = 9
        self.cam = xiapi.Camera()
        self.cam.open_device()
        self.cam.set_imgdataformat('XI_MONO8')
        self.cam.disable_aeag()  # automatic exposure / gain
        self.cam.disable_auto_wb()  # white balance
        self.cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
        self.cam.set_framerate(fps)
        self.cam.set_exposure(100.0e3)  # in us

        # Camera horizontal and vertical resolution (pixel)
        # This is actually the ROI resolution, after cropping the image
        self.xRes = 600
        self.yRes = 600
        # here place the 9 cam ROIs after calibrarion/realignment
        self.rois = [(1,42,self.xRes,self.yRes),    # cam1
                     (702,31,self.xRes,self.yRes),   # cam2
                     (1389,26,self.xRes,self.yRes),  # cam3 
                     (20,732,self.xRes,self.yRes),   # cam4
                     (712,722,self.xRes,self.yRes),  # cam5
                     (1400,712,self.xRes,self.yRes), # cam6
                     (28,1428,self.xRes,self.yRes),  # cam7
                     (722,1417,self.xRes,self.yRes), # cam8
                     (1415,1408,self.xRes,self.yRes) # cam9      
                     ]
        
    def capture(self, **kwargs):
        """This function should acquire one frame from the camera and return it as
        a numpy array (uint8: values from 0-255). Multiple color channels are stacked
        into a single B/W image (for multi-spectral)"""
        img = xiapi.Image()
        self.cam.start_acquisition()
        self.cam.get_image(img)
        self.cam.get_image(img)  # do it twice, to avoid synch problems
        self.cam.stop_acquisition()
        data = img.get_image_data_numpy()
        frame = np.array([crop(data, ROI) for ROI in self.rois]).transpose((1,2,0))
        return frame.astype('uint8')
    
    def preview(self, **kwargs):
        """This function should acquire one frame from the camera and return it as
        a numpy array (uint8: values from 0-255)."""
        img = xiapi.Image()
        self.cam.start_acquisition()
        self.cam.get_image(img)
        self.cam.stop_acquisition()
        data = img.get_image_data_numpy()
        return data.astype('uint8')
    
    def getResolution(self):
        """Return the horizontal and vertical resolution of the camera (int: pixel)"""
        return (self.yRes, self.xRes)
    
    def getExposure(self):
        """Return the current set exposure time (float: milliseconds).
        NOTE: if the camera works with a different time unit, remember to convert."""
        return self.cam.get_exposure() / 1000  # in ms
    
    def getExposureLim(self):
        """Return the min and max values for exposure and the increase/decrease
        step (float: milliseconds)
        NOTE: if the camera works with a different time unit, remember to convert."""
        expMin = self.cam.get_exposure_minimum() / 1000
        expMax = self.cam.get_exposure_maximum() / 1000
        expStep = self.cam.get_exposure_increment() / 1000
        return (expMin, expMax, expStep)
    
    def setExposure(self, exp):
        """Set exposure time to exp (float: milliseconds)
        NOTE: if the camera works with a different time unit, remember to convert."""
        self.cam.set_exposure(exp * 1000)
    
    def setFramerate(self, fps):
        """Set the camera framerate (float: fps)"""
        self.cam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
        self.cam.set_framerate(fps)
    
    def close(self):
        """Stop acquisition, disconnect the camera and clear the object from memory."""
        self.cam.close_device()
    
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    asd = XiCam()
    asd.setExposure(100)
    frame = asd.capture()
    asd.close()
#%%
    for _i in range(9):
        plt.figure()
        plt.imshow(frame[:,:,_i])
    # img = PIL.Image.fromarray(frame, 'L')
    # # img.show()
    # plt.figure(666)
    # plt.imshow(frame[:,:,4])