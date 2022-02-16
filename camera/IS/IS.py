# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:16:21 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import os, sys
#os.environ['PATH'] = os.path.dirname(__file__) + ';' + os.environ['PATH']
basepath = os.path.dirname(__file__)
sys.path.insert(0, basepath)
import tisgrabber as IC
import numpy as np

class ImagingSource:
    """Set up and configure an ImagingSource camera using proprietary drivers

syntax: cam = ImagingSource()

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
        super().__delattr__()
        self.cam.StopLive()
        
    def __init__(self, num=0, res='640p', fps=60.0, binning=False, Live=False, **kwargs):
        self.cam = IC.TIS_CAM()
#        import pdb; pdb.set_trace()  # DEBUG
        dev = self.cam.GetDevices()
        try:
            if not self.cam.open(dev[num]):
                self.cam.ShowDeviceSelectionDialog()  # Use device selection dialog
        except IndexError:
            self.cam.ShowDeviceSelectionDialog()  # Use device selection dialog
        if res == '1280p':
            vfmt = 'RGB32 (1280x960)'
        else:
            vfmt = 'RGB32 (640x480)'
        if binning:
            vfmt = vfmt + ' [Binning 2x]'
        self.cam.SetVideoFormat(vfmt)
        self.cam.SetFrameRate(fps)
        # Save some properties
        self.expmin, self.expmax, _ = self.getExposureLim()
        self.nchannels = 3  # Number of color channels captured
        # Disable auto-settings
        self.cam.SetPropertySwitch("Exposure","Auto",0)  # Auto-exposure
        self.cam.SetPropertySwitch("WhiteBalance","Auto",0)  # Auto-WhiteBalance
        self.cam.SetPropertySwitch("Gain","Auto",0)  # Auto-gain
        self.cam.SetPropertySwitch("Tone Mapping","Auto",0)  # Auto-color correction?
        # Some default values
        self.cam.SetPropertyAbsoluteValue("Exposure", "Value", 1e-2)  # 10ms
        self.cam.SetPropertyAbsoluteValue("Gain", "Value", 0.0)  # No gain
        self.cam.SetPropertyValue("Brightness", "Value", 0)
        self.cam.SetPropertyValue("Hue", "Value", 0)
        self.cam.SetPropertyValue("Contrast", "Value", 0)
        self.cam.SetPropertyValue("Saturation", "Value", 64)
        self.cam.SetPropertyValue("Sharpness", "Value", 0)
        self.cam.SetPropertyValue("Gamma", "Value", 100)
        self.cam.SetPropertyValue("Denoise", "Value", 0)
        # Start live stream. Better to put it here (less time for acquisition)
        if Live:
            self.cam.StartLive(1)  # 1: Show live screen
        else:
            self.cam.StartLive(0)  # 0: Don't show live screen
    
    def capture(self):
        """Capture one frame and return it as a 3D numpy array (heigth x width x color_channels)"""
        self.cam.SnapImage()
        frame = self.cam.GetImage()
        return np.flip(frame, axis=1)  # Horizontal flip
    
    def getResolution(self):
        """Function that returns a tuple with the camera resolution (heigth, width)"""
        x = self.cam.GetImageDescription()
        return x[1], x[0]
    
    def getExposure(self):
        exp = [0]
        self.cam.GetPropertyAbsoluteValue("Exposure", "Value", exp)
        return exp[0]*1e3  # convert in ms
    
    def getExposureLim(self):
        pmin = [0]
        pmax = [0]
        self.cam.GetPropertyAbsoluteValueRange("Exposure", "Value", pmin, pmax)
        return (pmin[0]*1e3, pmax[0]*1e3, (pmax[0]-pmin[0])/10000*1e3 )  # convert in ms
    
    def setExposure(self, exp):
        # NOTE: expmin/expmax are in [ms], exp comes from the slider so it is also in [ms]
        if exp < self.expmin:
            exp = self.expmin
        if exp > self.expmax:
            exp = self.expmax
        exp /= 1e3  # convert from ms to seconds for the camera
        self.cam.SetPropertyAbsoluteValue("Exposure", "Value", exp)
        
    def setFramerate(self, fps):
        self.cam.SetFrameRate(fps)
    
    def close(self):
        """Shut down camera"""
        self.cam.StopLive()


if __name__ == '__main__':
    a = ImagingSource(res='640p', binning=True, Live=True)
    a.setExposure(10)  # should be in ms
    
    x = a.getResolution()
    y = a.getExposure()
    z = a.getExposureLim()
    a.close()
        