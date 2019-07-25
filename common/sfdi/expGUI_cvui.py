# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:51:14 2019

@author: luibe59
"""
import numpy as np
import cv2 as cv
import cvui
import sys
#sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work
import PyCapture2 as pc
from sfdi.acquisitionRoutine2 import acquisitionRoutine

class expGUI_cvui:
    """A class to implement a simple GUI in openCV for brightness control (Point Grey version).
Added an histogram window to evaluate saturation.
New version, using improved GUI elements found in cvui
NOTE: you should use the same cam object used for acquisition, with the same parameters.
Only adjust the brightness once, with the longest exposure time you plan to use, and keep
the values constant for all your acquisitions.

syntax: gui = expGUI_cvui(cam,[window])

- cam -> PyCapture Camera object, properly connected.
- window -> optional parameter, pass the name of the OpenCV window with the projected image
            NOTE: if it does not exist it will be created, be careful
- par -> an object containing the parameters from 'parameters.cfg'
- use the sliders to adjust color brightness and/or exposure time.
- the real-time acquisition runs until 'Enter' or 'Escape' is pressed.
"""
    def __init__(self,cam,par,wname='pattern',correction=[]):
        self.par = par # container for parameters
        self.cam = cam # camera objects
        self.Bb = [255] # Define brightness for reference
        self.Bg = [255]
        self.Br = [255]
        self.n = [1] # number of acquisitions
        self.exposure = [10] # Arbitrary starting value
        self.stop = [False] # Boolean value for stop checkbox
        self.h = 0
        self.step = 1 # Step to increase counters
        self.wname = wname # name of the pattern window
        self.n_acq = 0 # counter, n. of acquisitions
        self.correction = correction # gamma correction array
        self.start()
        
    def set_exposure(self):
        """Control camera exposure."""
        try:
            self.cam.setProperty(type=pc.PROPERTY_TYPE.SHUTTER,absValue=self.exposure[0])
        except pc.Fc2error as fc2Err:
            print('Error setting exposure: %s' % fc2Err)
    def set_blue(self):
        corr = np.ceil(self.correction[int(self.Bb[0])])
        self.ref[0:self.h,:,0] = corr # BLUE stripe
    def set_green(self):
        corr = np.ceil(self.correction[int(self.Bg[0])])
        self.ref[self.h:2*self.h,:,1] = corr # GREEN stripe
    def set_red(self):
        corr = np.ceil(self.correction[int(self.Br[0])])
        self.ref[2*self.h:3*self.h,:,2] = corr # RED stripe
        
    def reference(self,xRes,yRes):
        """Use this function to control the brightness level of the three channels"""
        ## Triple reference -> 3 horizontal stripes with B,G,R values. Control intensities below
        self.h = int(yRes/3)
        self.ref = np.zeros((yRes,xRes,3),dtype='uint8')
        self.ref[0:self.h,:,0] = self.Bb[0] # BLUE stripe
        self.ref[self.h:2*self.h,:,1] = self.Bg[0] # GREEN stripe
        self.ref[2*self.h:3*self.h,:,2] = self.Br[0] # RED stripe
        
        cvui.imshow(self.wname,self.ref)         
        
    def start(self):
        ## Background color - main window
        self.bg = np.zeros((530,840,3),'uint8')
        self.bg[:,:,:] = (20,20,20)

        ## initialize windows, resize and move
        cvui.init(['gui',self.wname,'histogram'],3)
        cv.resizeWindow('gui',840,530)
        cv.resizeWindow('histogram',512,300)

        ## Get info about exposure time
        info = self.cam.getPropertyInfo(pc.PROPERTY_TYPE.SHUTTER)
        tstep = (info.absMax - info.absMin)/542
		
        ## Set a limit to exposure to 1000ms (to use when you disable fps)
        if info.absMax > 500:
            expMax = 500
            tstep = 0.5
        else:
            expMax = info.absMax
        
        try:
            self.cam.startCapture()
        except pc.Fc2error as fc2Err:
            print('Error starting capture: %s' % fc2Err)

        ## Create and show reference picture
        self.reference(xRes=853,yRes=480)
        
        while(True):
            ## Calibration loop
            ## Capture image from camera
            for i in range(10): # number of retries
                try:
                    im = self.cam.retrieveBuffer()
                    break # break if image was retrieved successfully
                except pc.Fc2error as fc2Err:
                    print('Error retrieving buffer: %s' % fc2Err)

            ## Convert to color and reshape
            im = im.convert(pc.PIXEL_FORMAT.BGR) # from RAW to color (BGR 8bit)
            data = im.getData() # a long array of data (python list)
            width = im.getCols()
            heigth = im.getRows()
            frame = np.reshape(data,(heigth,width,3)) # Reshape to 2D color image

            ## Clean background
            self.bg[:,:,:] = (20,20,20)
            
            ## If resolution is 1280p, downsample to 640p
            if (width == 1280):
                frame = cv.resize(frame,(640,480),cv.INTER_NEAREST)
            
            ## Draw frame on main window
            cvui.image(self.bg,0,50,frame)

            ## Draw trackbar for exposure and adjust value if changed
            cvui.text(self.bg,20,20,'Exposure(ms)')
            
            if (cvui.trackbar(self.bg,130,0,700,self.exposure,info.absMin,expMax,1,"%.2Lf",\
                              cvui.TRACKBAR_DISCRETE,tstep)):
                self.set_exposure()
            
            ## Create column to draw counters for RGB
            cvui.beginColumn(self.bg,700,60,200,-1,20)
            ## BLUE
            cvui.text('BLUE')
            cvui.counter(self.Bb,self.step,"%d")
            ## Check invalid values
            if (self.Bb[0] > 255):
                self.Bb[0] = 255
            if (self.Bb[0] < 0):
                self.Bb[0] = 0
            ## GREEN
            cvui.text('GREEN')
            cvui.counter(self.Bg,self.step,"%d")
            ## Check invalid values
            if (self.Bg[0] > 255):
                self.Bg[0] = 255
            if (self.Bg[0] < 0):
                self.Bg[0] = 0
            ## RED
            cvui.text('RED')
            cvui.counter(self.Br,self.step,"%d")
            ## Check invalid values
            if (self.Br[0] > 255):
                self.Br[0] = 255
            if (self.Br[0] < 0):
                self.Br[0] = 0
            
            ## Number of acquisitions
            cvui.text('n. of acquisitions')
            cvui.counter(self.n,self.step,"%d")
            ## Check invalid values
            if (self.n[0] < 0):
                self.n[0] = 0
            
            cvui.endColumn()

            ## Draw main window
            cvui.imshow('gui',self.bg)

            ## Update pattern
            self.set_blue()
            self.set_green()
            self.set_red()
            cvui.imshow(self.wname,self.ref)
            

            ## calculate histograms
            hist_im = np.zeros((300,512,3),dtype=np.uint8) # background
            hist_b = cv.calcHist([frame],[0],None,[256],[0,256]) # BLUE histogram
            cv.normalize(hist_b,hist_b,alpha=0,beta=300,norm_type=cv.NORM_MINMAX)
            hist_g = cv.calcHist([frame],[1],None,[256],[0,256]) # GREEN histogram
            cv.normalize(hist_g,hist_g,alpha=0,beta=300,norm_type=cv.NORM_MINMAX)
            hist_r = cv.calcHist([frame],[2],None,[256],[0,256]) # RED histogram
            cv.normalize(hist_r,hist_r,alpha=0,beta=300,norm_type=cv.NORM_MINMAX)

            ## Draw histograms
            cvui.sparkline(hist_im,hist_b,0,0,512,300,0x0000ff)
            cvui.sparkline(hist_im,hist_g,0,0,512,300,0x00ff00)
            cvui.sparkline(hist_im,hist_r,0,0,512,300,0xff0000)
            cvui.imshow('histogram',hist_im)

            ## Get button input
            k = cv.waitKey(1) & 0xFF
            ## ENTER: continue program
            if k == 13:
                cv.destroyWindow('gui')
                cv.destroyWindow('histogram')
                self.cam.stopCapture()
                break
            ## ESCAPE: quit program
            elif k == 27: 
                print('Quitting...')
                cv.destroyAllWindows()
                self.cam.stopCapture()
                self.cam.disconnect()
                raise SystemExit
            ## '+': increase exposure one step
            elif k == 43:
                self.exposure[0] += (info.absMax-info.absMin)/542
                self.set_exposure()
            ## '-': decrease exposure one step
            elif k == 45:
                self.exposure[0] -= (info.absMax-info.absMin)/542
                self.set_exposure()
            ## '1': set step to 1
            elif k == 49:
                self.step = 1
            ## '5': set step to 5
            elif k == 53:
                self.step = 5
            ## '0': set step to 10
            elif k == 48:
                self.step = 10
    
        self.n_acq = 0 # initialize counter
        #TODO: put spatial frequencies in the config file (json can parse lists)
        f = self.par.fx / np.arange(self.par.nFreq,0,-1) # generate frequencies array
        f = np.insert(f,0,0) # add DC (zero frequency) at the beginning
        while((self.n_acq < self.n[0] or self.n[0] == 0) and not(self.stop[0]) ):
            ## Acquisition loop
            acquisitionRoutine(self.cam,self.par.xRes,self.par.yRes,self.par.w,f,self.par.nFreq,
                               self.par.nPhase,self.par.dt,self.correction,self.par.Bb,self.par.Bg,self.par.Br,
                               outPath=self.par.outPath,name=self.par.name,n_acq=self.n_acq)
            self.n_acq += 1 # increase counter
