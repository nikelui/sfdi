# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:51:14 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import numpy as np
import cv2 as cv
import cvui
from sfdi.acquisition.acquisitionRoutine_keynote import acquisitionRoutine

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
    def __init__(self, cam, par, wname='pattern', correction=[]):
        self.par = par # container for parameters
        self.cam = cam # camera objects
        self.n = [1] # number of acquisitions
        self.exposure = [33] # Arbitrary starting value
        self.stop = [False] # Boolean value for stop checkbox
        self.h = 0
        self.step = 1 # Step to increase counters
        self.wname = wname # name of the pattern window
        self.n_acq = 0 # counter, n. of acquisitions
        self.correction = correction # gamma correction array
        # Since bool() does not work very well with strings
        if self.par['blueboost'] in ['False','false',0,'0','None','none','No','no','']:
            self.blueboost = False
        else:
            self.blueboost = True # Everything else is true
        self.start()

    def set_exposure(self):
        """Control camera exposure."""
        self.cam.setExposure(self.exposure[0])  # in ms
        
    def reference(self,xRes,yRes):
        """Use this function to control the brightness level of the three channels"""
        ## Triple reference -> 3 horizontal stripes with B,G,R values. Control intensities below
        ## New approach: multiple stripes
        self.h = 30
        self.rowb = [x for x in range(yRes) if x % (self.h*3) < self.h]
        self.rowg = [x for x in range(yRes) if x % (self.h*3) >= self.h and x % (self.h*3) < self.h*2]
        self.rowr = [x for x in range(yRes) if x % (self.h*3) >= self.h*2]
        self.ref = np.zeros((yRes,xRes,3),dtype='uint8')
        self.ref[self.rowb,:,0] = 255 # BLUE stripe
        self.ref[self.rowg,:,1] = 255 # GREEN stripe
        self.ref[self.rowr,:,2] = 255 # RED stripe
        
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
        expMin, expMax, tstep = self.cam.getExposureLim()
		
        ## Set a limit to exposure to 500ms (to use when you disable fps)
        if expMax > 500:
            expMax = 500
            tstep = 0.5

        ## Create and show reference picture
        self.reference(xRes=self.par['xres'],yRes=self.par['yres'])
        
        while(True):
            ## Calibration loop
            ## Capture image from camera
            for i in range(10): # number of retries
                frame = self.cam.preview(nframes=1, save=False)
                break # break if image was retrieved successfully
            height, width = frame.shape[:2]  # image has 3 dimensions

            ## Clean background
            self.bg[:,:,:] = (20,20,20)
            
            ## If resolution is larger than 640p, downsample to 640p
            if (width > 640 or height > 480):
                frame = cv.resize(frame,(640,480),cv.INTER_NEAREST)
            
            ## Draw frame on main window
            cvui.image(self.bg,0,50,cv.cvtColor(frame,cv.COLOR_GRAY2RGB))

            ## Draw trackbar for exposure and adjust value if changed
            cvui.text(self.bg,20,20,'Exposure(ms)')
            
            if (cvui.trackbar(self.bg,130,0,700,self.exposure,expMin,expMax,1,"%.2Lf",\
                              cvui.TRACKBAR_DISCRETE,tstep)):
                self.set_exposure()
            
            ## Number of acquisitions
            cvui.text(self.bg,700,60,'n. of acquisitions')
            cvui.counter(self.bg,700,80,self.n,self.step,"%d")
            ## Check invalid values
            if (self.n[0] < 0):
                self.n[0] = 0
            
#            cvui.endColumn()

            ## Draw main window
            cvui.imshow('gui',self.bg)

#            ## Update pattern
#            self.set_blue()
#            self.set_green()
#            self.set_red()
            cvui.imshow(self.wname,self.ref)
            
            ## calculate RGB histograms
            if frame.shape[-1] == 3:
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
            
            ## for now, assume monochrome and calculate B/W histogram
            else:
                hist_im = np.zeros((300,512),dtype=np.uint8) # background
                hist = cv.calcHist([frame],[0],None,[256],[0,256]) # B/W histogram
                cv.normalize(hist,hist,alpha=0,beta=300,norm_type=cv.NORM_MINMAX)
                ## Draw histogram
                cvui.sparkline(hist_im,hist,0,0,512,300,0xffffff)

            cvui.imshow('histogram',hist_im)

            ## Get button input
            k = cv.waitKey(1) & 0xFF
            ## ENTER: continue program
            if k == 13:
                cv.destroyWindow('gui')
                cv.destroyWindow('histogram')
                break
            ## ESCAPE: quit program
            elif k == 27: 
                print('Quitting...')
                cv.destroyAllWindows()
                self.cam.close()
                raise SystemExit
            ## '+': increase exposure one step
            elif k == 43:
                self.exposure[0] += tstep
                self.set_exposure()
            ## '-': decrease exposure one step
            elif k == 45:
                self.exposure[0] -= tstep
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
        f = self.par['fx'] # new version, put the frequencies in the parameters file
        while((self.n_acq < self.n[0] or self.n[0] == 0) and not(self.stop[0]) ):
            ## Acquisition loop
            ## TODO: break from infinite loop using the return value from acquisitionRoutine
            ## TODO: remove every reference to BB, BG, BR in the parameters
            ret = acquisitionRoutine(self.cam,self.par['xres'],self.par['yres'],self.par['width'],f,len(f)-1,
                               self.par['nphase'],self.par['dt'],self.correction,self.par['bb'],
                               self.par['bg'],self.par['br'],outPath=self.par['outpath'],
                               name=self.par['name'],fname=self.par['fname'],n_acq=self.n_acq,
                               nchannels=self.cam.nchannels,blueBoost=self.blueboost,
                               diagonal=self.par['diagonal'])
            self.n_acq += 1 # increase counter
            self.stop[0] = ret # use return value to break from loop
