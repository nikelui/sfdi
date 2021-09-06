# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:41:37 2019

@author: luibe59
"""
import numpy as np
import time,os,sys
import cv2 as cv
from sfdi.sinPattern import sinPattern
from sfdi.camCapt_pg import camCapt_pg
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work
import PyCapture2 as pc
import threading

def saveFunc(nFreq,nPhase,nchannels,curr_path,name,dataMat):
    """Function to save data, run in a separate thread"""
    for i in range(nchannels*3):
        for j in range(nFreq+1):
            for k in range(nPhase):
                cv.imwrite(curr_path + '/' + name + '_%d-%d-%d.bmp' % (i,j,k),\
                           dataMat[:,:,k + (nPhase*j) + i*(nPhase*(nFreq+1))].astype('uint8'))
    print("Pictures saved to: %s\n" % curr_path)
    
def debugPrint(xRes,yRes,text):
    """Debug function to use instead of sinPattern()"""
    Im = np.zeros((yRes,xRes,3),dtype='uint8')
    cv.putText(Im,text,
               fontFace=cv.FONT_HERSHEY_TRIPLEX,
               org=(xRes//3,yRes//2),
               fontScale=3,
               color=(255,255,255),
               lineType=8,
               thickness=2)
    return Im
    

def acquisitionRoutine(cam,xRes,yRes,w,f,nFreq,nPhase=3,dt=100.,correction=[],Bb=255,Bg=255,Br=255,
                       outPath='./',name='im',fname='test',n_acq=1,nchannels=3,blueBoost=False):
    """Routine to acquire a full SFDI image set and save data, now with threading.

NOTE: to work correctly, you need to have an OpenCV window called 'pattern' showing fullscreen on your projector

@Inputs:
    - cam: PyCapture Camera object
    - xRes,yRes: resolution of projector
    - w: physical size of projected screen
    - f: array with spatial frequency values
    - nFreq: number of spatial frequencies (should be len(f)-1)
    - nPhase: number of phases for demodulation (default: 3)
    - dt: buffer time between image acquisition (ms)
    - correction: gamma correction array for sinusoidal pattern
    - Bb,Bg,Br: intensity of red, blue and green colour.
    - outPath: path where to save output images (default: current folder)
    - name: name to prepend to saved images before the _xyz tag.
    - n_acq: acquisition n. (Debug puposes)
    - nchannels: number of color channels that the camera can acquire (default: 3 for RGB)
    - blueBoost: flag to increase exposure time for blue
"""
    ## Timing tests
    start = cv.getTickCount()
    
    t_patt = []
    t_imshow = []
    t_wait = []
    t_sleep = []
    t_capture = []
    
    t_stamp = int(time.time()) # get timestamp for the current acquisition
    
    # Retrieve camera resolution
    H,W = cam.getResolution()  # height x width
    dataMat = np.zeros((H,W,nPhase*(nFreq+1)*nchannels*3),dtype=float)
    
#    try:
#        (vid,_) = cam.getVideoModeAndFrameRate()
#    except pc.Fc2error as fc2Err:
#        print('Error retrieving videoMode and frameRate: %s' % fc2Err)
#    # Matrix to store pictures  
#    if vid == pc.VIDEO_MODE.VM_1280x960Y8 or vid == pc.VIDEO_MODE.VM_1280x960Y16:
#        dataMat = np.zeros((960,1280,nPhase*(nFreq+1)*nchannels*3),dtype=float)
#    elif vid == pc.VIDEO_MODE.VM_640x480Y8 or vid == pc.VIDEO_MODE.VM_640x480Y16:
#        dataMat = np.zeros((480,640,nPhase*(nFreq+1)*nchannels*3),dtype=float)
    
    stop = False # Insert a flag for stopping
    
    ## New: increase the exposure time for blue channel (SNR is very low on tissues)
    ## consider to use a slightly longer pause between patterns
    #TODO: wrap in try/except for safety
    if (blueBoost):
        expT = cam.getExposure()
        cam.setExposure(expT+33.3)
        
#        # first save old value
#        try:
#            prop = cam.getProperty(pc.PROPERTY_TYPE.SHUTTER)
#            expT = float(prop.absValue)
#        except pc.Fc2error as fc2Err:
#            print('Error getting exposure property: %s' % fc2Err)
#        # Increase exposure by a multiple of 16.67ms (projector framerate is 60Hz)
#        try:
#            prop = pc.Property(pc.PROPERTY_TYPE.SHUTTER,absControl=True,absValue=expT+33.3,autoManualMode=False)
#            cam.setProperty(prop) # DONE
#        except pc.Fc2error as fc2Err:
#            print('Error setting BLUE exposure time: %s' % fc2Err)
    
    pshift = 2*np.pi/nPhase # phase shift for demodulation [default = 2/3 pi]
    
    # Acquire BLUE / BG
    for i in range(nFreq+1):
        for p in range(nPhase):
            t1 = cv.getTickCount()
            _,_,_,Ib = sinPattern(xRes,yRes,w,f[i],pshift*p,Bb,correction,'b')
            # DEBUG
            #Ib = debugPrint(xRes,yRes,'%d_%d%d%d' % (n_acq,0,i,p))
            t2 = cv.getTickCount()
            ## fix: opencv 4.0.0 does not like float images, so convert to uint8
            Ib = (Ib*255).astype('uint8')             
            cv.imshow('pattern',Ib[...,::-1])
            t3 = cv.getTickCount()
            k = cv.waitKey(1)
            if k & 0xff == 27: # if press 'ESCAPE', raise flag
                stop = True
            t4 = cv.getTickCount()
            if (blueBoost):
                time.sleep(dt/1000 + 0.05)
            else:
                time.sleep(dt/1000)
            t5 = cv.getTickCount()
#            frame = camCapt_pg(cam,1,False)
            frame = cam.capture(nframes=1, save=False)
            t6 = cv.getTickCount()
            # Change approach: keep everything in a big matrix and save later
            for ch in range(nchannels):
                dataMat[:,:, p + (nPhase*i) + ch*(nPhase*(nFreq+1))] = frame[:,:,ch]
            t_patt.append((t2-t1)/cv.getTickFrequency())
            t_imshow.append((t3-t2)/cv.getTickFrequency())
            t_wait.append((t4-t3)/cv.getTickFrequency())
            t_sleep.append((t5-t4)/cv.getTickFrequency())
            t_capture.append((t6-t5)/cv.getTickFrequency())
    ##time.sleep(dt/2000)
    
    if (blueBoost):
        # Here, return exposure to previous value
        cam.setExposure(expT)
#        try:
#            prop = pc.Property(pc.PROPERTY_TYPE.SHUTTER,absControl=True,absValue=expT,autoManualMode=False)
#            cam.setProperty(prop) # DONE
#        except pc.Fc2error as fc2Err:
#            print('Error setting exposure time: %s' % fc2Err)
        
    # Acquire GREEN
    for i in range(nFreq+1):
        for p in range(nPhase):
            t1 = cv.getTickCount()
            _,_,Ig,_ = sinPattern(xRes,yRes,w,f[i],pshift*p,Bg,correction,'g')
            #Ig = debugPrint(xRes,yRes,'%d_%d%d%d' % (n_acq,1,i,p))
            t2 = cv.getTickCount()
            ## fix: opencv 4.0.0 does not like float images, so convert to uint8
            Ig = (Ig*255).astype('uint8')  
            cv.imshow('pattern',Ig[...,::-1])
            t3 = cv.getTickCount()
            k = cv.waitKey(1)
            if k & 0xff == 27: # if press 'ESCAPE', raise flag
                stop = True
            t4 = cv.getTickCount()
            time.sleep(dt/1000)
            t5 = cv.getTickCount()
#            frame = camCapt_pg(cam,1,False)
            frame = cam.capture(nframes=1, save=False)
            t6 = cv.getTickCount()
            # Change approach: keep everything in a big matrix and save later
#            dataMat[:,:, p + (nPhase*i) + 1*(nPhase*(nFreq+1))] = frame[:,:,0] # save GB channel (1)
#            dataMat[:,:, p + (nPhase*i) + 2*(nPhase*(nFreq+1))] = frame[:,:,1] # save green channel (2)
#            dataMat[:,:, p + (nPhase*i) + 3*(nPhase*(nFreq+1))] = frame[:,:,2] # save GR channel (3)
            for ch in range(nchannels):
                dataMat[:,:, p + (nPhase*i) + (ch + nchannels)*(nPhase*(nFreq+1))] = frame[:,:,ch]
            t_patt.append((t2-t1)/cv.getTickFrequency())
            t_imshow.append((t3-t2)/cv.getTickFrequency())
            t_wait.append((t4-t3)/cv.getTickFrequency())
            t_sleep.append((t5-t4)/cv.getTickFrequency())
            t_capture.append((t6-t5)/cv.getTickFrequency())
    ##time.sleep(dt/2000)
    
    # Acquire RED
    for i in range(nFreq+1):
        for p in range(nPhase):
            t1 = cv.getTickCount()
            _,Ir,_,_ = sinPattern(xRes,yRes,w,f[i],pshift*p,Br,correction,'r')
            #Ir = debugPrint(xRes,yRes,'%d_%d%d%d' % (n_acq,2,i,p))
            t2 = cv.getTickCount()
            ## fix: opencv 4.0.0 does not like float images, so convert to uint8
            Ir = (Ir*255).astype('uint8')  
            cv.imshow('pattern',Ir[...,::-1])
            t3 = cv.getTickCount()
            k = cv.waitKey(1)
            if k & 0xff == 27: # if press 'ESCAPE', raise flag
                stop = True
            t4 = cv.getTickCount()
            time.sleep(dt/1000)
            t5 = cv.getTickCount()
#            frame = camCapt_pg(cam,1,False)
            frame = cam.capture(nframes=1, save=False)
            t6 = cv.getTickCount()
            # Change approach: keep everything in a big matrix and save later
            for ch in range(nchannels):
                dataMat[:,:, p + (nPhase*i) + (ch + nchannels*2)*(nPhase*(nFreq+1))] = frame[:,:,ch]
            t_patt.append((t2-t1)/cv.getTickFrequency())
            t_imshow.append((t3-t2)/cv.getTickFrequency())
            t_wait.append((t4-t3)/cv.getTickFrequency())
            t_sleep.append((t5-t4)/cv.getTickFrequency())
            t_capture.append((t6-t5)/cv.getTickFrequency())
    
    end = cv.getTickCount()
    
    #cam.disconnect() # not here if you need multiple acquisitions
    #cv.destroyAllWindows()
    
    print("Capture time: %.2f seconds\n" % (float(end-start)/cv.getTickFrequency()))
    
#    info = cam.getProperty(pc.PROPERTY_TYPE.SHUTTER)
#    expt = int(info.absValue) # get exposure time, to normalize RAW data
    expt = cam.getExposure()
    
    # for loop to save on file
    curr_path = (outPath + '/%d_%s_%dms' % (t_stamp,fname,expt)) # create one folder for each timestamp
    if not os.path.exists(curr_path):
        os.makedirs(curr_path)
    
    th = threading.Thread(target=saveFunc,args=(nFreq,nPhase,nchannels,curr_path,name,dataMat,))
    th.start()
    ## This is a bottleneck (especially if saving at full resolution)
    ## TODO: might consider putting it on a separate thread for speed
    #saveFunc(nFreq,nPhase,curr_path,name,dataMat)
    
    return stop # if flag was raised, this should be True, otherwise False
    ## End
if (__name__ == '__main__'):
    xRes,yRes = (640,480)
    Im = debugPrint(xRes,yRes,"%d_%d%d%d" % (1,0,0,0))
    cv.imshow('pattern',Im)
    cv.waitKey(0)
    cv.destroyAllWindows()