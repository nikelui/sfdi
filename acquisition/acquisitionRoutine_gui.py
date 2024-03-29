# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:41:37 2019

@author: luibe59
"""
import numpy as np
import PIL
from PIL import ImageTk as PIL_ImageTk
from tkinter import NW
import time, os
import threading
from sfdi.acquisition.sinPattern import sinPattern

def saveFunc(i, j, k, nchannels, curr_path, name, frame):
    """Function to save data, run in a separate thread"""
    for _i in range(nchannels):
        im = PIL.Image.fromarray(frame[:,:,_i].astype('uint8'))
        im.save('{}/{}_{:d}-{:d}-{:d}.bmp'.format(curr_path, name, i*nchannels+_i, j, k),
                format='BMP', compression='raw')
        # print("Pictures saved to: {}\n".format(curr_path))
    
def acquisitionRoutine(root):
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
    start = time.time()
    t_stamp = int(start) # get timestamp for the current acquisition
    
    # Unpacking parameters here to avoid changing half the script
    nPhase = root.par['nphase']
    #nFreq = root.par['nfreq']
    nchannels = root.cam.nchannels
    correction = root.par['gamma']
    xRes = root.par['xres']
    yRes = root.par['yres']
    dt = root.par['dt']
    w = root.par['width']
    f = root.par['fx']
    nFreq = len(f)
    Bb = Bg = Br = 255
    outPath = root.par['outpath']
    fname = root.par['fname']
    name = root.par['name']
    
    # Retrieve camera resolution
    H,W = root.cam.getResolution()  # height x width
    # dataMat = np.zeros((H, W,nPhase*(nFreq+1)*nchannels*3),dtype=float)
    root.stop = False # Insert a flag for stopping
    
    # get exposure time, to normalize RAW data
    expt = int(root.cam.getExposure())
    curr_path = '{}/{:d}_{:s}_{:d}ms'.format(outPath, t_stamp, fname, expt)  # create one folder for each timestamp
    if not os.path.exists(curr_path):
        os.makedirs(curr_path)
    
    ## New: increase the exposure time for blue channel (SNR is very low on tissues)
    ## consider to use a slightly longer pause between patterns
    #TODO: wrap in try/except for safety
    if (root.par['blueboost']):
        expT = root.cam.getExposure()
        root.cam.setExposure(expT + 33.3)   
    pshift = 2*np.pi/nPhase # phase shift for demodulation [default = 2/3 pi]
    
    # Acquire BLUE / BG
    for i in range(nFreq):
        for p in range(nPhase):
            _,_,_,Ib = sinPattern(xRes, yRes, w, f[i], pshift*p, Bb, correction,'b')
            # DEBUG
            #Ib = debugPrint(xRes,yRes,'%d_%d%d%d' % (n_acq,0,i,p))
            ## fix: opencv 4.0.0 does not like float images, so convert to uint8
            Ib = (Ib*255).astype('uint8')             
            patt = PIL.Image.fromarray(Ib)
            pattern = PIL_ImageTk.PhotoImage(patt)
            root.PatternCanvas.create_image(0, 0, image=pattern, anchor=NW)
            root.update()
            if (root.par['blueboost']):
                time.sleep(dt/1000 + 0.05)
            else:
                time.sleep(dt/1000)
            frame = root.cam.capture(nframes=1, save=False)
            ## New: save single images (because of memory constrains)
            th = threading.Thread(target=saveFunc,args=(0, i, p, nchannels, curr_path, name, frame,))
            th.start()
    
    if (root.par['blueboost']):
        # Here, return exposure to previous value
        root.cam.setExposure(expT)
        
    # Acquire GREEN
    for i in range(nFreq):
        for p in range(nPhase):
            _,_,Ig,_ = sinPattern(xRes,yRes,w,f[i],pshift*p,Bg,correction,'g')
            #Ig = debugPrint(xRes,yRes,'%d_%d%d%d' % (n_acq,1,i,p))
            ## fix: opencv 4.0.0 does not like float images, so convert to uint8
            Ig = (Ig*255).astype('uint8')  
            patt = PIL.Image.fromarray(Ig)
            pattern = PIL_ImageTk.PhotoImage(patt)
            root.PatternCanvas.create_image(0, 0, image=pattern, anchor=NW)
            root.update()
            time.sleep(dt/1000)
            frame = root.cam.capture(nframes=1, save=False)
            ## New: save single images (because of memory constrains)
            th = threading.Thread(target=saveFunc,args=(1, i, p, nchannels, curr_path, name, frame,))
            th.start()
            # Change approach: keep everything in a big matrix and save later
            # for ch in range(nchannels):
            #     dataMat[:,:, p + (nPhase*i) + (ch + nchannels)*(nPhase*(nFreq+1))] = frame[:,:,ch]
    ##time.sleep(dt/2000)
    
    # Acquire RED
    for i in range(nFreq):
        for p in range(nPhase):
            _,Ir,_,_ = sinPattern(xRes,yRes,w,f[i],pshift*p,Br,correction,'r')
            #Ir = debugPrint(xRes,yRes,'%d_%d%d%d' % (n_acq,2,i,p))
            ## fix: opencv 4.0.0 does not like float images, so convert to uint8
            Ir = (Ir*255).astype('uint8')
            patt = PIL.Image.fromarray(Ir)
            pattern = PIL_ImageTk.PhotoImage(patt)
            root.PatternCanvas.create_image(0, 0, image=pattern, anchor=NW)
            root.update()
            time.sleep(dt/1000)
            frame = root.cam.capture(nframes=1, save=False)
            ## New: save single images (because of memory constrains)
            th = threading.Thread(target=saveFunc,args=(2, i, p, nchannels, curr_path, name, frame,))
            th.start()
            # Change approach: keep everything in a big matrix and save later
            # for ch in range(nchannels):
            #     dataMat[:,:, p + (nPhase*i) + (ch + nchannels*2)*(nPhase*(nFreq+1))] = frame[:,:,ch]
    end = time.time()   
    print("Capture time: {:.2f} seconds\n".format(float(end-start)))   
    
    # th = threading.Thread(target=saveFunc,args=(nFreq,nPhase,nchannels,curr_path,name,dataMat,))
    # th.start()
    ## End