# -*- coding: utf-8 -*-
"""
Created on Thur Aug 16 16:17:28 2021

@author: Tushant - SRH Heidelberg University
email: tushuni@gmail.com
"""
import numpy as np
import PIL
from PIL import Image
from PIL import ImageTk as PIL_ImageTk
from tkinter import NW
import time, os
#from pathlib import Path

def gamma_calibration(root):

    ## Timing tests
    start = time.time()
    t_stamp = int(start) # get timestamp for the current acquisition

    dt = root.par['dt']
    name = "Gamma_CAPTURE"

    #print("Directory Path:", Path().absolute()) # Directory of current working directory, not __file__  
    #outPath = Path().absolute()
    outPath = root.par['outpath']
    # get exposure time, to normalize RAW data
    expt = int(root.cam.getExposure())
    curr_path = '{}/{:d}_{:s}_{:d}ms'.format(outPath, t_stamp, "Gamma_Cal", expt)  # create one folder for each timestamp
    if not os.path.exists(curr_path):
        os.makedirs(curr_path)

    for i in np.arange(0, 256, 5):
        vector = np.zeros((480, 854, 3), dtype='uint8')
        vector[:,:,:] = i
        im = Image.fromarray(vector)
        pattern = PIL_ImageTk.PhotoImage(im)
        root.PatternCanvas.create_image(0, 0, image=pattern, anchor=NW)
        root.update()
        time.sleep(dt/1000 + 0.05)
        frame = root.cam.capture(nframes=1, save=False)
        im = PIL.Image.fromarray(frame.astype('uint8'))
        im.save('{}/{}_{:d}.bmp'.format(curr_path, name, i),format='BMP', compression='raw')