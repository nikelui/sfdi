# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 11:24:14 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Main gui script for interface in tkInter
"""
import os, io
from time import time
import numpy as np
import tkinter as tk
from tkinter import ttk
import PIL
from PIL import ImageTk as PIL_ImageTk
from sfdi.acquisition.acquisitionRoutine_gui import acquisitionRoutine
from sfdi.acquisition.Gamma_Calibration import gamma_calibration


class MainWindow(tk.Tk):
    def __init__(self, cam, par):
        """
        Parameters
        ----------
        cam : Camera class
        par : DICTIONARY
            Python dict containing the parameters

        Returns
        -------
        None.
        """
        super().__init__()
        
        # Theme
        if os.name == 'nt':
            self.style = ttk.Style(self)
            self.style.theme_use('winnative')
        else:
            self.style = ttk.Style(self)
            self.style.theme_use('clam')
        
        # Values to track
        self.exposure = tk.DoubleVar(value=33.3)  # ms, exposure time
        self.expMin = 0
        self.expMax = 0
        self.expStep = 0
        self.Nacquisition = tk.StringVar(value=1)
        self.stop = True  # flag to stop current routine
        
        # here should go the camera
        self.cam = cam
        # acquisition parameters
        self.par = par
        
        # configure window
        self.title('RaspiSFDI gui')
        self.geometry('{}x{}+0+0'.format(self.par['xres'], self.par['yres']*2))
        self.resizable(False,False)
        self.attributes('-topmost', True)
        self.attributes('-fullscreen', True)
        # self.attributes('-toolwindow', True)
        

        
        ######## Pattern ########
        
        # Frame to contain projected canvas
        self.TopFrame = ttk.Frame(self, height=self.par['yres'], width=self.par['xres'])
        self.TopFrame['padding'] = (0,0,0,0)
        self.TopFrame.pack()
        
        # load test image for default projection
        test = PIL.Image.open('../gui/test.png')
        self.test_image = test.resize((self.par['xres'], self.par['yres']))
        self.pattern = PIL_ImageTk.PhotoImage(self.test_image)
        
        # Canvas to show projected patterns
        self.PatternCanvas = tk.Canvas(self.TopFrame, height=self.par['yres'],
                                       width=self.par['xres'], bd=-2, bg='black')
        self.PatternCanvas.create_image(0, 0, image=self.pattern, anchor=tk.NW)
        self.PatternCanvas.pack()
        
        ######## Preview ########
        
        # Frame to keep previews
        self.PreviewFrame = ttk.Frame(self, height=self.par['yres']//2, width=self.par['xres'])
        self.PreviewFrame['padding'] = (0, 0, 0, 0)
        self.PreviewFrame.pack()

        # Canvas to display preview
        self.PreviewCanvas = tk.Canvas(self.PreviewFrame, height=self.par['yres']//2,
                                       width=self.par['xres']//2, bd=-2, bg='black')
        self.PreviewCanvas.place(x=0, y=0)
        self.setlocation_preview()  # display overlay preview
        
        # Canvas to display histogram
        self.HistCanvas = tk.Canvas(self.PreviewFrame, height=self.par['yres']//2,
                                    width=self.par['xres']//2, bd=-2, bg='black')
        self.HistCanvas.place(x=self.par['xres']//2, y=0)
        self.updateHist()
        # TODO: draw the histogram on HistCanvas
        
        ######## Commands ########
        # DEBUG frame
        self.style.configure('Debug.TFrame', border=1)
        self.CommandFrame = ttk.Frame(self, height=self.par['yres']//2, width=self.par['xres'],
                                      style='Debug.TFrame')
        self.CommandFrame['padding'] = (10, 10, 10, 10)
        self.CommandFrame.pack()
        
        # Exposure slider
        expMin, expMax, expStep = self.cam.getExposureLim()
        self.expMin = expMin
        self.expMax = expMax
        self.expStep = expStep
        self.ExpLabel = ttk.Label(self.CommandFrame, text='Exposure (ms):')
        self.ExpLabel.grid(row=0, column=0, padx=10)
        
        self.ExpSlider = ttk.Scale(self.CommandFrame, from_=expMin, to=expMax, variable=self.exposure,
                                   orient=tk.HORIZONTAL, command=self.sliderUpdate,
                                   length=self.par['xres']//2)
        self.ExpSlider.set(self.exposure.get())  # initial update
        self.ExpSlider.grid(row=0, column=2, padx=10, pady=10)

        self.ExpIndicator = ttk.Label(self.CommandFrame, textvariable=self.exposure, width=5)
        self.ExpIndicator.grid(row=0, column=1, padx=10)
        
        # Increase / decrease buttons
        self.expButtonFrame = ttk.Frame(self.CommandFrame, height=30, width=20)
        self.expButtonFrame.grid(row=0, column=3)
        
        self.upButton = ttk.Button(self.expButtonFrame, text='+', command=self.increaseExp, width=2)
        self.upButton.pack(padx=1, pady=1)
        self.downButton = ttk.Button(self.expButtonFrame, text='-', command=self.decreaseExp, width=2)
        self.downButton.pack(padx=1, pady=1)
        
        # DEBUG button
        self.debug = ttk.Button(self.CommandFrame, text='Debug', command=self.debugButton)
        self.debug.grid(row=2, column=2, padx=10)
        
        # Start button
        self.startButton = ttk.Button(self.CommandFrame, text='START', command=self.sfdiRoutine)
        self.startButton.grid(row=1, column=0, padx=10, pady=10)
        
        # number of acquisition
        self.acqNum = ttk.Spinbox(self.CommandFrame, from_=0, to=10, format='%.0f', increment=1,
                                  textvariable=self.Nacquisition, state='readonly', width=3)
        self.acqNum.grid(row=1, column=1, padx=0)
        
        # Select gamma or SFDI routine
        self.routineMenu = ttk.Combobox(self.CommandFrame, state='readonly', width=30,
                                        values=('Calibrate gamma', 'SFDI acquisition'))
        self.routineMenu.current(1)  # default is SFDI
        self.routineMenu.grid(row=1, column=2, padx=10)
        
        # Capture button
        self.captureButton = ttk.Button(self.CommandFrame, text='Single\ncapture', command=self.capture)
        self.captureButton.grid(row=2, column=0, padx=10, pady=10)
        
        self.style.configure('TButton')
        
        # Close button
        self.closeButton = ttk.Button(self.CommandFrame, text='Quit', command=self.close, width=5)
        self.closeButton.grid(row=2, column=3)
        
    ############## Callbacks ################  
    def setlocation_preview(self):
        self.cam.stoppreview()
        self.cam.setstartpreview() 
    
    def sliderUpdate(self, exp):
        self.exposure.set(exp)
        self.cam.setExposure(self.exposure.get())
    
    def debugButton(self):
        print('Exp: {}'.format(self.exposure.get()))
        print('Mode: {}\nType: {}'.format(self.routineMenu.get(),
                                          type(self.routineMenu.get())))
        
        self.updateHist((np.random.randint(0,15,size=(256,3)).T * np.arange(256)).T)
        
    def increaseExp(self):
        current_exp = self.exposure.get()
        if current_exp + self.expStep <= self.expMax:
            self.exposure.set(current_exp + self.expStep)
            self.cam.setExposure(self.exposure.get())
    
    def decreaseExp(self):
        current_exp = self.exposure.get()
        if current_exp - self.expStep >= self.expMin:
            self.exposure.set(current_exp - self.expStep)
            self.cam.setExposure(self.exposure.get())
            
    def capture(self):
        frame = self.cam.capture()
        im = PIL.Image.fromarray(frame.astype('uint8'))
        im.save('{}.bmp'.format(int(time())))  # use timestamp as filename
        
    def close(self):
        self.destroy()
        self.cam.close()
    
    def histogram(self):
        # Create the in-memory stream
        stream = io.BytesIO()   
        # Generates a 3D RGB array and stores it in rawCapture
        self.cam.cam.capture(stream, format="jpeg",use_video_port=True)
        # "Rewind" the stream to the beginning so we can read its content
        stream.seek(0)
        image = PIL.Image.open(stream)
        #frame = raw_capture.array
        R, G, B = image.split()
        hist = np.stack([R.histogram(), G.histogram(), B.histogram()], axis=1)
        return hist
        
    def updateHist(self):
        if self.stop:  # if routine is not running
            # Assume hist is a (256 x 3) array
            # get histogram
            hist = self.histogram()
            # first clean the canvas
            self.HistCanvas.delete('all')
            # scale x and y coordinates to the canvas size
            xaxis = np.arange(0, 256) / 255 * (self.par['xres']/2)
            # loop through 3 color channels
            color = ['red', 'green', 'blue']
            for _i in range(3):
                yaxis = hist[:,_i] / np.max(hist) * (self.par['yres']/2)
                # print(yaxis)  # debug
                # note: y coordinates are reversed, zero is at the top
                coords = [(x, self.par['xres']//4 - y) for x, y in zip(xaxis, yaxis)]
                self.HistCanvas.create_line(coords, fill=color[_i], width=1.5)
        self.after(500, self.updateHist)  # iteratively update
    
    def sfdiRoutine(self):
        # TODO: need a way to stop acquisition
        if self.routineMenu.get() == 'SFDI acquisition':
            print('Start SFDI!')
            acquisitionRoutine(self)
            print('Camera config Info')
            self.cam.Setting_info()
        elif self.routineMenu.get() == 'Calibrate gamma':
            print('Start gamma!')
            gamma_calibration(self)
            print('Camera config Info')
            self.cam.Setting_info()
        else:
            print('You should not be here!')
        
# if __name__ == "__main__":
#     from sfdi.acquisition import __path__ as par_path
#     from sfdi.common.readParams import readParams
#     # Only works with raspberry pi camera
#     # from sfdi.camera.DummyCam import DummyCam 
    
#     par = readParams('{}/parameters.ini'.format(par_path[0]))
#     cam = DummyCam()
    
#     win = MainWindow(cam, par)
#     win.mainloop()