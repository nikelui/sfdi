# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 11:24:14 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Main gui script for interface in tkInter
"""
import sys, os
from time import time
import numpy as np
import tkinter as tk
from tkinter import ttk
import PIL
from PIL import ImageTk as PIL_ImageTk
# sys.path.append('../common') # Add the common folder to path
from camera.DummyCam import DummyCam as Camera  # change imported library as needed

def sfdiRoutine():
    # TODO: import this
    print('Start SFDI!')


class MainWindow(tk.Tk):
    def __init__(self):
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
        
        # configure window
        self.title('RaspiSFDI gui')
        self.geometry('854x960+0+0')
        self.resizable(False,False)
        self.attributes('-topmost', True)
        # self.attributes('-toolwindow', True)
        
        # here should go the camera
        self.cam = Camera()
        
        ######## Pattern ########
        
        # Frame to contain projected canvas
        self.TopFrame = ttk.Frame(self, height=480, width=854)
        self.TopFrame['padding'] = (0,0,0,0)
        self.TopFrame.pack()
        
        # load test image for default projection
        test = PIL.Image.open('test.png')
        self.test_image = test.resize((854,480))
        self.pattern = PIL_ImageTk.PhotoImage(self.test_image)
        
        # Canvas to show projected patterns
        self.PatternCanvas = tk.Canvas(self.TopFrame, height=480, width=854,
                                       bd=-2, bg='black')
        self.PatternCanvas.create_image(0, 0, image=self.pattern, anchor=tk.NW)
        self.PatternCanvas.pack()
        
        ######## Preview ########
        
        # Frame to keep previews
        self.PreviewFrame = ttk.Frame(self, height=240, width=854)
        self.PreviewFrame['padding'] = (0, 0, 0, 0)
        self.PreviewFrame.pack()

        # Canvas to display preview
        self.PreviewCanvas = tk.Canvas(self.PreviewFrame, height=240, width=427,
                                       bd=-2, bg='red')
        self.PreviewCanvas.place(x=0, y=0)
        # TODO: get camera preview to display in PreviewCanvas
        
        # Canvas to display histogram
        self.HistCanvas = tk.Canvas(self.PreviewFrame, height=240, width=427,
                                       bd=-2, bg='black')
        self.HistCanvas.place(x=427, y=0)
        # TODO: draw the histogram on HistCanvas
        
        ######## Commands ########
        # DEBUG frame
        self.style.configure('Debug.TFrame', border=1)
        self.CommandFrame = ttk.Frame(self, height=240, width=854, style='Debug.TFrame')
        self.CommandFrame['padding'] = (10, 10, 10, 10)
        self.CommandFrame.pack()
        
        # Exposure slider
        expMin, expMax, expStep = self.cam.getExposureLim()
        self.expMin = expMin
        self.expMax = expMax
        self.expStep = expStep
        self.ExpLabel = ttk.Label(self.CommandFrame, text='Exposure (ms):', font=('Calibri', 18))
        self.ExpLabel.grid(row=0, column=0, padx=10)
        
        self.ExpSlider = ttk.Scale(self.CommandFrame, from_=expMin, to=expMax, variable=self.exposure,
                                   orient=tk.HORIZONTAL, command=self.SliderUpdate, length=400)
        self.ExpSlider.set(self.exposure.get())  # initial update
        self.ExpSlider.grid(row=0, column=2, padx=10, pady=10)

        self.ExpIndicator = ttk.Label(self.CommandFrame, textvariable=self.exposure, font=('Calibri', 18), width=5)
        self.ExpIndicator.grid(row=0, column=1, padx=10)
        
        # Increase / decrease buttons
        self.expButtonFrame = ttk.Frame(self.CommandFrame, height=30, width=20)
        self.expButtonFrame.grid(row=0, column=3)
        
        self.upButton = ttk.Button(self.expButtonFrame, text='+', command=self.increaseExp, width=2)
        self.upButton.pack(padx=1, pady=1)
        self.downButton = ttk.Button(self.expButtonFrame, text='-', command=self.decreaseExp, width=2)
        self.downButton.pack(padx=1, pady=1)
        
        # # DEBUG button
        # self.debug = ttk.Button(self.CommandFrame, text='Debug', command=self.debugButton)
        # self.debug.grid(row=0, column=4, padx=10)
        
        # Start button
        self.startButton = ttk.Button(self.CommandFrame, text='START', command=sfdiRoutine)
        self.startButton.grid(row=1, column=0, padx=10, pady=35)
        
        # Capture button
        self.captureButton = ttk.Button(self.CommandFrame, text='Capture', command=self.capture)
        self.captureButton.grid(row=1, column=1, padx=10, pady=35)
        
        self.style.configure('TButton', font=('Calibri', 14))
        
    ############## Callbacks ################  
    
    def SliderUpdate(self, exp):
        self.exposure.set(exp)
        self.cam.setExposure(self.exposure.get())
    
    def debugButton(self):
        print(self.exposure.get())
        
    def increaseExp(self):
        current_exp = self.exposure.get()
        self.exposure.set(current_exp + self.expStep)
        self.cam.setExposure(self.exposure.get())
    
    def decreaseExp(self):
        current_exp = self.exposure.get()
        self.exposure.set(current_exp - self.expStep)
        self.cam.setExposure(self.exposure.get())
        
    def capture(self):
        frame = self.cam.capture()
        im = PIL.Image.fromarray(frame)
        im.save('{}.bmp'.format(int(time())))  # use timestamp as filename
        
if __name__ == "__main__":
    win = MainWindow()
    win.mainloop()