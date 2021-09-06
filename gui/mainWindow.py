# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 11:24:14 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Main gui script for interface in tkInter
"""
import sys, os
import tkinter as tk
from tkinter import ttk
import PIL
from PIL import ImageTk as PIL_ImageTk
# sys.path.append('../common') # Add the common folder to path
from camera.DummyCam import DummyCam as Camera  # change imported library as needed

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
        self.exposure = 33.3  # ms, exposure time
        
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
        s = ttk.Style()
        s.configure('Debug.TFrame', background='gray', border=1)
        self.CommandFrame = ttk.Frame(self, height=240, width=854, style='Debug.TFrame')
        self.CommandFrame['padding'] = (0, 0, 0, 0)
       
        self.CommandFrame.columnconfigure(0, weight=1)
        self.CommandFrame.columnconfigure(1, weight=2)
        self.CommandFrame.columnconfigure(2, weight=1)
        self.CommandFrame.columnconfigure(3, weight=1)
        self.CommandFrame.pack()
        
        # Exposure slider
        expMin, expMax, expStep = self.cam.getExposureLim()
        self.ExpLabel = ttk.Label(self.CommandFrame, text='Exposure (ms):',
                                  font=('Calibri', 18))
        self.ExpLabel.grid(row=0, column=0, padx=10)
        self.ExpSlider = ttk.Scale(self.CommandFrame, from_=expMin, to=expMax,
                                   orient=tk.HORIZONTAL)
        self.ExpSlider.grid(row=0, column=1, padx=10, pady=10,
                            sticky=tk.N+tk.S+tk.W+tk.E)
        # Exposure indicator
        self.ExpIndicator = ttk.Label(self.CommandFrame,
                            text='{:.1f}'.format(self.exposure), font=('Calibri', 18))
        self.ExpIndicator.grid(row=0, column=2, padx=10)
        

if __name__ == "__main__":
    win = MainWindow()
    win.mainloop()