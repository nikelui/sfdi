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
# sys.path.append('../common') # Add the common folder to path
# from sfdi.camera.DummyCam import DummyCam as Camera  # change imported library as needed

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
        
        # configure window
        self.title('RaspiSFDI gui')
        self.geometry('854x960+0+0')
        self.resizable(False,False)
        self.attributes('-topmost', True)
        self.attributes('-toolwindow', True)
        
        # here should go the camera
        # self.cam = Camera()
        
        # Canvas to display projected image
        self.TopFrame = ttk.Frame(self, height=480, width=854)
        self.TopFrame['padding'] = (0,0,0,0)
        # self.TopFrame['margin'] = (0,0,0,0)
        self.TopFrame.pack()
        
        # Canvas to show projected patterns
        self.PatternCanvas = tk.Canvas(self.TopFrame, height=480, width=854, bd=0, bg='gray')
        self.PatternCanvas.pack()
        
        
        
        


if __name__ == "__main__":
    win = MainWindow()
    # win.focus_force()
    win.mainloop()