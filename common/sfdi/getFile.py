# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:43:51 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import tkinter as tk # for file dialog
from tkinter import filedialog

def getFile(title='Select file'):
    """Call a file dialog to select a file. You can pass an optional string for the dialog title."""
    ## to hide main window
    root = tk.Tk()
    root.withdraw()
    root.lift()
    ## open dialog to select file
    path = filedialog.askopenfilename(initialdir='./',title=title,filetype=[('text files','*.txt'),
                                      ('Matlab files','*.mat'),('all files','*.*')])
    
    return path

def getFiles(title='Select files'):
    """Call a file dialog to select multiple files. You can pass an optional string for the dialog title."""
    ## to hide main window
    root = tk.Tk()
    root.withdraw()
    root.lift()
    ## open dialog to select file
    path = filedialog.askopenfilenames(initialdir='./',title=title,filetype=[('text files','*.txt'),
                                      ('Matlab files','*.mat'),('all files','*.*')])
    
    return path