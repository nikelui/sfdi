# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:25:04 2019

@author: luibe59
"""
import tkinter as tk # for file dialog
from tkinter import filedialog

def getPath(title='Select folder'):
    """Call a file dialog to select a path. You can pass an optional string
for the dialog title."""
    ## to hide main window
    root = tk.Tk()
    root.withdraw()
    root.lift()
    ## open dialog to select folder
    path = filedialog.askdirectory(initialdir='./',title=title)
    return path