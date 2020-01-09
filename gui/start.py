# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:48:24 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

import sys
from PyQt5.QtWidgets import QApplication,QDesktopWidget
sys.path.append('../common') # Add the common folder to path
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work
import sfdi
from sfdi.readParams2 import readParams
from mycsv import csvread

from MainWindow import MainWindow

if __name__ == '__main__':
    
    # Read acquisition parameters in a dictionary
    par = readParams('../acquisition/parameters.cfg')
    
    # Automatically set screen resolution
    screen1 = QDesktopWidget().screenGeometry(0) # main screen
    screen2 = QDesktopWidget().screenGeometry(1) # secondary screen / projector
    
    par['W'] = int(screen1.width())
    par['H'] = int(screen1.height())
    par['xRes'] = int(screen2.width())
    par['yRes'] = int(screen2.height())
    
    app = QApplication(sys.argv)
    win = MainWindow(par=par)
    #app.exec_()