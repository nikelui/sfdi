# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:20:54 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

import sys
sys.path.append('../')
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work

import sfdi

from rawDataLoad import rawDataLoad
from stackPlot import stackPlot
from sfdi.readParams3 import readParams

par = readParams('../../processing/parameters.ini')
AC,_ = rawDataLoad(par,'Select data folder')
stackPlot(AC)