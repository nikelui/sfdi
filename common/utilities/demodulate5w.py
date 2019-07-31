# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:20:54 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""

import sys
sys.path.append('../')
import sfdi

from rawDataLoad import rawDataLoad
from stackPlot import stackPlot
from sfdi.readParams2 import readParams

par = readParams('../../processing/parameters.cfg')
AC,_ = rawDataLoad(par,'Select data folder')
stackPlot(AC)