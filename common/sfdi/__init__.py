"""A collection of useful function and classes used to perform SFDI.
The most important functions were re-organized into a package for better manageability
"""
__version__ = 0.10
## Import sub-modules
from sfdi.expGUI_cvui import expGUI_cvui
from sfdi.demodulate import demodulate
from sfdi.getPath import getPath
from sfdi.setCamera_pg import setCamera_pg
from sfdi.setWindow import setWindow
from sfdi.camCapt_pg import camCapt_pg
from sfdi.crop import crop
from sfdi.sinPattern import sinPattern
from sfdi.acquisitionRoutine import acquisitionRoutine
from sfdi.readParams import readParams