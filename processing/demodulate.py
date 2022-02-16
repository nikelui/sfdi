# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:20:42 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
from numpy import sqrt

def demodulate(Im1, Im2, Im3):
    """Algorithm to perform time-domain amplitude demodulation.

Syntax: AC,DC = demodulate(Im1,Im2,Im3)

- Input are three B/W images in the form of numpy 2D arrays. They should contain a sinusoidal
    pattern with a phase delay of 2/3 pi between them.
- Output are the AC and DC components of the pattern (float arrays).
"""
    ## Convert first
    Im1 = Im1.astype(float)
    Im2 = Im2.astype(float)
    Im3 = Im3.astype(float)
    AC = sqrt(2)/3 * sqrt((Im1-Im2)**2 + (Im2-Im3)**2 + (Im3-Im1)**2)
    DC = (Im1 + Im2 + Im3) / 3.0
    return(AC,DC)