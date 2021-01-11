# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 09:59:26 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
Scrpt to convert from numpy .npz archives to Matlab .mat files
"""
import sys, os
import numpy as np
from scipy.io import savemat
from sfdi.common.sfdi.getPath import getPath

# command line arguments
flags = sys.argv

if "-debug" in flags:
    debug = True
else:
    debug = False

path = getPath("select data path")
files = [x for x in os.listdir(path) if '.npz' in x]

for file in files:
    name = file[:-4]
    if debug:
        print(name)  # Debug
    data = np.load('{}/{}.npz'.format(path, name))
    var = data.files
    mdict = {v:data[v] for v in var}
    savemat('{}/{}.mat'.format(path, name), mdict)
    print('saved file: {}'.format(name))