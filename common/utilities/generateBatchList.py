# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:44:11 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import os,sys

sys.path.append('../')
sys.path.append('C:/PythonX/Lib/site-packages') ## Add PyCapture2 installation folder manually if doesn't work

from sfdi.getPath import getPath

batchpath = getPath('select directory containing data to batch process')
temp = os.listdir(batchpath)
files = [x for x in temp if '.npy' in x] 

with open('{}/batchfile.txt'.format(batchpath), 'w') as bfile:
    for i, file in enumerate(files):
        if i == 0:
            bfile.write('["{}/{}",\n'.format(batchpath, file))
        elif i == len(files) - 1:
            bfile.write('"{}/{}"]'.format(batchpath, file)) 
        else:
            bfile.write('"{}/{}",\n'.format(batchpath, file))