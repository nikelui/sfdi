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

with open('{}/batchfile.ini'.format(batchpath), 'w') as bfile:
    bfile.write('[DEFAULT]\n')
    bfile.write('# Remember to copy the automatically generated paths under the correct variables\n')
    bfile.write('# NOTE: you need to put all the values in a single line after the key')
    bfile.write('ph_data=\n')
    bfile.write('ph_ref=\n')
    bfile.write('batch_data=[\n\n]\n')
    bfile.write('# here is the directories list\n')
    for dd in [x for x in os.listdir(batchpath) if os.path.isdir('{}/{}'.format(batchpath, x))]:
        bfile.write('"{}/{}",\n'.format(batchpath, dd))
    