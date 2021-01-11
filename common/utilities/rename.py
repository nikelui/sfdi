# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 10:39:38 2021

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
Script to convert raw data filenames from old to new convention
"""
import os,re

names = [f for f in os.listdir('TS_tests/') if re.match(r'[0-9]', f)]
names.sort()

for a in names:
    temp = a.split('_')
    #rename = '_'.join((temp[0],temp[2][:-4],temp[1]+'.mat'))
    #rename = '_'.join((temp[0],temp[2],temp[1]))
    rename = '_'.join((temp[0],''.join(temp[1:-1]),temp[-1]))
    rename = rename.replace('base','Base')
    rename = rename.replace('occlusion','Occlusion')
    rename = rename.replace('release','Release')
    os.rename('TS_tests/'+a,'TS_tests/'+rename)
    #print(rename)
