# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:48:49 2019

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import json
from collections import OrderedDict # New: this will keep the parameters ordered

def readParams(filename):
    """A function to read a config file and store parameters in a dict.
New implememtation, using json to parse values (allows to read lists)"""
    p = OrderedDict() # empty dictionary
    ff = open(filename,'r') # open file
    
    for line in ff: # loop over file
        if line.startswith('#'):
            continue
        else:
            a = line.strip().split('=')
            key = a[0]
            ## Try to parse correct value
            try:
                value = json.loads(a[1])
            except json.JSONDecodeError:
                value = str(a[1]) # if it cannot be parsed, default to string
            p[key] = value
    ff.close()
    return p