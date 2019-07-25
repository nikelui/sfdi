# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:43:56 2019

@author: luibe59
"""
class parameters(object):
    """Just an empty class, to populate with parameters."""
    pass

def readParams(filename):
    """A function to read a config file and store parameters in an object"""
    p = parameters() # empty object
    ff = open(filename,'r') # open file
    
    for line in ff: # loop over file
        if line.startswith('#'):
            continue
        else:
            a = line.strip().split('=')
            key = a[0]
            ## Try to parse correct value
            try:
                value = int(a[1])
            except ValueError:
                try:
                    value = float(a[1])
                except ValueError:
                    value = str(a[1])
            
            setattr(p,key,value) # save data into variable
    ff.close()
    return p