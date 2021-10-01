# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:07:34 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import json
import configparser

def readParams(filename):
    """A function to read a .ini file and store parameters in a dict.
Uses json to parse values to the correct datatype (allows to read lists)"""
    p = {} # empty dictionary
    cfg = configparser.ConfigParser()
    cfg.read(filename) # open file
    
    for key in cfg['DEFAULT'].keys():
        try:
            value = json.loads(cfg['DEFAULT'][key])
        except json.JSONDecodeError:
            value = str(cfg['DEFAULT'][key]) # if it cannot be parsed, default to string
        p[key] = value
        
    return p

