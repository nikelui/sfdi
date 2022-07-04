# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 14:58:45 2022

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se

Command line script required by pyInstaller. The only function is to load "main"
from the source code and run it.
"""
from phantom_calculator.__main__ import main

if __name__ == '__main__':
    main()