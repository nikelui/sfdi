# -*- coding: utf-8 -*-
"""
Created on Thu Jan  13 10:59:24 2020

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import sys
sys.path.append('../../../../common') # Add the common folder to path

from fbs_runtime.application_context.PyQt5 import ApplicationContext
from MainWindow import MainWindow


if __name__ == '__main__':
    appctxt = ApplicationContext()       # 1. Instantiate ApplicationContext
    
    window = MainWindow()    
    
    exit_code = appctxt.app.exec_()      # 2. Invoke appctxt.app.exec_()
    sys.exit(exit_code)