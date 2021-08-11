# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 09:49:53 2021

@author: luibe59
"""
import numpy as np
from matplotlib import pyplot as plt

gammaFile = 'C:/Users/luibe59/Documents/Processing Code/common/gammaCorrection_smartBeam.csv'
gamma = np.genfromtxt(gammaFile, dtype=float, delimiter=',')

plt.figure(1, figsize=(10,6))
plt.plot(np.arange(256), gamma, '-k', linewidth=2)
plt.xlabel('input', fontsize=14)
plt.ylabel('output', fontsize=14)
plt.grid(True, linestyle=':')
plt.tight_layout()