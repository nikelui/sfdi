# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:56:27 2022

@author: Luigi Belcastro - Linköping University
email: luigi.belcastro@liu.se
"""
import tkinter as tk
from tkinter import ttk
# import numpy as np
# from matplotlib import pyplot as plt

# TODO: read validation doc
def validate(P):
    if str.isdigit(P) or P == "":
        return True
    else:
        return False

# NOTES and ASSUMPTIONS
# - 1mg ink == 1uL
# - Black ink has a flat absorption spectrum (different from melanin)
# TODO: Include also Aluminum oxide parameters

root = tk.Tk()
root.title('Phantom calculator')

root.columnconfigure(tuple(range(6)), weight=1)
root.columnconfigure(0, weight=3)
root.rowconfigure(tuple(range(6)), weight=1)

# SELECT mode
mode_label = ttk.Label(root, text='MODE')
mode_label.grid(column=0, row=0, padx=30, pady=10)

mode = tk.IntVar()
radio1 = ttk.Radiobutton(root, text='Calculate Volumes', value=1, variable=mode)
radio1.grid(column=1, row=0, columnspan=2, padx=30)
radio2 = ttk.Radiobutton(root, text='Calculate O.P', value=2, variable=mode)
radio2.grid(column=3, row=0, columnspan=2, padx=30)

# LEFT side: optical properties and wavelengths boxes
vcdm = root.register(validate)

wv_target = ttk.Label(root, text='wv (target)')
wv_target.grid(column=0, row=1)
wv_box = ttk.Entry(root, validate='all', validatecommand=(vcdm, "%P"), width=6)
wv_box.grid(column=1, row=1, padx=10, pady=10)

mua_target = ttk.Label(root, text='mua (target)')
mua_target.grid(column=0, row=2)
mua_box = ttk.Entry(root, validate='all', validatecommand=(vcdm, "%P"), width=6)
mua_box.grid(column=1, row=2, padx=10, pady=10)

mus_target = ttk.Label(root, text='mu\'s (target)')
mus_target.grid(column=0, row=3)
mus_box = ttk.Entry(root, validate='all', validatecommand=(vcdm, "%P"), width=6)
mus_box.grid(column=1, row=3, padx=10, pady=10)

wv_range = ttk.Label(root, text='wv [min max]')
wv_range.grid(column=0, row=4, padx=30, pady=10)
wv_min = ttk.Entry(root, validate='all', validatecommand=(vcdm, "%P"), width=6)
wv_min.grid(column=1, row=4, padx=10, pady=10)
wv_max = ttk.Entry(root, validate='all', validatecommand=(vcdm, "%P"), width=6)
wv_max.grid(column=2, row=4, padx=10, pady=10)
# TODO: extra validation, wv_max > wv_min

# RIGHT side: volume / concentration boxes
mua_label = ttk.Label(root, text='mua')
mua_label.grid(column=4, row=1, padx=30, pady=10)
mus_label = ttk.Label(root, text='mu\'s')
mus_label.grid(column=5, row=1, padx=10, pady=10)

concentration = ttk.Label(root, text='C (mg/100mL)')
concentration.grid(column=3, row=2, padx=30, pady=10)
C_mua = ttk.Entry(root, validate='all', validatecommand=(vcdm, "%P"), width=6)
C_mua.grid(column=4, row=2, padx=10, pady=10)
C_mus = ttk.Entry(root, validate='all', validatecommand=(vcdm, "%P"), width=6)
C_mus.grid(column=5, row=2, padx=10, pady=10)
weigth = ttk.Label(root, text='Quantity (mg)')
weigth.grid(column=3, row=3, padx=10, pady=10)
w_mua = ttk.Entry(root, validate='all', validatecommand=(vcdm, "%P"), width=6)
w_mua.grid(column=4, row=3, padx=10, pady=10)
w_mus = ttk.Entry(root, validate='all', validatecommand=(vcdm, "%P"), width=6)
w_mus.grid(column=5, row=3, padx=10, pady=10)
volume = ttk.Label(root, text='TOT volume')
volume.grid(column=3, row=4, padx=10, pady=10)
v_tot = ttk.Entry(root, validate='all', validatecommand=(vcdm, "%P"), width=6)
v_tot.grid(column=4, row=4, padx=10, pady=10)
root.mainloop()


# # TARGET optical properties
# mua_target = 0.021
# mus_target = 1.4
# wv_target = 800

# # optical properties at 650nm
# C_TiO_ref = 73  # mg/100ml
# C_ink_ref = 25  # mg/100ml
# musp_ref = 1  # mm^-1
# mua_ref = 0.02  # mm^-1

# B_TiO = 1.25
# A_ref = musp_ref / (np.power(650, -B_TiO))  # 3282

# mus_ref_at_target = A_ref * np.power(wv_target, -B_TiO)  # reference concentration at wv_target

# # output
# C_TiO_target = C_TiO_ref * mus_target / mus_ref_at_target  # mg/100ml
# C_ink_target = C_ink_ref * mua_target / mua_ref  # mg/100ml


# # Plots
# wv = np.arange(400, 901, 1)
# MUA = np.ones(wv.shape) * mua_ref
# MUS_REF = A_ref * np.power(wv, -B_TiO)
# MUS_TARGET = MUS_REF * mus_ref_at_target / musp_ref

# # plt.plot(wv, MUS_REF, label='{:.1f} mg/100ml'.format(C_TiO_ref))
# # plt.plot(wv, MUS_TARGET, label='{:.1f} mg/100ml'.format(C_TiO_target))