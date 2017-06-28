#!/usr/bin/python
# -*- coding: utf-8 -*-

import decimal
import numpy as np
import sys
import os.path
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import math as mt
from numpy import linalg as LA
import scipy.constants as const
import warnings
import spect_base_module as sbm
import spect_classes as spcl
import time
import lineshape
import copy
import time
import spect_main_module as smm
import pickletools
import pickle


Temps = [175.0, 175.0, 175.0, 175.0, 175.0]
Press = [0.001, 10.0, 1.0, 0.1, 0.01]

pl.ion()
pl.figure(17)

nom = './abscoeff_ch4_T{:03d}K_P{:04d}Pa.pic'.format(175,0)
abs_coeff_tot_0, _, _ = pickle.load(open(nom,'r'))
gnok = abs_coeff_tot_0.spectrum > np.max(abs_coeff_tot_0.spectrum)/10.

for T, P in zip(Temps,Press):
    nom = './abscoeff_ch4_T{:03d}K_P{:04d}Pa.pic'.format(int(T), int(100.*P))
    abs_coeff_tot, emi_coeff_tot, abs_coeffs_iso = pickle.load(open(nom,'r'))
    pl.plot(abs_coeff_tot.spectral_grid.grid[gnok], (abs_coeff_tot.spectrum[gnok]-abs_coeff_tot_0.spectrum[gnok])/abs_coeff_tot_0.spectrum[gnok], label = 'T = {}, P = {}'.format(T, P))

pl.legend()
pl.show()

sys.exit()

db_file = '/home/fedefab/Scrivania/Research/Dotto/Spect_data/HITRAN/HITRAN2012_2-5mu.par'

### LOADING LINES
print('Loading lines...')

time0 = time.time()

wn_range = [2958.,2959.]
linee = spcl.read_line_database(db_file,freq_range = wn_range)

t600, p600, ch4 = pickle.load(open('./local_vibtemp_ch4.pic','r'))

print(t600, p600)

abso = smm.prepare_spe_grid([2958.,2958.+13009*5.e-4])
Temp = 167.965
Pres = 2.8960e-04

Q_part = spcl.CalcPartitionSum(6, 1, temp = Temp)
iii = 0
i_ok = [11,64,119,282,344,360]

for lin in linee:
    iii+=1
    if iii in i_ok:
        S = lin.CalcStrength_nonLTE(Temp, Temp, Q_part)
        print('Strength', S)
        print(iii, lin.Freq)
        lin.MakeShapeLine(abso.spectral_grid, Temp, Pres, ch4.iso_1.MM, keep_memory = True,verbose=True)
        Gco = lin.Calc_Gcoeffs(Temp, Pres, ch4.iso_1)
        print('Gcoabs', Gco['absorption'])
