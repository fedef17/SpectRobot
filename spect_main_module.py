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
import pickle
import lineshape
from multiprocessing import Process, Queue

hit08_25 = '/home/fedefab/Scrivania/Research/Dotto/Spect_data/HITRAN/HITRAN08_2-5mu.par'

############ MAIN ROUTINES USED FOR SPECT_ROBOT.PY
def prepare_spe_grid(wn_range, sp_step = 5.e-4, units = 'cm_1'):
    """
    Prepares the SpectralObject for the range considered.
    """

    spoffo = np.arange(wn_range[0],wn_range[1]+sp_step/2,sp_step,dtype = np.float64)
    spect_grid = spcl.SpectralGrid(spoffo, units = units)
    spoffo = np.zeros(len(spect_grid.grid), dtype = np.float64)
    abs_coeff = spcl.SpectralObject(spoffo, spect_grid, units = units)

    return abs_coeff


def spect_calc_LTE(linee, abs_coeff, mol, iso, MM, Temp, Pres, sp_step = 5.e-4, max_lines = 0, imxsig = 13010):
    """
    This routine calculates the spectral quantities (absorption and emission coefficient in non-LTE) needed in RadTransfer.py. All quantities are in the wavenumber space [cm^{-1}].
    To be added:
    - non-LTE correction of the partition function Q.
    - calculation of LUTs.
    - optimization of the lineshape calculation.
    - switch for scattering.
    """
    print('Inside spect_calc.. reading lines and calculating linshapes')

    linee_mol = [lin for lin in linee if lin.Mol == mol and lin.Iso == iso]
    essesss = [lin.Strength for lin in linee_mol]
    essort = np.sort(np.array(essesss))[-max_lines]

    linee_mol = [lin for lin in linee_mol if lin.Strength > essort]

    lin_grid = np.arange(-imxsig*sp_step/2,imxsig*sp_step/2,sp_step, dtype = np.float64)

    Q_part = spcl.CalcPartitionSum(mol, iso, temp = Temp)

    time0 = time.time()
    time_100 = time.time()
    shapes = []
    lws = []
    dws = []
    for ii,lin in zip(range(len(linee_mol)),linee_mol):
        #print('linea {} at {}'.format(ii,lin.Freq))
        ind_ok, fr_grid_ok = spcl.closest_grid(abs_coeff.spectral_grid, lin.Freq)
        lin_grid_ok = spcl.SpectralGrid(lin_grid+fr_grid_ok, units = 'cm_1')

        S = lin.CalcStrength_nonLTE(Temp,Temp,Q_part)
        shape, lw, dw = lin.MakeShapeLine(lin_grid_ok, Temp, Pres, MM, Strength = S)
        shapes.append(shape)
        dws.append(dw)
        lws.append(lw)

        #abs_coeff.add_to_spectrum(shape)
        if ii % 100 == 0:
            #print('Made 100 lines in {} s'.format(time.time()-time_100))
            time_100 = time.time()

    print('Made {} lines in {} s'.format(len(linee_mol),time.time()-time0))

    return shapes, lws, dws



def spect_calc_nonLTE(isomolec,level,lines_level_up,lines_level_down):
    """

    """
