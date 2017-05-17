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


def parallel_scs_LTE(wn_range_tot, n_threads = 8, db_file = hit08_25, mol = None):
    """
    Drives the calculation of the Absorption coefficient.
    """

    time0 = time.time()

    linee_tot = spcl.read_line_database(db_file, mol = mol, freq_range = wn_range_tot)

    #molec_ok = [6, 23, 26]
    # però a select gli devo passare le molec vere perchè devo fare due conti con il non LTE e mi servono le temperature dei livelli
    #spcl.select_lines(molecs = molec_ok, take_top_lines = 0.3)

    abs_coeff = smm.prepare_spe_grid(wn_range_tot)

    processi = []
    coda = []
    outputs = []

    for i in range(n_threads):
        coda.append(Queue())
        processi.append(Process(target=do_for_th,args=(wn_range_tot, linee_tot, abs_coeff, i, coda[i])))
        processi[i].start()

    for i in range(n_threads):
        outputs.append(coda[i].get())

    for i in range(n_threads):
        processi[i].join()


    shapes_tot = []

    for output in outputs:
        shapes_tot += output

    abs_coeff.add_lines_to_spectrum(shapes_tot)

    print('Finito spettro con {} linee in {} s!'.format(len(linee_tot), time.time()-time0))

    return abs_coeff


def do_for_th(wn_range, linee_tot, abs_coeff, i, coda):
    step_nlin = len(linee_tot)/n_threads
    linee = linee_tot[step_nlin*i:step_nlin*(i+1)]
    if i == n_threads-1:
        linee = linee_tot[step_nlin*i:]

    print('Hey! Questo è il ciclo {} con {} linee su {}!'.format(i,len(linee),len(linee_tot)))

    shapes, lws, dws = smm.spect_calc_LTE(linee, abs_coeff, 6, 1, 16., 296., 1000.)

    print('Ciclo {} concluso in {} s!'.format(i,time.time()-time0))

    coda.put(shapes)

    return


def spect_calc_LTE(linee_mol, abs_coeff, mol, iso, MM, Temp, Pres, LTE = True, sp_step = 5.e-4, max_lines = 0, imxsig = 13010, fraction_to_keep = None):
    """
    This routine calculates the spectral shapes of all lines at the given temperature and pressure. All quantities are in the wavenumber space [cm^{-1}].
    """
    print('Inside spect_calc.. reading lines and calculating linshapes')

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
