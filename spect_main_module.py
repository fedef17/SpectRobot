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
import copy

n_threads = 8

############ MAIN ROUTINES USED FOR SPECT_ROBOT.PY
def date_stamp():
    strin = '_'+time.ctime().split()[2]+'-'+time.ctime().split()[1]+'-'+time.ctime().split()[4]
    return strin


# CLASSES


class LookUpTable(object):
    """
    This class represent a look-up table for a specific molecule/isotope.
    """

    def __init__(self, isomolec, tag = None):
        self.tag = tag
        self.mol = isomolec.mol
        self.iso = isomolec.iso
        self.MM = isomolec.MM
        self.isomolec = copy.deepcopy(isomolec)
        self.level_sets = []
        return

    def make(self, spectral_grid, lines, PTcouples):
        """
        Builds the LUT.
        """

        print('Producing LUT for mol {}, iso {}. The following levels are considered: {}'.format(self.mol,self.iso,self.isomolec.levels))

        for lev in self.isomolec.levels:
            print('Building LutSet for level {} of mol {}, iso {}'.format(lev,self.mol,self.iso))
            set1 = LutSet(self.mol, self.iso, self.MM, getattr(self.isomolec, lev))
            set1.make(spectral_grid, lines, PTcouples)
            self.level_sets.append(set1)

        return

    def CPU_time_estimate(self,lines,PTcouples):
        linee_ok = [lin for lin in lines if (lin.Mol == self.mol and lin.Iso == self.iso)]

        n_lev = len(self.isomolec.levels)
        n_lin = len(linee_ok)
        n_PT = len(PTcouples)

        time = n_lin * 3./30000. * n_PT * 3
        print('Estimated time is about {:8.1f} min'.format(time))

        return time

    def export(self, filename):
        pickle.dump(self, open(filename,'w'))
        return



class LutSet(object):
    """
    This class represent a single entry of look-up table for a single level, for all temps/press and for all ctypes. Each ctype is an element of a dictionary.
    """

    def __init__(self, mol, iso, MM, level):
        self.mol = mol
        self.iso = iso
        self.MM = MM
        self.level = copy.deepcopy(level)
        return

    def make(self, spectral_grid, lines, PTcouples):
        """
        Produces the full set for PTcouples.
        """
        ctypes = ['sp_emission','ind_emission','absorption']
        set_ = dict(zip(ctypes,[[],[],[]]))

        minimal_level_string = ''
        for qu in self.level.get_quanta()[0]:
            minimal_level_string += '{:1d}'.format(qu)
            minimal_level_string += ' '
        minimal_level_string = minimal_level_string[:-1]

        for ctype in ctypes:
            print('Producing set for '+ctype+'...')

            if ctype == 'sp_emission' or ctype == 'ind_emission':
                linee_mol = [lin for lin in lines if (minimal_level_string in lin.Up_lev_str and lin.Mol == self.mol and lin.Iso == self.iso)]
            elif ctype == 'absorption':
                linee_mol = [lin for lin in lines if (minimal_level_string in lin.Lo_lev_str and lin.Mol == self.mol and lin.Iso == self.iso)]

            if len(linee_mol) == 0:
                print('NO lines in {} for level {}'.format(ctype,minimal_level_string))
                continue

            for [P,T] in PTcouples:
                print('PTcouple: {}, {}'.format(P,T))
                gigi = spcl.SpectralGcoeff(ctype, spectral_grid, self.mol, self.iso, self.MM, minimal_level_string)
                gigi.BuildCoeff(lines, T, P)
                set_[ctype].append(gigi)
                print('Added')

        return


# FUNCTIONS


def prepare_spe_grid(wn_range, sp_step = 5.e-4, units = 'cm_1'):
    """
    Prepares the SpectralObject for the range considered.
    """

    spoffo = np.arange(wn_range[0],wn_range[1]+sp_step/2,sp_step,dtype = np.float64)
    spect_grid = spcl.SpectralGrid(spoffo, units = units)
    spoffo = np.zeros(len(spect_grid.grid), dtype = np.float64)
    abs_coeff = spcl.SpectralObject(spoffo, spect_grid, units = units)

    return abs_coeff


def parallel_scs_LTE(wn_range_tot, n_threads = n_threads, db_file = None, mol = None):
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


def do_for_th(linee_tot, abs_coeff, i, coda):
    step_nlin = len(linee_tot)/n_threads
    linee = linee_tot[step_nlin*i:step_nlin*(i+1)]
    if i == n_threads-1:
        linee = linee_tot[step_nlin*i:]

    print('Hey! Questo è il ciclo {} con {} linee su {}!'.format(i,len(linee),len(linee_tot)))

    shapes, lws, dws = smm.abs_coeff_calc_LTE(linee, abs_coeff, 6, 1, 16., 296., 1000.)

    print('Ciclo {} concluso in {} s!'.format(i,time.time()-time0))

    coda.put(shapes)

    return


def abs_coeff_calc_LTE(linee_mol, abs_coeff, mol, iso, MM, Temp, Pres, LTE = True, sp_step = 5.e-4, max_lines = 0, imxsig = 13010, fraction_to_keep = None):
    """
    This routine calculates the spectral shapes of all lines at the given temperature and pressure. All quantities are in the wavenumber space [cm^{-1}].

    To be addes:
        - Switch for LUTs loading, if already produced
        - Switch for LUTs production
        - Switch for simple calculation at a fixed (P,T)
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


def emiss_coeff_calc_LTE(linee_mol, abs_coeff, mol, iso, MM, Temp, Pres, LTE = True, sp_step = 5.e-4, max_lines = 0, imxsig = 13010, fraction_to_keep = None):
    pass


def read_Gcoeffs_from_LUTs():
    # if not makeLUTs:
    #     # Check if LUTs are present
    #     if os.path.isfile(cartLUTs+fileLUTs):
    #         print('Loading LUTs from {} ...'.format(cartLUTs+fileLUTs))
    #         LUTs = read_LUTs(isomolecs, cartLUTs+fileLUTs)
    #     else:
    #         raise ValueError('LUTs not found at {}! Set param < makeLUTs > to True if you want to calculate them for the first time.'.format(cartLUTs+fileLUTs))
    # elif makeLUTs:
    #     LUTs = make_LUTs(isomolecs, cartLUTs+fileLUTs)

    pass


def calc_single_Gcoeff():
    pass

def makeLUT_nonLTE_Gcoeffs(spectral_grid, lines, molecs, atmosphere, cartLUTs = None, tagLUTs = 'LUT_', n_pres_levels = None, pres_step_log = 0.1, temp_step = 5.0, save_LUTs = True, n_threads = n_threads):
    """
    Calculates the G_coeffs for the isomolec_levels at Temp and Pres.
    :param isomolecs: A list of isomolecs objects or a single one.
    """

    # Define pressure levels, for each pres check which temps are needed for the full atm and define a set of temperatures to be used at that pressure. Build something to be given as input to make LUTs.
    logpres0 = np.log(np.max(atmosphere.pres))
    logpresMIN = np.log(np.min(atmosphere.pres))
    log_pressures = logpresMIN + np.arange(0,(logpres0-logpresMIN)+0.5*pres_step_log,pres_step_log)
    print('Built set of {} pressure levels from {} to {} with logstep = {}.'.format(len(log_pressures),logpresMIN,logpres0,pres_step_log))

    pressures = np.exp(log_pressures)

    n_dim = len(atmosphere.grid)

    temps = []
    okke = (atmosphere.pres <= pressures[1])
    print(np.any(okke))
    temps_pres = atmosphere.temp[okke]
    temps.append([np.min(temps_pres),np.max(temps_pres)])
    print('For level {} with pres {}, temp in range: {} <-> {}'.format(len(temps)-1,pressures[0],temps[-1][0],temps[-1][1]))

    i = 0
    for pres0, pres1, pres2 in zip(pressures[:-2],pressures[1:-1],pressures[2:]):
        # Find the corresponding Temps.
        okke = (atmosphere.pres >= pres0) & (atmosphere.pres <= pres2)
        print(i,np.any(okke))
        temps_pres = atmosphere.temp[okke]
        temps.append([np.min(temps_pres),np.max(temps_pres)])
        print('For level {} with pres {}, temp in range: {} <-> {}'.format(len(temps)-1,pres1,temps[-1][0],temps[-1][1]))
        i+=1

    okke = (atmosphere.pres >= pressures[-2])
    print(np.any(okke))
    temps_pres = atmosphere.temp[okke]
    temps.append([np.min(temps_pres),np.max(temps_pres)])
    print('For level {} with pres {} hPa, temp is in range: {} <-> {}'.format(len(temps)-1,pressures[-1],temps[-1][0],temps[-1][1]))

    # round_values for temp at 5 K steps and build couples P/T
    PTcouples = []
    for pres, trange in zip(pressures,temps):
        t_0 = int(trange[0]) / int(temp_step)
        t_1 = int(trange[1]+temp_step) / int(temp_step)
        all_t = np.arange(t_0*temp_step,(t_1+1.)*temp_step,temp_step)
        #print(trange,all_t)
        for temp in all_t:
            PTcouples.append([pres,temp])

    print('Building LUTs for {} pres/temp couples... This may take some time... like {} minutes, maybe, not sure at all. Good luck ;)'.format(len(PTcouples),3.*len(PTcouples)))

    LUTS = []
    names = []
    for molec in molecs:
        for isoname in molec.all_iso:
            isomol = getattr(molec, isoname)
            if len(isomol.levels) == 0:
                continue
            taggg = tagLUTs + '_' + molec.name + '_' + isoname
            names.append(molec.name+'_'+isoname)
            LUT = LookUpTable(tag = taggg, isomolec = isomol)
            print(time.ctime())
            print("Hopefully this calculation will take about {} minutes, but actually I really don't know, take your time :)".format(LUT.CPU_time_estimate(lines, PTcouples)))
            LUT.make(spectral_grid, lines, PTcouples)
            print(time.ctime())
            LUT.export(filename = cartLUTs+taggg+date_stamp()+'.pic')
            LUTS.append(LUT)

    LUTs_wnames = dict(zip(names,LUTS))

    return LUTs_wnames
