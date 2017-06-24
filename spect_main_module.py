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
import subprocess

n_threads = 8

############ MAIN ROUTINES USED FOR SPECT_ROBOT.PY
def date_stamp():
    strin = '_'+time.ctime().split()[2]+'-'+time.ctime().split()[1]+'-'+time.ctime().split()[4]
    return strin

def equiv(num1, num2, thres = 1e-8):
    """
    Tells if two float numbers are to be considered equivalent.
    """

    if num1 == 0:
        if num2 == 0:
            return True
        else:
            return False
    if abs((num1-num2)/num1) < thres:
        return True

    return False



# CLASSES

class BayesSet(object):
    """
    Class to represent the parameters space that drive the forward model.
    """
    pass


class LookUpTable(object):
    """
    This class represent a look-up table for a specific molecule/isotope.
    """

    def __init__(self, isomolec, tag = None):
        if tag is not None:
            self.tag = tag
        else:
            self.tag = 'LUT_mol{0:02d}_iso{:1d}'.format(isomolec.mol,isomolec.iso)
        self.mol = isomolec.mol
        self.iso = isomolec.iso
        self.MM = isomolec.MM
        self.isomolec = copy.deepcopy(isomolec)
        self.sets = dict()
        return

    def make(self, spectral_grid, lines, PTcouples, export_levels = True, cartLUTs = None, control = True):
        """
        Builds the LUT for isomolec in nonLTE: one LutSet for each level, vibrational population is left outside to be added later.
        """

        self.spectral_grid = copy.deepcopy(spectral_grid)

        if cartLUTs is None:
            os.mkdir('./LUTS_'+date_stamp())

        print('Producing LUT for mol {}, iso {}. The following levels are considered: {}'.format(self.mol,self.iso,self.isomolec.levels))
        print('This calculation will take about {} Gb of disk space. Is there enough??'.format(2*len(PTcouples)*len(self.isomolec.levels)*3*len(spectral_grid.grid)*8/1.e9))

        lines = [lin for lin in lines if (lin.Mol == self.mol and lin.Iso == self.iso)]

        for lev in self.isomolec.levels:
            print('Building LutSet for level {} of mol {}, iso {}'.format(lev,self.mol,self.iso))
            filename = cartLUTs + self.tag + '_' + lev + date_stamp() + '.pic'
            set1 = LutSet(self.mol, self.iso, self.MM, level = getattr(self.isomolec, lev), filename = filename)
            self.sets[lev] = copy.deepcopy(set1)
            self.sets[lev].prepare_export(PTcouples)


        num=0
        time0 = time.time()
        for [Pres,Temp] in PTcouples:
            num+=1
            print('Calculating shapes for PTcouple {} out of {}. P = {}, T = {}'.format(num,len(PTcouples),Pres,Temp))

            time1 = time.time()
            lines_proc = spcl.calc_shapes_lines(spectral_grid, lines, Temp, Pres, self.isomolec)

            print("PTcouple {} out of {}. P = {}, T = {}. Lineshapes calculated in {:5.1f} s, time from start {:7.1f} min".format(num,len(PTcouples),Pres,Temp,time.time()-time1,(time.time()-time0)/60.))

            if control:
                comm = 'echo "PTcouple {} out of {}. P = {}, T = {}. Lineshapes calculated in {:5.1f} s, time from start {:7.1f} min" >> control_spectrobot'.format(num,len(PTcouples),Pres,Temp,time.time()-time1,(time.time()-time0)/60.)
                os.system(comm)

            time1 = time.time()

            for lev in self.isomolec.levels:
                #print('Building LutSet for level {} of mol {}, iso {}'.format(lev,self.mol,self.iso))
                self.sets[lev].add_PT(spectral_grid, lines_proc, Pres, Temp, keep_memory = False)

            mess = "Extracted single levels G_coeffs in {:5.1f} s. PT couple completed. Saving..".format(time.time()-time1)
            #print(mess)
            if control:
                comm = 'echo '+mess+' >> control_spectrobot'
                os.system(comm)

        for lev in self.isomolec.levels:
                self.sets[lev].finalize_IO()

        return


    def CPU_time_estimate(self,lines,PTcouples):
        linee_ok = [lin for lin in lines if (lin.Mol == self.mol and lin.Iso == self.iso)]

        n_lev = len(self.isomolec.levels)
        n_lin = len(linee_ok)
        n_PT = len(PTcouples)

        time = n_lin * 3./30000. * n_PT
        #print('Estimated time is about {:8.1f} min'.format(time))

        return time

    def export(self, filename):
        pickle.dump(self, open(filename,'w'))
        return



class LutSet(object):
    """
    This class represent a single entry of look-up table for a single level, for all temps/press and for all ctypes. Each ctype is an element of a dictionary.
    """

    def __init__(self, mol, iso, MM, level = None, filename = None):
        self.mol = mol
        self.iso = iso
        self.MM = MM
        if level is not None:
            self.level = copy.deepcopy(level)
            self.unidentified_lines = False
        else:
            self.level = None
            self.unidentified_lines = True
        self.filename = filename
        self.sets = []
        return

    def prepare_read(self):
        if self.filename is None:
            raise ValueError('ERROR!: NO filename set for LutSet.')
        else:
            self.temp_file = open(self.filename, 'rb')
        PTcouples = pickle.load(self.temp_file)
        return PTcouples

    def prepare_export(self, PTcouples):
        """
        Opens the pic file for export, dumps PTcouples on top.
        """
        if self.filename is None:
            raise ValueError('ERROR!: NO filename set for LutSet.')
        else:
            self.temp_file = open(self.filename, 'wb')
        pickle.dump(PTcouples, self.temp_file)

        return

    def finalize_IO(self):
        self.temp_file.close()
        self.temp_file = None
        return

    def load_from_file(self):
        """
        Loads from file just the data regarding level's LutSet. Better not to load all levels together due to memory limits.
        """
        fileo = open(self.filename,'rb')
        self.PTcouples = pickle.load(fileo)

        for PT in PTcouples:
            gigi = pickle.load(fileo)
            for pig in gigi.values():
                pig.double_precision()
                pig.restore_grid(spectral_grid)
            self.sets.append(gigi)

        fileo.close()

        return

    def load_singlePT_from_file(self, spectral_grid):
        """
        Loads from file just the data regarding level's LutSet, for a single PT. Better not to load all the LOS together due to memory limits.
        """
        if self.temp_file is None:
            self.prepare_read()

        gigi = pickle.load(self.temp_file)

        for pig in gigi.values():
            pig.double_precision()
            pig.restore_grid(spectral_grid)

        return gigi

    def free_memory(self):
        self.sets = []
        return

    def find(self, Pres, Temp):
        """
        Looks for [Pres,Temp] in PTcouples and returns the corresponding list index.
        """
        if [Pres,Temp] not in self.PTcouples:
            raise ValueError('{} couple not found!'.format([Pres,Temp]))
        else:
            ok = self.PTcouples.index([Pres,Temp])

        return ok

    def calculate(self, Pres, Temp):
        """
        Finds the closer temps and pres in PTcouples.
        """
        ctypes = ['sp_emission','ind_emission','absorption']

        Ps = np.array([PT[0] for PT in PTcouples])
        Ts = np.array([PT[1] for PT in PTcouples])

        closest_P1 = np.min(np.unique(np.abs(Ps-Pres)))
        closest_P2 = np.sort(np.unique(np.abs(Ps-Pres)))[1]

        # Now that I found the two closest Ps, I check which temps are closer to my case

        T_P1 = np.array([PT[1] for PT in PTcouples if PT[0] == closest_P1])
        T_P2 = np.array([PT[1] for PT in PTcouples if PT[0] == closest_P2])

        closest_TA = np.min(np.abs(Ts-Temp))
        closest_TB = np.sort(np.unique(np.abs(Ts-Temp)))[1]

        # I'm doing first the P interpolation
        set_ = []
        ok1 = self.find(closest_P1,closest_TA)
        coeff_ok1 = self.sets[ok1]
        ok2 = self.find(closest_P1,closest_TB)
        coeff_ok2 = self.sets[ok2]
        ok3 = self.find(closest_P2,closest_TA)
        coeff_ok3 = self.sets[ok3]
        ok4 = self.find(closest_P2,closest_TB)
        coeff_ok4 = self.sets[ok4]

        set_ = []
        for ctype in ctypes:
            coeff_ok13 = coeff_ok1[ctype].interpolate(coeff_ok3[ctype], Pres = Pres)
            coeff_ok24 = coeff_ok2[ctype].interpolate(coeff_ok4[ctype], Pres = Pres)
            set_[ctype] = coeff_ok13.interpolate(coeff_ok24, Temp = Temp)

        return set_


    def make(self, spectral_grid, lines, PTcouples, control = True):
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

        if control:
            comm = 'echo "Producing set for mol {}, iso {}, level {}. Time is {}" > control_spectrobot'.format(self.mol,self.iso,minimal_level_string,time.ctime())
            os.system(comm)
            time0 = time.time()

        for ctype in ctypes:
            #print('Producing set for '+ctype+'...')

            if ctype == 'sp_emission' or ctype == 'ind_emission':
                linee_mol = [lin for lin in lines if (minimal_level_string in lin.Up_lev_str and lin.Mol == self.mol and lin.Iso == self.iso)]
            elif ctype == 'absorption':
                linee_mol = [lin for lin in lines if (minimal_level_string in lin.Lo_lev_str and lin.Mol == self.mol and lin.Iso == self.iso)]

            if len(linee_mol) == 0:
                #print('NO lines in {} for level {}'.format(ctype,minimal_level_string))
                continue

            if control:
                comm = 'echo "ctype is {}. {} lines found." >> control_spectrobot'.format(ctype,len(linee_mol))
                os.system(comm)

            num = 0
            for [P,T] in PTcouples:
                #print('PTcouple: {}, {}'.format(P,T))
                num += 1
                time1 = time.time()

                gigi = spcl.SpectralGcoeff(ctype, spectral_grid, self.mol, self.iso, self.MM, minimal_level_string)
                gigi.BuildCoeff(lines, T, P)
                set_[ctype].append(gigi)
                if control:
                    comm = 'echo "PTcouple {} out of {}. P = {}, T = {}. PT done in {:5.1f} s, time from start {:7.1f} min" >> control_spectrobot'.format(num,len(PTcouples),P,T, time.time()-time1,(time.time()-time0)/60.)
                    os.system(comm)
                #print('Added')

        return


    def add_PT(self, spectral_grid, lines, Pres, Temp, keep_memory = False, control = True):
        """
        Adds a single PT couple to the set. If keep_memory is set to True, the resulting G coeffs are stored in the set, instead are just dumped in the pickle file.
        """

        ctypes = ['sp_emission','ind_emission','absorption']
        set_ = dict()

        if not self.unidentified_lines:
            minimal_level_string = self.level.minimal_level_string()
        else:
            minimal_level_string = ''

        for ctype in ctypes:
            #print('Producing set for '+ctype+'...')
            gigi = spcl.SpectralGcoeff(ctype, spectral_grid, self.mol, self.iso, self.MM, minimal_level_string, unidentified_lines = self.unidentified_lines)
            gigi.BuildCoeff(lines, Temp, Pres, preCalc_shapes = True)

            #print(np.max(gigi.spectrum))

            gigi.erase_grid()
            #gigi.half_precision()

            # if ctype == 'absorption':
            #     print(gigi.spectrum)
            #     print(np.max(gigi.spectrum))
            #     pl.ion()
            #     pl.figure(17)
            #     pl.plot(spectral_grid.grid, gigi.spectrum)
            #     pl.show()

            set_[ctype] = copy.deepcopy(gigi)
            #print('Added')

        #print('dampoooooooooooooooooooooooo')
        pickle.dump(set_, self.temp_file)

        if not keep_memory:
            del set_
        else:
            self.sets.append(set_)

        return

    def export(self, filename):
        pickle.dump(self, open(filename,'w'))
        return

    def add_dump(self, set_):
        pickle.dump(set_, self.temp_file)
        return


# FUNCTIONS


def prepare_spe_grid(wn_range, sp_step = 5.e-4, units = 'cm_1'):
    """
    Prepares the SpectralObject for the range considered.
    """

    spoffo = np.arange(wn_range[0],wn_range[1]+sp_step/2,sp_step,dtype = float)
    spect_grid = spcl.SpectralGrid(spoffo, units = units)
    spoffo = np.zeros(len(spect_grid.grid), dtype = float)
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

    #print('Finito spettro con {} linee in {} s!'.format(len(linee_tot), time.time()-time0))

    return abs_coeff


def do_for_th(linee_tot, abs_coeff, i, coda):
    step_nlin = len(linee_tot)/n_threads
    linee = linee_tot[step_nlin*i:step_nlin*(i+1)]
    if i == n_threads-1:
        linee = linee_tot[step_nlin*i:]

    #print('Hey! Questo è il ciclo {} con {} linee su {}!'.format(i,len(linee),len(linee_tot)))

    shapes, lws, dws = smm.abs_coeff_calc_LTE(linee, abs_coeff, 6, 1, 16., 296., 1000.)

    #print('Ciclo {} concluso in {} s!'.format(i,time.time()-time0))

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
    #print('Inside spect_calc.. reading lines and calculating linshapes')

    lin_grid = np.arange(-imxsig*sp_step/2,imxsig*sp_step/2,sp_step, dtype = float)

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

    #print('Made {} lines in {} s'.format(len(linee_mol),time.time()-time0))

    return shapes, lws, dws


def emiss_coeff_calc_LTE(linee_mol, abs_coeff, mol, iso, MM, Temp, Pres, LTE = True, sp_step = 5.e-4, max_lines = 0, imxsig = 13010, fraction_to_keep = None):
    pass


def read_Gcoeffs_from_LUTs(cartLUTs, fileLUTs):
    """
    Reads the LookUpTables produced by makeLUT_nonLTE_Gcoeffs. Doesn't load all data, just loads the level structure and the filenames of the level LutSets.
    """

    LUTs = pickle.load(open(cartLUTs+fileLUTs,'r'))

    return LUTs


def makeLUT_nonLTE_Gcoeffs(spectral_grid, lines, molecs, atmosphere, cartLUTs = None, tagLUTs = 'LUT_', n_pres_levels = None, pres_step_log = 0.1, temp_step = 5.0, save_LUTs = True, n_threads = n_threads, test = False, thres = 0.01):
    """
    Calculates the G_coeffs for the isomolec_levels at Temp and Pres.
    :param isomolecs: A list of isomolecs objects or a single one.
    """

    # Define pressure levels, for each pres check which temps are needed for the full atm and define a set of temperatures to be used at that pressure. Build something to be given as input to make LUTs.
    logpres0 = np.log(np.max(atmosphere.pres))
    logpresMIN = np.log(np.min(atmosphere.pres))
    log_pressures = logpresMIN + np.arange(0,(logpres0-logpresMIN)+0.5*pres_step_log,pres_step_log)
    #print('Built set of {} pressure levels from {} to {} with logstep = {}.'.format(len(log_pressures),logpresMIN,logpres0,pres_step_log))

    airbr = np.argmax(np.array([lin.Air_broad for lin in lines]))

    pressures = np.exp(log_pressures)

    n_dim = len(atmosphere.grid)

    temps = []
    #### QUI c'è un problema se il passo di pressures è più fitto di quello di atmosphere.pres.............. DA RISOLVERE!
    okke = (atmosphere.pres <= pressures[1])
    #print(np.any(okke))
    temps_pres = atmosphere.temp[okke]
    temps.append([np.min(temps_pres),np.max(temps_pres)])
    #print('For level {} with pres {}, temp in range: {} <-> {}'.format(len(temps)-1,pressures[0],temps[-1][0],temps[-1][1]))

    i = 0
    for pres0, pres1, pres2 in zip(pressures[:-2], pressures[1:-1], pressures[2:]):
        # Find the corresponding Temps.
        okke = (atmosphere.pres >= pres0) & (atmosphere.pres <= pres2)
        #print(i,np.any(okke))
        temps_pres = atmosphere.temp[okke]
        temps.append([np.min(temps_pres),np.max(temps_pres)])
        #print('For level {} with pres {}, temp in range: {} <-> {}'.format(len(temps)-1,pres1,temps[-1][0],temps[-1][1]))
        i+=1

    okke = (atmosphere.pres >= pressures[-2])
    #print(np.any(okke))
    temps_pres = atmosphere.temp[okke]
    temps.append([np.min(temps_pres),np.max(temps_pres)])
    #print('For level {} with pres {} hPa, temp is in range: {} <-> {}'.format(len(temps)-1,pressures[-1],temps[-1][0],temps[-1][1]))

    # round_values for temp at 5 K steps and build couples P/T
    PTcouples = []
    for pres, trange in zip(pressures,temps):
        t_0 = int(trange[0]) / int(temp_step)
        t_1 = int(trange[1]+temp_step) / int(temp_step)
        all_t = np.arange(t_0*temp_step,(t_1+1.)*temp_step,temp_step)
        ##print(trange,all_t)
        for temp in all_t:
            PTcouples.append([pres,temp])

    mms = []
    for mol in molecs:
        mms += [getattr(mol,isom).MM for isom in mol.all_iso]

    PTcouples_ok = []
    temps_lowpres = []
    for [Pres, Temp] in PTcouples:
        dw, lw, wsh = lines[airbr].CheckWidths(Temp, Pres, min(mms))
        if lw < thres*dw:
            print('Skippo pressure level: {} << {}'.format(lw,dw))
            if Temp not in temps_lowpres:
                temps_lowpres.append(Temp)
        else:
            PTcouples_ok.append([Pres, Temp])

    Pres_0 = 1.e-8
    for Temp in temps_lowpres:
        PTcouples_ok.insert(0, [Pres_0, Temp])

    PTcouples = PTcouples_ok

    if test:
        print('Keeping ONLY 10 PTcouples for testing')
        PTcouples = PTcouples[:10]

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
            #print(time.ctime())
            print("Hopefully this calculation will take about {} minutes, but actually I really don't know, take your time :)".format(LUT.CPU_time_estimate(lines, PTcouples)))
            LUT.make(spectral_grid, lines, PTcouples, export_levels = True, cartLUTs = cartLUTs)
            print(time.ctime())
            LUT.export(filename = cartLUTs+taggg+date_stamp()+'.pic')
            LUTS.append(LUT)

    LUTs_wnames = dict(zip(names,LUTS))

    return LUTs_wnames


def make_abscoeff_isomolec(wn_range, isomolec, Temps, Press, LTE = True, fileLUTs = None, cartLUTs = None, useLUTs = False, lines = None):
    """
    Builds the absorption and emission coefficients for isomolec, both in LTE and non-LTE. If in non-LTE, isomolec levels have to contain the attribute local_vibtemp, produced by calling level.add_local_vibtemp(). If LTE is set to True, LTE populations are used.
    LUT is the object created by makeLUT_nonLTE_Gcoeffs(). Contains
    """

    try:
        len(Press)
        len(Temps)
    except:
        Press = [Press]
        Temps = [Temps]

    #print('Sto entrandooooooooooooo, mol {}, iso {}'.format(isomolec.mol, isomolec.iso))
    abs_coeff = prepare_spe_grid(wn_range)
    spectral_grid = abs_coeff.spectral_grid

    unidentified_lines = False
    if len(isomolec.levels) == 0:
        unidentified_lines = True
    #    print('acazuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu')

    set_tot = dict()
    if not unidentified_lines:
        for lev in isomolec.levels:
            levvo = getattr(isomolec, lev)
            strin = cartLUTs+'LUTLOS_mol_{}_iso_{}_{}.pic'.format(isomolec.mol, isomolec.iso, lev)
            set_tot[lev] = LutSet(isomolec.mol, isomolec.iso, isomolec.MM, level = levvo, filename = strin)
            set_tot[lev].prepare_export([zui for zui in zip(Press,Temps)])
    else:
        #print('siamo quaaaA')
        strin = cartLUTs+'LUTLOS_mol_{}_iso_{}_alllev.pic'.format(isomolec.mol, isomolec.iso)
        set_tot['all'] = LutSet(isomolec.mol, isomolec.iso, isomolec.MM, level = None, filename = strin)
        set_tot['all'].prepare_export([zui for zui in zip(Press,Temps)])


    if not useLUTs:
        if lines is None:
            raise ValueError('when calling smm.make_abscoeff_isomolec() with useLUTs = False, you need to give the list of spectral lines of isomolec as input')
        else:
            #print(len(lines))
            lines = [lin for lin in lines if lin.Mol == isomolec.mol and lin.Iso == isomolec.iso]
            #print(len(lines))

        ctypes = ['sp_emission','ind_emission','absorption']

        numh = 0
        for Pres, Temp in zip(Press,Temps):
            numh+=1
            #print('Siamo a step {} catuuuullooooooo'.format(numh))
            #print(isomolec.levels)
            lines_proc = spcl.calc_shapes_lines(spectral_grid, lines, Temp, Pres, isomolec)
            #print(len(lines_proc))
            if not unidentified_lines:
                for lev in isomolec.levels:
                    #print('Siamo a mol {}, iso {}, lev {} bauuuuuuuuu'.format(isomolec.mol, isomolec.iso, lev))
                    levello = getattr(isomolec, lev)
                    set_tot[lev].add_PT(spectral_grid, lines_proc, Pres, Temp)
            else:
                #print('Siamo a mol {}, iso {}, all_levssss miaoooooooooooo'.format(isomolec.mol, isomolec.iso))
                set_tot['all'].add_PT(spectral_grid, lines_proc, Pres, Temp)

    else:
        #read Gcoeffs from LUTS and attach to levels
        LUTs = read_Gcoeffs_from_LUTs(cartLUTs, fileLUTs)
        for lev in isomolec.levels:
            levello = getattr(isomolec, lev)
            #print('Loading LutSet for level {}...'.format(lev))
            LUTs.sets[lev].load_from_file()
            for Pres, Temp in zip(Press,Temps):
                set_ = LUTs.sets[lev].calculate(Pres, Temp)
                set_tot[lev].add_dump(set_)
            LUTs.sets[lev].free_memory()
            #print('Added')

    for val in set_tot.values():
        val.finalize_IO()
        #print('Finalizzzooooooooooo')

    abs_coeffs = []
    emi_coeffs = []

    for nam, val in zip(set_tot.keys(), set_tot.values()):
        val.prepare_read()
        #print('Reading... -> '+nam)

    for num in range(len(Temps)):
        #print('oyeeeeeeeeeee ', num)
        abs_coeff = prepare_spe_grid(wn_range)
        emi_coeff = prepare_spe_grid(wn_range)
            # THIS IS WITH LTE PARTITION FUNCTION!!
        Q_part = spcl.CalcPartitionSum(isomolec.mol, isomolec.iso, temp = Temps[num])

        if unidentified_lines:
            Gco = set_tot['all'].load_singlePT_from_file(spectral_grid)
            pop = 1 / Q_part
            abs_coeff.add_to_spectrum(Gco['absorption'], Strength = pop)
            abs_coeff.add_to_spectrum(Gco['ind_emission'], Strength = -pop)
            emi_coeff.add_to_spectrum(Gco['sp_emission'], Strength = pop)
        else:
            for lev in isomolec.levels:
                levello = getattr(isomolec, lev)
                if LTE:
                    vibt = Temps[num]
                else:
                    vibt = levello.local_vibtemp[num]
                #Gco = levello.Gcoeffs[num]
                Gco = set_tot[lev].load_singlePT_from_file(spectral_grid)
                pop = spcl.Boltz_ratio_nodeg(levello.energy, vibt) / Q_part
                abs_coeff.add_to_spectrum(Gco['absorption'], Strength = pop)
                abs_coeff.add_to_spectrum(Gco['ind_emission'], Strength = -pop)
                emi_coeff.add_to_spectrum(Gco['sp_emission'], Strength = pop)

        abs_coeffs.append(abs_coeff)
        emi_coeffs.append(emi_coeff)

    return abs_coeffs, emi_coeffs


def read_obs(filename, formato = 'gbb'):
    if formato == 'gbb':
        outs = sbm.read_obs(filename)
        spectra = outs[-2].T
        flags = outs[-1].T
        wn_arr = outs[-3]
        gri = spcl.SpectralGrid(wn_arr, units = 'nm')
        obss = []
        for col,zol in zip(spectra, flags):
            spet = spcl.SpectralIntensity(col, gri, units = 'Wm2')
            spet.add_mask(zol)
            obss.append(spet)
    else:
        raise ValueError('formato {} not available'.format(formato))

    return obss


def read_orbits(filename, formato = 'VIMSselect', tag = None):
        orbits = []
        if formato == 'VIMSselect':
            infile = open(filename,'r')
            sbm.find_spip(infile)
            linee = infile.readlines()
            for lin in linee:
                nomi = 'num dist sub_obs_lat sub_obs_lon limb_tg_alt limb_tg_lat limb_tg_lon limb_tg_sza pixel_rot phase_ang sub_solar_lat sub_solar_lon sun_dist time'
                nomi = nomi.split()
                cose = map(float,lin.split())
                orb = dict(zip(nomi,cose))
                orb['num'] = int(cose[0])
                orb['filename'] = filename
                orb['tag'] = tag
                orbits.append(orb)
            infile.close()
        else:
            raise ValueError('formato {} not available'.format(formato))

        return orbits


def read_input_observed(observed_cart):
    """
    Reads inputs regarding observations. The folder observed_cart has to contain an "orbit_***.dat", an "observ_***.dat". If more files are found inside, all observations are read.
    """

    lss = subprocess.check_output(['ls', observed_cart])
    #files_orbit = [fil for fil in lss.split('\n') if 'orbit_' in fil]
    tag_observ = [fil[7:] for fil in lss.split('\n') if 'observ_' in fil]

    set_pixels = []
    obs_tot = []
    orbit_tot = []

    for tag in tag_observ:
        orbit_tot += read_orbits(observed_cart+'orbit_'+tag, formato = 'VIMSselect', tag = tag)
        obs_tot += read_obs(observed_cart+'observ_'+tag, formato = 'gbb')

    for ob, orb in zip(obs_tot, orbit_tot):
        orb['observation'] = copy.deepcopy(ob)
        pix = sbm.Pixel(orb.keys(), orb.values())
        set_pixels.append(pix)

    return set_pixels
