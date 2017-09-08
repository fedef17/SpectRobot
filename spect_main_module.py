#!/usr/bin/python
# -*- coding: utf-8 -*-

import decimal
import numpy as np
from numpy.linalg import inv
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
#import pickle
import cPickle as pickle
import lineshape
from multiprocessing import Process, Queue
import copy
import subprocess
from scipy.interpolate import RectBivariateSpline as spline2D

n_threads = 8

############ MAIN ROUTINES USED FOR SPECT_ROBOT.PY
def date_stamp():
    strin = '_'+time.ctime().split()[2]+'-'+time.ctime().split()[1]+'-'+time.ctime().split()[4]
    return strin

def find_free_name(filename, maxnum = 1000, split_at = '.'):
    if maxnum <= 100:
        form = '_{:02d}'
    elif maxnum <= 1000:
        form = '_{:03d}'
    else:
        form = '_{:05d}'

    num = 1
    fileorig = filename
    ind = fileorig.index(split_at)
    while os.path.isfile(filename):
        filename = fileorig[:ind]+form.format(num)+fileorig[ind:]
        num +=1
        if num > maxnum:
            raise ValueError('Check filenames! More than {} with the same name'.format(maxnum))

    return filename

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


def check_lines_mols(lines, molecs):
    """
    Returns just the lines that involve molecs and molecs levels if set.
    """

    lines_ok = []
    for mol in molecs:
        for iso in mol.all_iso:
            isomol = getattr(mol, iso)
            if len(isomol.levels) > 0:
                lines_ok += [lin for lin in lines if lin.Mol == isomol.mol and lin.Iso == isomol.iso and isomol.has_level(lin.Lo_lev_str)[0] and isomol.has_level(lin.Up_lev_str)[0]]
            else:
                lines_ok += [lin for lin in lines if lin.Mol == isomol.mol and lin.Iso == isomol.iso]

    return lines_ok



# CLASSES

class BayesSet(object):
    """
    Class to represent the FULL parameters space that drive the forward model. One BayesSet may contain more RetSets and other single RetParams. Each RetSet contains more RetParams (e.g. a VMR profile contains more single values)
    """
    def __init__(self, tag = None):
        self.tag = tag
        self.sets = dict()
        self.n_tot = 0
        self.order = []
        self.old_params = []
        return

    def add_set(self, set_):
        self.sets[set_.name] = copy.deepcopy(set_)
        self.n_tot += set_.n_par
        self.order.append(set_.name)
        return

    def values(self):
        return [par.value for par in self.params()]

    def params(self):
        pars = []
        for cos in self.order:
            for par in self.sets[cos].set:
                pars.append(par)
        return pars

    def build_jacobian(self, masks = None):
        jac = []
        if masks is not None:
            masktot = []
            for mas in masks:
                masktot += list(mas)
            masktot = np.array(masktot, dtype = bool)

        for nam in self.order:
            for par in self.sets[nam].set:
                dertot = []
                for der in par.derivatives:
                    dertot += list(der.spectrum)
                if masks is None:
                    jac.append(dertot)
                else:
                    dertot = np.array(dertot)
                    jac.append(dertot[masktot])

        jac = np.array(jac)
        jac = jac.T # n x m: n is len(y), m is self.n_tot
        self.jacobian = jac
        return jac

    def update_params(self, delta_x):
        self.old_params.append([par.value for par in self.params()])
        for par, val in zip(self.params(), delta_x):
            par.update_par(val)
        return

    def VCM_apriori(self):
        S_ap = np.identity(self.n_tot)
        for i, aper in zip(range(self.n_tot), [par.apriori_err for par in self.params()]):
            S_ap[i,i] = aper**2
        return S_ap

    def apriori_vector(self):
        return np.array([par.apriori for par in self.params()])

    def param_vector(self):
        return np.array([par.value for par in self.params()])

    def store_avk(self, av_kernel):
        self.av_kernel = copy.deepcopy(av_kernel)
        return

    def store_VCM(self, VCM):
        self.VCM = copy.deepcopy(VCM)
        return


class RetSet(object):
    """
    A set of parameters referring to the same quantity (a VMR profile, lat distributions, a Temp profile..). Contains more RetParams.
    """
    def __init__(self, name, params):
        self.name = name
        self.set = []
        self.n_par = len(params)
        for par in params:
            self.set.append(copy.deepcopy(par))
        return

    # def update_params(self, new_values):
    #     for par, new in zip(self.set, new_values):
    #         par.update_par(new)
    #     return

    def items(self):
        return zip([par.key for par in self.set], self.set)

    def keys(self):
        return [par.key for par in self.set]

# class RetSet_vmr(RetSet):
#     """
#     RetSet for vmr.
#     """
#     def __init__(self, gas_name, params):
#         RetSet.__init__(self, gas_name, params)
#         return
#
#
# class RetSet_temp(RetSet):
#     """
#     RetSet for temp.
#     """
#     def __init__(self, params):
#         RetSet.__init__(self, 'temp', params)
#         return
#
#
# class RetSet_isoab(RetSet):
#     """
#     RetSet for iso abundance.
#     """
#     def __init__(self, gas_name, iso_num, params):
#         RetSet.__init__(self, gas_name, params)
#         self.iso = iso_num
#         return


def alt_triangle(alt_grid, node_alt, step = None, node_lo = None, node_up = None, first = False, last = False):
    if step is not None:
        node_lo = node_alt - step
        node_up = node_alt + step
    cos = np.zeros(len(alt_grid), dtype = float)
    for alt, ii in zip(alt_grid, range(len(alt_grid))):
        if first:
            if alt < node_alt:
                cos[ii] = 1.0
            elif alt < node_up:
                cos[ii] = 1.0-(alt-node_alt)/(node_up-node_alt)
            else:
                cos[ii] = 0.0
        elif last:
            if alt > node_alt:
                cos[ii] = 1.0
            elif alt > node_lo:
                cos[ii] = 1.0-(node_alt-alt)/(node_alt-node_lo)
            else:
                cos[ii] = 0.0
        else:
            if alt < node_lo or alt > node_up:
                cos[ii] = 0.0
            elif alt >= node_alt:
                cos[ii] = 1.0-(alt-node_alt)/(node_up-node_alt)
            else:
                cos[ii] = 1.0-(node_alt-alt)/(node_alt-node_lo)

    # pl.plot(cos, alt_grid, label = '{}'.format(node_alt))
    alt_grid = sbm.AtmGrid('alt', alt_grid)
    #cos = sbm.AtmProfile(alt_grid, cos, 'mask', 'lin')
    cos = sbm.AtmGridMask(alt_grid, cos, 'lin')

    return cos


def lat_box(lat_limits, lat_ok):
    """
    Creates a lat mask grid with a box function. boxes are centred at the mean lat, lat_ok can be any lat contained in the box.
    ATTENZIONEEEE: lat_limits sono i punti di inizio dei box lat, andando da nord a sud. Per box lat distribuiti 10 gradi ogni dieci da polo a polo, lat_limits = [-90,-80,...,70,80]. certamente bisogna pensarsi qualcosa di meglio.
    """
    lat_grid = sbm.AtmGrid('lat', lat_limits)
    cos = []
    for lat1, lat2 in zip(lat_limits[:-1], lat_limits[1:]):
        if lat_ok >= lat1 and lat_ok < lat2:
            cos.append(1.0)
        else:
            cos.append(0.0)

    if lat_ok > lat_limits[-1]:
        cos.append(1.0)
    else:
        cos.append(0.0)

    cos = sbm.AtmGridMask(lat_grid, np.array(cos), 'box')

    return cos


def centre_boxes(lat_limits):
    """
    from the array of lat_limits [-90, -60, -30, 30, ..] to the array of box centres [-75, -45, 0, ..]. The box centres are used as grid values.
    """
    lat_centres = []
    for la1, la2 in zip(lat_limits[:-1], lat_limits[1:]):
        lat_centres.append((la1+la2)/2.0)

    return lat_centres


class LinearProfile_2D(RetSet):
    """
    A profile constructed through linear interpolation of a set of params, with horizontal (i.e. latitudinal) boxes.
    """

    def __init__(self, name, atmosphere, alt_nodes, lat_limits, apriori_profs, apriori_prof_errs, first_guess_profs = None):
        self.name = name
        self.set = []
        self.n_par = len(alt_nodes)*len(lat_limits)

        alt_grid = sbm.AtmGrid('alt', atmosphere.grid.coords['alt'])

        if first_guess_profs is None:
            first_guess_profs = apriori_profs

        for apriori_prof, apriori_prof_err, first_guess_prof, lat in zip(apriori_profs, apriori_prof_errs, first_guess_profs, lat_limits):
            coso = LinearProfile_1D_new(name, alt_grid, alt_nodes, apriori_prof, apriori_prof_err)

            latbox = lat_box(lat_limits, lat)
            for cos in coso.set:
                numask = cos.maskgrid.merge(latbox)
                par = RetParam(name, [lat, cos.key], numask, cos.apriori, cos.apriori_err)
                self.set.append(copy.deepcopy(par))

        return


    def profile(self):
        """
        Calculates the profile summing on the parameters.
        """
        prof = sbm.AtmProfZeros(self.set[0].maskgrid.grid, self.name, self.set[0].maskgrid.interp.values()[0])
        for par in self.set:
            prof += par.maskgrid*par.value

        return prof


class LinearProfile_1D_new(RetSet):
    """
    A profile constructed through linear interpolation of a set of params.
    """

    def __init__(self, name, alt_grid, alt_nodes, apriori_prof, apriori_prof_err, first_guess_prof = None):
        self.name = name
        self.set = []
        self.n_par = len(alt_nodes)
        if first_guess_prof is None:
            first_guess_prof = apriori_prof

        # First point
        maskgrid = alt_triangle(alt_grid.grid[0], alt_nodes[0], node_up = alt_nodes[1], first = True)
        par = RetParam(name, alt_nodes[0], maskgrid, apriori_prof[0], apriori_prof_err[0], first_guess = first_guess_prof[0])
        self.set.append(copy.deepcopy(par))

        # Points in the middle
        for alo, alt, aup, ap, fi, er in zip(alt_nodes[:-2], alt_nodes[1:-1], alt_nodes[2:], apriori_prof, first_guess_prof, apriori_prof_err):
            maskgrid = alt_triangle(alt_grid.grid[0], alt, node_up = aup, node_lo = alo)
            par = RetParam(name, alt, maskgrid, ap, er, first_guess = fi)
            self.set.append(copy.deepcopy(par))

        # Last point
        maskgrid = alt_triangle(alt_grid.grid[0], alt_nodes[-1], node_lo = alt_nodes[-2], last = True)
        par = RetParam(name, alt_nodes[-1], maskgrid, apriori_prof[-1], apriori_prof_err[-1], first_guess = first_guess_prof[-1])
        self.set.append(copy.deepcopy(par))

        return

    def profile(self):
        """
        Calculates the profile summing on the parameters.
        """
        prof = sbm.AtmProfZeros(self.set[0].maskgrid.grid, self.name, self.set[0].maskgrid.interp.values()[0])
        for par in self.set:
            prof += par.maskgrid*par.value

        return prof


class LinearProfile_1D(RetSet):
    """
    A profile constructed through linear interpolation of a set of params.
    """

    def __init__(self, name, atmosphere, alt_nodes, apriori_prof, apriori_prof_err, first_guess_prof = None):
        self.name = name
        self.set = []
        self.n_par = len(alt_nodes)
        if first_guess_prof is None:
            first_guess_prof = apriori_prof

        # First point
        maskgrid = alt_triangle(atmosphere.grid.grid[0], alt_nodes[0], node_up = alt_nodes[1], first = True)
        par = RetParam(name, alt_nodes[0], maskgrid, apriori_prof[0], apriori_prof_err[0], first_guess = first_guess_prof[0])
        self.set.append(copy.deepcopy(par))

        # Points in the middle
        for alo, alt, aup, ap, fi, er in zip(alt_nodes[:-2], alt_nodes[1:-1], alt_nodes[2:], apriori_prof, first_guess_prof, apriori_prof_err):
            maskgrid = alt_triangle(atmosphere.grid.grid[0], alt, node_up = aup, node_lo = alo)
            par = RetParam(name, alt, maskgrid, ap, er, first_guess = fi)
            self.set.append(copy.deepcopy(par))

        # Last point
        maskgrid = alt_triangle(atmosphere.grid.grid[0], alt_nodes[-1], node_lo = alt_nodes[-2], last = True)
        par = RetParam(name, alt_nodes[-1], maskgrid, apriori_prof[-1], apriori_prof_err[-1], first_guess = first_guess_prof[-1])
        self.set.append(copy.deepcopy(par))

        self.orig_atmosphere = atmosphere

        return


    def profile(self):
        """
        Calculates the profile summing on the parameters.
        """
        prof = sbm.AtmProfZeros(self.set[0].maskgrid.grid, self.name, self.set[0].maskgrid.interp.values()[0])
        for par in self.set:
            prof += par.maskgrid*par.value

        return prof



class RetParam(object):
    """
    A single parameter in the parameter space.
    """

    def __init__(self, nameset, key, maskgrid, apriori, apriori_err, first_guess = None, constrain_positive = True):
        self.nameset = nameset
        self.key = key
        self.maskgrid = copy.deepcopy(maskgrid)
        if first_guess is None:
            first_guess = apriori
        self.value = first_guess
        self.apriori = apriori
        self.apriori_err = apriori_err
        self.derivatives = []
        self.old_values = []
        self.constrain_positive = constrain_positive
        return

    def update_par(self, delta_par):
        self.old_values.append(self.value)
        new_value = self.value+delta_par
        if self.constrain_positive:
            while new_value <= 0.0:
                delta_par /= 2
                new_value = self.value+delta_par
        self.value = new_value
        return

    def add_hires_deriv(self, derivative):
        self.hires_deriv = copy.deepcopy(derivative)
        return

    def store_deriv(self, derivative, num):
        try:
            self.derivatives[num] = copy.deepcopy(derivative)
        except:
            self.derivatives.append(copy.deepcopy(derivative))
        return


def lut_name(mol, iso, LTE):
    if LTE:
        name = 'LUT_mol{:02d}_iso{:1d}_LTE'.format(mol,iso)
    else:
        name = 'LUT_mol{:02d}_iso{:1d}_nonLTE'.format(mol,iso)
    return name


class LookUpTable(object):
    """
    This class represent a look-up table for a specific molecule/isotope.
    """

    def __init__(self, isomolec, wn_range, LTE):
        self.tag = lut_name(isomolec.mol, isomolec.iso, LTE)
        self.wn_range = copy.deepcopy(wn_range)
        self.mol = isomolec.mol
        self.iso = isomolec.iso
        self.MM = isomolec.MM
        self.isomolec = copy.deepcopy(isomolec)
        self.sets = dict()
        self.PTcouples = []
        self.LTE = LTE
        return

    def merge(self, LUT):
        """
        To make a single lut from two luts with different ptcouples but same levels.
        """
        for lev_name in self.sets.keys():
            lev1 = self.sets[lev_name]
            lev2 = LUT.sets[lev_name]
            if not self.LTE:
                if not lev1.level.equiv(lev2.level.lev_string):
                    raise ValueError('Levels are different, cannot merge LUTs')
            lev1.add_file(lev2.filename, lev2.PTcouples)

        self.PTcouples += LUT.PTcouples
        return

    def make(self, spectral_grid, lines, PTcouples, export_levels = True, cartLUTs = None, control = True):
        """
        Builds the LUT for isomolec in nonLTE: one LutSet for each level, vibrational population is left outside to be added later.
        """

        self.PTcouples = copy.deepcopy(PTcouples)
        self.spectral_grid = copy.deepcopy(spectral_grid)

        if cartLUTs is None:
            os.mkdir('./LUTS_'+date_stamp())

        print('Producing LUT for mol {}, iso {}. The following levels are considered: {}'.format(self.mol,self.iso,self.isomolec.levels))
        print('This calculation will take about {} Gb of disk space. Is there enough??'.format(2*len(PTcouples)*len(self.isomolec.levels)*3*len(spectral_grid.grid)*8/1.e9))

        lines = [lin for lin in lines if (lin.Mol == self.mol and lin.Iso == self.iso)]

        if not self.LTE:
            for lev in self.isomolec.levels:
                print('Building LutSet for level {} of mol {}, iso {}'.format(lev,self.mol,self.iso))
                filename = cartLUTs + self.tag + '_' + lev + date_stamp() + '.pic'
                filename = find_free_name(filename, maxnum = 10)
                set1 = LutSet(self.mol, self.iso, self.MM, level = getattr(self.isomolec, lev), filename = filename)
                self.sets[lev] = copy.deepcopy(set1)
                self.sets[lev].prepare_export(PTcouples, self.spectral_grid)
        else:
            print('Building LTE LutSet for mol {}, iso {}'.format(self.mol,self.iso))
            filename = cartLUTs + self.tag + '_alllev' + date_stamp() + '.pic'
            filename = find_free_name(filename, maxnum = 10)
            set1 = LutSet(self.mol, self.iso, self.MM, level = None, filename = filename)
            self.sets['all'] = copy.deepcopy(set1)
            self.sets['all'].prepare_export(PTcouples, self.spectral_grid)


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

            if not self.LTE:
                for lev in self.isomolec.levels:
                    #print('Building LutSet for level {} of mol {}, iso {}'.format(lev,self.mol,self.iso))
                    self.sets[lev].add_PT(spectral_grid, lines_proc, Pres, Temp, keep_memory = False)
            else:
                self.sets['all'].add_PT(spectral_grid, lines_proc, Pres, Temp, keep_memory = False)

            mess = "Extracted single levels G_coeffs in {:5.1f} s. PT couple completed. Saving..".format(time.time()-time1)
            #print(mess)
            if control:
                comm = 'echo '+mess+' >> control_spectrobot'
                os.system(comm)

        if not self.LTE:
            for lev in self.isomolec.levels:
                self.sets[lev].finalize_IO()
        else:
            self.sets['all'].finalize_IO()

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
        pickle.dump(self, open(filename,'w'), protocol = -1)
        return

    def find_lev(self, lev_string):
        for lev in self.sets.keys():
            if self.sets[lev].level.equiv(lev_string):
                ok = True
                return ok, lev

        return False, None



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
        self.filename = filename # Actual file in use
        self.filenames = [filename] # All files that refer to the same lutset. Used in case of incomplete PTcouples.
        self.sets = []
        self.spectral_grid = None
        self.PTcouples = None
        return

    def add_file(self, filename, PTcouples):
        """
        To complete the set, this is to allow reading from files made at different times.
        """
        self.filenames.append(filename)
        self.PTcouples += PTcouples
        return

    def prepare_read(self):
        if self.filename is None:
            raise ValueError('ERROR!: NO filename set for LutSet.')
        else:
            self.temp_file = open(self.filename, 'rb')
        PTcouples = pickle.load(self.temp_file)
        return PTcouples

    def prepare_export(self, PTcouples, spectral_grid):
        """
        Opens the pic file for export, dumps PTcouples on top.
        """
        if self.filename is None:
            raise ValueError('ERROR!: NO filename set for LutSet.')
        else:
            self.temp_file = open(self.filename, 'wb')
        self.PTcouples = copy.deepcopy(PTcouples)
        self.spectral_grid = spectral_grid
        pickle.dump(PTcouples, self.temp_file, protocol = -1)

        return

    def finalize_IO(self):
        self.temp_file.close()
        self.temp_file = None
        return

    def load_from_file(self, load_just_PT = False, spectral_grid = None):
        """
        Loads from file just the data regarding level's LutSet. Better not to load all levels together due to memory limits.
        """
        fileo = open(self.filename,'rb')
        self.PTcouples = pickle.load(fileo)

        if load_just_PT:
            fileo.close()
            return

        for PT in self.PTcouples:
            gigi = pickle.load(fileo)
            for pig in gigi.values():
                pig.double_precision()
                try:
                    pig.restore_grid(self.spectral_grid)
                except: # for compatibility
                    pig.restore_grid(spectral_grid)
            self.sets.append(gigi)

        fileo.close()

        return

    def load_from_files(self, load_just_PT = False, spectral_grid = None):
        """
        Loads from file just the data regarding level's LutSet. Better not to load all levels together due to memory limits.
        Adapted for more than one file.
        """

        self.PTcouples = []
        for filename in self.filenames:
            fileo = open(filename,'rb')
            self.PTcouples += pickle.load(fileo)

            if load_just_PT:
                fileo.close()
            else:
                for PT in self.PTcouples:
                    gigi = pickle.load(fileo)
                    for pig in gigi.values():
                        pig.double_precision()
                        try:
                            pig.restore_grid(self.spectral_grid)
                        except: # for compatibility
                            pig.restore_grid(spectral_grid)
                            if spectral_grid is None:
                                raise ValueError('No spectral grid given.')
                    self.sets.append(gigi)

                fileo.close()

        return

    def load_singlePT_from_file(self, spectral_grid = None):
        """
        Loads from file just the data regarding level's LutSet, for a single PT. Better not to load all the LOS together due to memory limits.
        """
        if self.temp_file is None:
            self.prepare_read()

        gigi = pickle.load(self.temp_file)

        for pig in gigi.values():
            pig.double_precision()
            try:
                pig.restore_grid(self.spectral_grid)
            except Exception as cazzillo: # for compatibility
                if spectral_grid is None:
                    raise cazzillo
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
            print(self.PTcouples)
            raise ValueError('{} couple not found!'.format([Pres,Temp]))
        else:
            ok = self.PTcouples.index([Pres,Temp])

        return ok

    def calculate(self, Pres, Temp):
        """
        Finds the closer temps and pres in PTcouples.
        """
        ctypes = ['sp_emission','ind_emission','absorption']

        Ps = np.unique(np.array([PT[0] for PT in self.PTcouples]))
        Ts = np.unique(np.array([PT[1] for PT in self.PTcouples]))

        try:
            if Pres <= np.min(Ps):
                closest_P1 = np.min(Ps)
                closest_TA_ind = np.argmin(np.abs(Ts-Temp))
                closest_TB_ind = np.argsort(np.abs(Ts-Temp))[1]
                closest_TA = Ts[closest_TA_ind]
                closest_TB = Ts[closest_TB_ind]

                ok1 = self.find(closest_P1,closest_TA)
                coeff_ok1 = self.sets[ok1]
                ok2 = self.find(closest_P1,closest_TB)
                coeff_ok2 = self.sets[ok2]
                set_ = dict()
                for ctype in ctypes:
                    set_[ctype] = coeff_ok1.interpolate(coeff_ok2, Temp = Temp)
            else:
                closest_P1_ind = np.argmin(np.abs(Ps-Pres))
                closest_P2_ind = np.argsort(np.abs(Ps-Pres))[1]
                closest_P1 = Ps[closest_P1_ind]
                closest_P2 = Ps[closest_P2_ind]

                # Now that I found the two closest Ps, I check which temps are closer to my case

                closest_T1_ind = np.argmin(np.abs(Ts-Temp))
                closest_T2_ind = np.argsort(np.abs(Ts-Temp))[1]
                closest_T1 = Ts[closest_T1_ind]
                closest_T2 = Ts[closest_T2_ind]

                # I'm doing first the P interpolation
                ok1 = self.find(closest_P1,closest_T1)
                coeff_ok1 = self.sets[ok1]
                ok2 = self.find(closest_P1,closest_T2)
                coeff_ok2 = self.sets[ok2]
                ok3 = self.find(closest_P2,closest_T1)
                coeff_ok3 = self.sets[ok3]
                ok4 = self.find(closest_P2,closest_T2)
                coeff_ok4 = self.sets[ok4]

                set_ = dict()
                for ctype in ctypes:
                    coeff_ok13 = coeff_ok1[ctype].interpolate(coeff_ok3[ctype], Pres = Pres)
                    coeff_ok24 = coeff_ok2[ctype].interpolate(coeff_ok4[ctype], Pres = Pres)
                    set_[ctype] = coeff_ok13.interpolate(coeff_ok24, Temp = Temp)
        except Exception as cazzillo:
            print('Unable to interpolate to pt couple {}'.format((Pres,Temp)))
            print('T set: {}'.format(Ts))
            print('P set: {}'.format(Ps))
            raise ValueError('cazzillo')

        return set_


    def make(self, spectral_grid, lines, PTcouples, control = True):
        """
        Produces the full set for PTcouples.
        """
        self.spectral_grid = copy.deepcopy(spectral_grid)

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
        if self.spectral_grid is None:
            self.spectral_grid = copy.deepcopy(spectral_grid)

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

            #print('iiiii add_PT iiiiii {} {} {}'.format(ctype, np.max(gigi.spectrum),np.min(gigi.spectrum)))

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
        pickle.dump(set_, self.temp_file, protocol = -1)

        if not keep_memory:
            del set_
        else:
            self.sets.append(set_)

        return

    def export(self, filename):
        pickle.dump(self, open(filename,'w'), protocol = -1)
        return

    def add_dump(self, set_):
        pickle.dump(set_, self.temp_file, protocol = -1)
        return


class AbsSetLOS(object):
    """
    This class represent a set of abs and emi coeffs along a LOS. This is to allow for saving in memory when calculating long sets.
    """

    def __init__(self, filename, indices = None):
        if indices is not None:
            self.indices = indices
        else:
            self.indices = []
        self.counter = 0
        self.remaining = 0
        self.filename = filename
        self.temp_file = None
        self.set = []
        return

    def prepare_read(self):
        #print('{} coefficients to be read sequentially..'.format(self.counter))
        if self.filename is None:
            raise ValueError('ERROR!: NO filename set for LutSet.')
        else:
            self.temp_file = open(self.filename, 'rb')
        self.remaining = self.counter
        return

    def prepare_export(self):
        """
        Opens the pic file for export, dumps PTcouples on top.
        """
        if self.filename is None:
            raise ValueError('ERROR!: NO filename set for LutSet.')
        else:
            self.temp_file = open(self.filename, 'wb')

        return

    def finalize_IO(self):
        self.temp_file.close()
        self.temp_file = None
        return

    def add_dump(self, set_):
        pickle.dump(set_, self.temp_file, protocol = -1)
        self.counter += 1
        #print('dampato. siamo a {}'.format(self.counter))
        #print(time.ctime())
        return

    def add_set(self, set_):
        self.set.append(set_)
        self.counter += 1
        return

    def read_one(self):
        if self.temp_file is None:
            self.prepare_read()

        set_ = pickle.load(self.temp_file)
        self.remaining -= 1

        return set_


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


def check_LUT_exists(PTcouples, cartLUTs, mol, iso, LTE):
    """
    Checks if a LUT is already present in cartLUTs for molname (Molec.name), isoname (iso_1, iso_2, ..) and if it is non-LTE ready (with levels) or not. Returns the PTcouples for which the LUT has not yet been calculated.
    """
    stringa = lut_name(mol, iso, LTE)
    fileok = [fil for fil in os.listdir(cartLUTs) if stringa in fil and not 'lev' in fil]

    if len(fileok) == 0:
        print('LUT does not exist at all')
        return False, PTcouples, None
    else:
        print('Found {} LUT files matching'.format(len(fileok)))

    pt_done = []
    pt_map = []
    for fil in fileok:
        print(cartLUTs+fil)
        fio = open(cartLUTs+fil, 'rb')
        lut_fil = pickle.load(fio)
        try:
            pt_fil = lut_fil.PTcouples
        except: # for compatibility with old LUTS
            lutset_fil = lut_fil.sets.values()[0]
            lutset_fil.load_from_file(load_just_PT = True)
            pt_fil = lutset_fil.PTcouples
        pt_done += pt_fil
        pt_map.append([cartLUTs+fil, pt_fil])
        fio.close()

    pt_to_do = []
    for pt in PTcouples:
        trovato = []
        for ptd in pt_done:
            trovato.append(np.all(sbm.isclose(np.array(pt),np.array(ptd))))
        if not np.any(np.array(trovato)):
            pt_to_do.append(pt)

    if len(pt_to_do) == 0:
        print('LUT exists and is complete')
    elif len(pt_to_do) > 0 and len(pt_done) > 0:
        print('LUT exists but {} PTcouples are missing'.format(len(pt_to_do)))
    else:
        print(pt_to_do)
        print(pt_done)
        print(mol, iso)
        print('LUT does not exist at all')

    return True, pt_to_do, pt_map


def check_and_build_allluts(inputs, sp_grid, lines, molecs, atmosphere = None, PTcouples = None, LUTopt = dict()):
    """
    Given the molecs in molecs, checks in the folder and builds the list of LUTS. If LUTS are missing runs the lut calculation?
    """
    allLUTs = dict()

    if PTcouples is None:
        if atmosphere is None:
            raise ValueError('Give either atmosphere or PTcouples in input as kwargs')
        PTcouples = calc_PT_couples_atmosphere(lines, molecs, atmosphere, **LUTopt)

    for molec in molecs:
        for isoname in molec.all_iso:
            isomol = getattr(molec, isoname)
            exists, pt_to_do, pt_map = check_LUT_exists(PTcouples, inputs['cart_LUTS'], isomol.mol, isomol.iso, isomol.is_in_LTE)

            if exists and len(pt_to_do) == 0:
                #leggi (hai solo una entry nel pt_map nel caso base)
                # caricare solo le PT couples giuste?
                filos = pt_map[0][0]
                with open(filos, 'r') as coso:
                    LUT_isomol = pickle.load(coso)
                if len(pt_map) > 1:
                    for [fi,ptfi] in pt_map[1:]:
                        with open(fi, 'r') as coso:
                            LUT_isomol.merge(pickle.load(coso))
            elif exists and len(pt_to_do) != 0:
                filos = pt_map[0][0]
                with open(filos, 'r') as coso:
                    LUT_isomol = pickle.load(coso)
                if len(pt_map) > 1:
                    for [fi,ptfi] in pt_map[1:]:
                        with open(fi, 'r') as coso:
                            LUT_isomol.merge(pickle.load(coso))
                print('Loaded {} existing PTcouples, producing new LUTs for {} PTcouples'.format(len(PTcouples)-len(pt_to_do), len(pt_to_do)))
                LUT_isomol_new = makeLUT_nonLTE_Gcoeffs(sp_grid, lines, isomol, isomol.is_in_LTE, PTcouples = pt_to_do, cartLUTs = inputs['cart_LUTS'], n_threads = inputs['n_threads'], **LUTopt)

                LUT_isomol.merge(LUT_isomol_new)
            else:
                # fai la lut da zero
                # find_wnrange(sp_grid, isomol)
                LUT_isomol = makeLUT_nonLTE_Gcoeffs(sp_grid, lines, isomol, isomol.is_in_LTE, PTcouples = PTcouples, cartLUTs = inputs['cart_LUTS'], n_threads = inputs['n_threads'], **LUTopt)

            allLUTs[(molec.name, isomol.iso)] = LUT_isomol

    return allLUTs


def calc_PT_couples_atmosphere(lines, molecs, atmosphere, pres_step_log = 0.4, temp_step = 5.0, max_pres = None, thres = 0.01):
    """
    Define pressure levels, for each pres check which temps are needed for the full atm and define a set of temperatures to be used at that pressure. Build something to be given as input to make LUTs.
    """

    logpres0 = np.log(np.max(atmosphere.pres))
    logpresMIN = np.log(np.min(atmosphere.pres))
    if max_pres is not None:
        logpres0 = np.log(max_pres)

    #print(logpresMIN, logpres0)
    #fixing at grid
    n0 = mt.ceil(logpres0/pres_step_log)
    logpres0 = n0*pres_step_log
    nMIN = mt.floor(logpresMIN/pres_step_log)
    logpresMIN = nMIN*pres_step_log
    #print(logpresMIN, logpres0)

    log_pressures = logpresMIN + np.arange(0,(logpres0-logpresMIN)+0.5*pres_step_log,pres_step_log)

    #print(log_pressures)
    #print('Built set of {} pressure levels from {} to {} with logstep = {}.'.format(len(log_pressures),logpresMIN,logpres0,pres_step_log))

    airbr = np.argmax(np.array([lin.Air_broad for lin in lines]))

    pressures = np.exp(log_pressures)

    #n_dim = atmosphere.grid.n_dim

    temps = []
    #### QUI c'è un problema se il passo di pressures è più fitto di quello di atmosphere.pres.............. DA RISOLVERE!
    ## mi sembra abbastanza una cazzata
    # grazie fede, molto utile
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

    okke = (atmosphere.pres >= pressures[-2]) & (atmosphere.pres <= pressures[-1])
    #print(np.any(okke))
    temps_pres = atmosphere.temp[okke]
    temps.append([np.min(temps_pres),np.max(temps_pres)])
    #print('For level {} with pres {} hPa, temp is in range: {} <-> {}'.format(len(temps)-1,pressures[-1],temps[-1][0],temps[-1][1]))

    # round_values for temp at 5 K steps and build couples P/T
    PTcouples = []
    for pres, trange in zip(pressures,temps):
        t_0 = (np.floor(trange[0]/temp_step)-1)*temp_step
        t_1 = (np.ceil(trange[1]/temp_step)+1)*temp_step
        all_t = np.arange(t_0,t_1+0.5*temp_step,temp_step)
        ##print(trange,all_t)
        for temp in all_t:
            PTcouples.append([pres,temp])

    mms = []
    if type(molecs) is not list:
        molecs = [molecs]
    for mol in molecs:
        if isinstance(mol, sbm.Molec):
            mms += [getattr(mol,isom).MM for isom in mol.all_iso]
        elif isinstance(mol, sbm.IsoMolec):
            mms.append(mol.MM)

    PTcouples_ok = []
    temps_lowpres = []
    for [Pres, Temp] in PTcouples:
        dw, lw, wsh = lines[airbr].CheckWidths(Temp, Pres, min(mms))
        if lw < thres*dw:
            #print('Skippo pressure level: {} << {}'.format(lw,dw))
            if Temp not in temps_lowpres:
                temps_lowpres.append(Temp)
        else:
            PTcouples_ok.append([Pres, Temp])

    Pres_0 = 1.e-8
    for Temp in temps_lowpres:
        PTcouples_ok.insert(0, [Pres_0, Temp])

    PTcouples = PTcouples_ok

    return PTcouples


def makeLUT_nonLTE_Gcoeffs(spectral_grid, lines, isomol, LTE, atmosphere = None, PTcouples = None, cartLUTs = None, pres_step_log = 0.4, temp_step = 5.0, save_LUTs = True, n_threads = n_threads, test = False, thres = 0.01, max_pres = None, check_num_couples = False):
    """
    Calculates the G_coeffs for the isomolec_levels at Temp and Pres.
    :param isomolecs: A list of isomolecs objects or a single one.
    """

    if PTcouples is None:
        PTcouples = calc_PT_couples_atmosphere(lines, isomol, atmosphere, pres_step_log = pres_step_log, temp_step = temp_step, max_pres = max_pres, thres = thres)

    if check_num_couples:
        return PTcouples

    if test:
        print('Keeping ONLY 10 PTcouples for testing')
        PTcouples = PTcouples[:10]

    print('Building LUTs for {} pres/temp couples... This may take some time... like {} minutes, maybe, not sure at all. Good luck ;)'.format(len(PTcouples),3.*len(PTcouples)))

    LUT = LookUpTable(isomol, spectral_grid.wn_range(), LTE)

    print(time.ctime())
    print("Hopefully this calculation will take about {} minutes, but actually I really don't know, take your time :)".format(LUT.CPU_time_estimate(lines, PTcouples)))

    LUT.make(spectral_grid, lines, PTcouples, export_levels = True, cartLUTs = cartLUTs)
    print(time.ctime())

    filename = cartLUTs+LUT.tag+date_stamp()+'.pic'
    filename = find_free_name(filename, maxnum = 10)
    LUT.export(filename)

    return LUT


def make_abscoeff_isomolec(wn_range_tot, isomolec, Temps, Press, LTE = True, allLUTs = None, useLUTs = False, lines = None, store_in_memory = False, tagLOS = None, cartDROP = None):
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

    if len(Temps) > 10:
        store_in_memory = True

    if cartDROP is None:
        cartDROP = 'stuff_'+date_stamp()
        if not os.path.exists(cartDROP):
            os.mkdir(cartDROP)
        cartDROP += '/'

    #print('Sto entrandooooooooooooo, mol {}, iso {}'.format(isomolec.mol, isomolec.iso))
    """
    Qui ci sono due wn_range, uno per le LUTs (wn_range) e uno per gli abs_coeff ed emi_coeff in output (wn_range_tot). Senza LUT i due sono uguali. Non so se vale la pena lasciare anche abs ed emi coeff a wn_range ridotta, risparmierei tempo di calcolo ma si incasina radtran.
    """
    if useLUTs:
        LUTs = allLUTs[(isomolec.mol_name, isomolec.iso)]
        wn_range = LUTs.wn_range
        spectral_grid = LUTs.spectral_grid
    else:
        wn_range = wn_range_tot
        coso = prepare_spe_grid(wn_range_tot)
        spectral_grid = coso.spectral_grid

    if tagLOS is None:
        tagLOS = 'LOS'
    tagg = tagLOS+'_mol_{}_iso_{}'.format(isomolec.mol, isomolec.iso)
    abs_coeffs = AbsSetLOS(cartDROP+'abscoeff_'+tagg+'.pic')
    emi_coeffs = AbsSetLOS(cartDROP+'emicoeff_'+tagg+'.pic')
    if store_in_memory:
        abs_coeffs.prepare_export()
        emi_coeffs.prepare_export()

    unidentified_lines = False
    if len(isomolec.levels) == 0:
        unidentified_lines = True
    #    print('acazuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu')

    set_tot = dict()
    if not unidentified_lines:
        for lev in isomolec.levels:
            levvo = getattr(isomolec, lev)
            strin = cartDROP+'LUTLOS_mol_{}_iso_{}_{}.pic'.format(isomolec.mol, isomolec.iso, lev)
            set_tot[lev] = LutSet(isomolec.mol, isomolec.iso, isomolec.MM, level = levvo, filename = strin)
            set_tot[lev].prepare_export([zui for zui in zip(Press,Temps)], spectral_grid)
    else:
        #print('siamo quaaaA')
        strin = cartDROP+'LUTLOS_mol_{}_iso_{}_alllev.pic'.format(isomolec.mol, isomolec.iso)
        set_tot['all'] = LutSet(isomolec.mol, isomolec.iso, isomolec.MM, level = None, filename = strin)
        set_tot['all'].prepare_export([zui for zui in zip(Press,Temps)], spectral_grid)


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
            #print(time.ctime())
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
        if not unidentified_lines:
            for lev in isomolec.levels:
                levello = getattr(isomolec, lev)
                ok, lev_lut = LUTs.find_lev(levello.lev_string)
                if not ok:
                    raise ValueError('mol {} iso {} Level {} not found'.format(isomolec.mol, isomolec.iso, levello.lev_string))
                LUTs.sets[lev_lut].load_from_files()
                for Pres, Temp in zip(Press,Temps):
                    set_ = LUTs.sets[lev_lut].calculate(Pres, Temp)
                    set_tot[lev].add_dump(set_)
                LUTs.sets[lev_lut].free_memory()
        else:
            LUTs.sets['all'].load_from_files()
            for Pres, Temp in zip(Press,Temps):
                set_ = LUTs.sets['all'].calculate(Pres, Temp)
                set_tot['all'].add_dump(set_)
            LUTs.sets['all'].free_memory()

    for val in set_tot.values():
        val.finalize_IO()
        #print('Finalizzzooooooooooo')

    for nam, val in zip(set_tot.keys(), set_tot.values()):
        val.prepare_read()
        #print('Reading... -> '+nam)

    for num in range(len(Temps)):
        #print('oyeeeeeeeeeee ', num)
        abs_coeff = prepare_spe_grid(wn_range_tot)
        emi_coeff = prepare_spe_grid(wn_range_tot)
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
                #for key, val in zip(Gco.keys(), Gco.values()):
                    #print('iiiii make_abs iiiiii {} {} {}'.format(key, np.max(val.spectrum),np.min(val.spectrum)))

                pop = spcl.Boltz_ratio_nodeg(levello.energy, vibt) / Q_part
                abs_coeff.add_to_spectrum(Gco['absorption'], Strength = pop)
                abs_coeff.add_to_spectrum(Gco['ind_emission'], Strength = -pop)
                emi_coeff.add_to_spectrum(Gco['sp_emission'], Strength = pop)

        if not store_in_memory:
            abs_coeffs.add_set(abs_coeff)
            emi_coeffs.add_set(emi_coeff)
        else:
            #print(pop)
            #print('iiiii make_abs 2 iiiiii {} {} {}'.format('absorb-ind_emiss', np.max(abs_coeff.spectrum),np.min(abs_coeff.spectrum)))
            #print('iiiii make_abs 2 iiiiii {} {} {}'.format('sp_emiss', np.max(emi_coeff.spectrum),np.min(emi_coeff.spectrum)))
            abs_coeffs.add_dump(abs_coeff)
            emi_coeffs.add_dump(emi_coeff)

    if store_in_memory:
        abs_coeffs.finalize_IO()
        emi_coeffs.finalize_IO()

    return abs_coeffs, emi_coeffs


def read_obs(filename, formato = 'gbb', wn_range = None):
    if formato == 'gbb':
        outs = sbm.read_obs(filename)
        spectra = outs[-2].T
        flags = outs[-1].T
        wn_arr = outs[-3]
        if wn_range is not None:
            cond = (wn_arr >= wn_range[0]) & (wn_arr <= wn_range[1])
            wn_arr = wn_arr[cond]

        gri = spcl.SpectralGrid(wn_arr, units = 'nm')
        obss = []
        for col,zol in zip(spectra, flags):
            if wn_range is not None:
                col = col[cond]
                zol = zol[cond]
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


def read_input_observed(observed_cart, wn_range = None):
    """
    Reads inputs regarding observations. The folder observed_cart has to contain an "orbit_***.dat", an "observ_***.dat". If more files are found inside, all observations are read.
    """

    print(os.listdir(observed_cart))
    tag_observ = [fil[7:] for fil in os.listdir(observed_cart) if 'observ_' in fil]

    set_pixels = []
    obs_tot = []
    orbit_tot = []

    for nomee in os.listdir(observed_cart):
        if 'band_titano' in nomee:
            bands = sbm.read_bands(observed_cart+nomee, wn_range = wn_range)
            break

    for nomee in os.listdir(observed_cart):
        if 'error' in nomee:
            # print(observed_cart+nomee)
            noise = sbm.read_noise(observed_cart+nomee, wn_range = wn_range)
            break

    for tag in tag_observ:
        orbit_tot += read_orbits(observed_cart+'orbit_'+tag, formato = 'VIMSselect', tag = tag)
        obs_tot += read_obs(observed_cart+'observ_'+tag, formato = 'gbb', wn_range = wn_range)

    for ob, orb in zip(obs_tot, orbit_tot):
        ob.add_noise(noise)
        ob.add_bands(bands)
        orb['observation'] = copy.deepcopy(ob)
        pix = sbm.VIMSPixel(orb.keys(), orb.values())
        set_pixels.append(pix)

    return set_pixels


def inversion(inputs, planet, lines, bayes_set, pixels, wn_range = None, chi_threshold = 0.01, max_it = 10, lambda_LM = 0.1, L1_reg = False, radtran_opt = dict(), useLUTs = True, debugfile = None, LUTopt = dict(), test = False):
    """
    Main routine for retrieval.
    """

    lines = check_lines_mols(lines, planet.gases.values())
    print('Begin inversion..')
    cartOUT = inputs['out_dir']

    if wn_range is None:
        sp_gri = pixels[0].observation.spectral_grid
        sp_gri.grid[0] -= 2*pixels[0].observation.bands.spectrum[0]
        sp_gri.grid[-1] += 2*pixels[0].observation.bands.spectrum[-1]
        sp_gri.convertto_cm_1()
        wn_range = [sp_gri.grid[0], sp_gri.grid[-1]]
        print(wn_range)
        sp_gri = prepare_spe_grid(wn_range)
        sp_gri = sp_gri.spectral_grid
    else:
        sp_gri = prepare_spe_grid(wn_range)
        sp_gri = sp_gri.spectral_grid

    # Update the VMRs of the retrieved gases with the a priori
    for gas in bayes_set.sets.keys():
        planet.gases[gas].add_clim(bayes_set.sets[gas].profile())


    # check if it is convenient to produce LUTs.
    # check if the LUTs have to be calculated or are already present. In case they are not, they'll be calculated
    if useLUTs:
        # finds max pressure
        alt_tg = []
        pres_max = []
        for pix in pixels:
            linea_0 = pix.low_LOS()
            linea_0.calc_atm_intersections(planet)
            press = linea_0.calc_along_LOS(planet.atmosphere, profname = 'pres', set_attr = False)
            alt_tg.append(linea_0.tangent_altitude)
            pres_max.append(max(press.ravel()))

        pres_max = max(pres_max)
        PTcoup_needed = calc_PT_couples_atmosphere(lines, planet.gases.values(), planet.atmosphere, **LUTopt)

        # if test:
        #     for i in range(20):
        #         print('TEST!! REDUCING TO 3 PT COUPLES!')
        #     PTcoup_needed = PTcoup_needed[:3]

        LUTS = check_and_build_allluts(inputs, sp_gri, lines, planet.gases.values(), PTcouples = PTcoup_needed, LUTopt = LUTopt)
        n_lut_tot = len(PTcoup_needed)
        print('{} PT couples needed'.format(n_lut_tot))
    else:
        LUTS = None

    sims = []
    obs = [pix.observation for pix in pixels]
    masks = [pix.observation.mask for pix in pixels]
    noise = [pix.observation.noise for pix in pixels]

    time0 = time.time()
    for num_it in range(max_it):
        print('we are at iteration: {}'.format(num_it))
        for pix, num in zip(pixels, range(len(pixels))):
            radtrans = []
            derivs = []

            linea_low = pix.low_LOS()
            linea0 = pix.LOS()
            linea_up = pix.up_LOS()
            print('pixel {} at tangent alt: {}'.format(num, pix.limb_tg_alt))

            # Using 1D interpolation scheme for FOV integration
            # x_FOV = np.array([linea_low.tangent_altitude, linea_0.tangent_altitude, linea_up.tangent_altitude])

            radtran_low = linea_low.radtran(wn_range, planet, lines, cartLUTs = inputs['cart_LUTS'], cartDROP = inputs['out_dir'], calc_derivatives = True, bayes_set = bayes_set, LUTS = LUTS, useLUTs = useLUTs, radtran_opt = radtran_opt)
            radtrans.append(radtran_low)
            deriv_ok = []
            for par in bayes_set.params():
                towl = par.hires_deriv
                lowres = towl.hires_to_lowres(pix.observation, spectral_widths = pix.observation.bands.spectrum)
                deriv_ok.append(lowres)
            derivs.append(deriv_ok)

            radtran = linea0.radtran(wn_range, planet, lines, cartLUTs = inputs['cart_LUTS'], cartDROP = inputs['out_dir'], calc_derivatives = True, bayes_set = bayes_set, LUTS = LUTS, useLUTs = useLUTs, radtran_opt = radtran_opt)
            radtrans.append(radtran)
            deriv_ok = []
            for par in bayes_set.params():
                towl = par.hires_deriv
                lowres = towl.hires_to_lowres(pix.observation, spectral_widths = pix.observation.bands.spectrum)
                deriv_ok.append(lowres)
            derivs.append(deriv_ok)

            radtran_up = linea_up.radtran(wn_range, planet, lines, cartLUTs = inputs['cart_LUTS'], cartDROP = inputs['out_dir'], calc_derivatives = True, bayes_set = bayes_set, LUTS = LUTS, useLUTs = useLUTs, radtran_opt = radtran_opt)
            radtrans.append(radtran_up)
            deriv_ok = []
            for par in bayes_set.params():
                towl = par.hires_deriv
                lowres = towl.hires_to_lowres(pix.observation, spectral_widths = pix.observation.bands.spectrum)
                deriv_ok.append(lowres)
            derivs.append(deriv_ok)

            intens_FOV = [rad[0] for rad in radtrans]
            intens_FOV_lowres = []
            for towl in intens_FOV:
                intens_FOV_lowres.append(towl.hires_to_lowres(pix.observation, spectral_widths = pix.observation.bands.spectrum))
                #intens_FOV_lowres.append(tolowres(towl, pix.observation))
            intens_FOV_lowres = np.array(intens_FOV_lowres)

            sim_FOV_ok = FOV_integr_1D(intens_FOV_lowres, pix.pixel_rot)
            try:
                sims[num] = copy.deepcopy(sim_FOV_ok)
            except:
                sims.append(copy.deepcopy(sim_FOV_ok))

            for par, der_low, der, der_up in zip(bayes_set.params(), derivs[0], derivs[1], derivs[2]):
                ders = np.array([der_low, der, der_up])
                der_FOV_ok = FOV_integr_1D(ders, pix.pixel_rot)
                par.store_deriv(der_FOV_ok, num = num)

        print('{} pixels done in {:5.1f} min'.format(num+1, (time.time()-time0)/60.))

        #INVERSIONE
        chi = chicalc(obs, sims, noise, masks, bayes_set.n_tot)
        print('chi is: {}'.format(chi))
        if num_it > 0:
            if abs(chi-chi_old)/chi_old < chi_threshold:
                print('FINISHEDDDD!! :D', chi)
                return
            elif chi > chi_old:
                print('mmm chi has raised', chi)
                return
        chi_old = chi
        print('old', [par.value for par in bayes_set.params()])
        inversion_algebra(obs, sims, noise, bayes_set, lambda_LM = lambda_LM, L1_reg = L1_reg, masks = masks)
        print('new', [par.value for par in bayes_set.params()])

        if debugfile is not None:
            pickle.dump([num_it, obs, sims, bayes_set], debugfile)
        # Update the VMRs of the retrieved gases with the new values
        for gas in bayes_set.sets.keys():
            planet.gases[gas].add_clim(bayes_set.sets[gas].profile())

    return


def FOV_integr_1D(radtrans, pixel_rot = 0.0):
    """
    FOV integration interpolating with a spline the spectrum between the external los.
    """
    from scipy import integrate

    pixel_rot = sbm.rad(pixel_rot)

    dmax = np.sqrt(2.)/2.*np.cos(np.pi/4-pixel_rot)
    #print(pixel_rot, dmax)
    delta = dmax-np.sin(pixel_rot)
    esse = 1/np.cos(pixel_rot)
    x_integ = np.array([-dmax, 0, dmax])
    #print(x_integ)
    spectrums = np.array([rad.spectrum for rad in radtrans])

    intens_spl = spline2D(x_integ, radtrans[0].spectral_grid.grid, spectrums, kx=2, ky=2)

    def integrand(x, ww, funz = intens_spl, delta = delta, dmax = dmax, esse = esse):
        if abs(x) <= delta:
            fu = intens_spl(x, ww)*esse
        else:
            fu = intens_spl(x, ww)*esse*abs(dmax-abs(x))/(dmax-delta)
        return fu

    integ_rad = copy.deepcopy(radtrans[0])
    spet_fov = []
    for ww in integ_rad.spectral_grid.grid:
        spint = integrate.quad(integrand, -dmax, dmax, args=(ww))[0]
        spet_fov.append(spint)

    integ_rad.spectrum = spet_fov
    return integ_rad


def genvec(obs, sims, noise, masks = None):
    if masks is not None:
        masktot = []
        for mas in masks:
            masktot += list(mas)
        masktot = np.array(masktot, dtype = bool)

    obs_vec = []
    sim_vec = []
    noi_vec = []
    for ob, noi, sim in zip(obs, noise, sims):
        obs_vec += list(ob.spectrum)
        sim_vec += list(sim.spectrum)
        noi_vec += list(noi.spectrum)

    obs_vec = np.array(obs_vec)
    sim_vec = np.array(sim_vec)
    noi_vec = np.array(noi_vec)

    if masks is not None:
        obs_vec = obs_vec[masktot]
        sim_vec = sim_vec[masktot]
        noi_vec = noi_vec[masktot]

    return obs_vec, sim_vec, noi_vec


def chicalc(obs, sims, noise, masks, n_ret):
    obs_vec, sim_vec, noi_vec = genvec(obs, sims, noise, masks = masks)

    chi = np.sqrt(np.sum(((obs_vec-sim_vec)/noi_vec)**2))
    chi = chi/(len(obs_vec)-n_ret)
    return chi

def inversion_algebra(obs, sims, noise, bayes_set, lambda_LM = 0.1, L1_reg = False, masks = None):
    """
    Bayesian optimal estimation. Levenberg-Marquardt iteration scheme.
    """

    jac = bayes_set.build_jacobian(masks = masks)
    xi = bayes_set.param_vector()

    obs_vec, sim_vec, noi_vec = genvec(obs, sims, noise, masks = masks)

    n_obs = len(obs_vec)
    S_y = np.identity(n_obs)
    for i,noi in zip(range(n_obs), noi_vec):
        S_y[i,i] = noi**2.0

    S_ap = bayes_set.VCM_apriori() #[par.apriori_err for par in bayes_set.params()]
    x_ap = bayes_set.apriori_vector()
    # S_ap = np.identity(n_params)
    # for i, noi in zip(range(bayes_set.n_tot), )

    KtSy = np.dot(jac.T, inv(S_y))
    G_inv = np.dot(KtSy, jac)
    S_inv = G_inv + inv(S_ap)
    LM = np.identity(S_ap.shape[0])*np.diag(S_inv)
    SxLM =  inv(S_inv + lambda_LM*LM)
    S_x = inv(S_inv)
    AVK = np.dot(S_x, G_inv)
    py = np.dot(KtSy, obs_vec-sim_vec)
    pa = np.dot(inv(S_ap), x_ap-xi)

    deltax = np.dot(SxLM, py+pa)

    bayes_set.update_params(deltax)
    bayes_set.store_avk(AVK)
    bayes_set.store_VCM(S_x)

    return


def tolowres(hires, obs):
    hires.convertto_nm()
    hires.interp_to_regular_grid()
    lowres = hires.convolve_to_grid(obs.spectral_grid, spectral_widths = obs.bands.spectrum)
    lowres.convertto('Wm2')
    return lowres
