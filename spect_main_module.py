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
from memory_profiler import profile

n_threads = 4

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

    !!!! CAUTION! : does not work with fake isomols.
    """

    lines_ok = []
    for mol in molecs:
        for iso in mol.all_iso:
            isomol = getattr(mol, iso)
            n_lm = 0
            if len(isomol.levels) > 0:
                lev_lines = [lin for lin in lines if lin.Mol == isomol.mol and lin.Iso == isomol.iso and isomol.has_level(lin.Lo_lev_str)[0] and isomol.has_level(lin.Up_lev_str)[0]]
                lines_ok += lev_lines
                n_lm += len(lev_lines)
            else:
                iso_lines = [lin for lin in lines if lin.Mol == isomol.mol and lin.Iso == isomol.iso]
                lines_ok += iso_lines
                n_lm = len(iso_lines)
            print('found {} lines for mol {} iso {}'.format(n_lm, isomol.mol, isomol.iso))

    return lines_ok


def keep_levels_wlines(planet, lines):
    """
    Keeps only the levels of the molecs in planet.gases that have some lines in linee.
    """

    for gas in planet.gases:
        mol = planet.gases[gas]
        for iso in mol.all_iso:
            isomol = getattr(mol, iso)
            iso_lines = [lin for lin in lines if lin.Mol == isomol.mol and lin.Iso == isomol.iso]
            erase_levels = []
            for lev in isomol.levels:
                levvo = getattr(isomol, lev)
                lev_lines = [lin for lin in iso_lines if levvo.equiv(lin.Lo_lev_str) or levvo.equiv(lin.Up_lev_str)]
                if len(lev_lines) == 0:
                    print('no lines for level {} of mol {} iso {}. Erasing..'.format(lev, isomol.mol, isomol.iso))
                    erase_levels.append(lev)

            for lev in erase_levels:
                isomol.erase_level(lev)

    return

def listbands(isomol, lines):
    """
    Simply lists all bands and the number of lines in lines.
    """
    isolin = [lin for lin in lines if lin.Mol == isomol.mol and lin.Iso == isomol.iso]

    for lev in isomol.levels:
        levvo = getattr(isomol, lev)
        for lev2 in isomol.levels:
            levvo2 = getattr(isomol, lev2)
            lev_lines = [lin for lin in isolin if (levvo2.equiv(lin.Lo_lev_str) and levvo.equiv(lin.Up_lev_str))]
            if len(lev_lines) > 0: print('{} -> {} : {} lines'.format(lev,lev2, len(lev_lines)))

    return

def keep_levels(planet, keep_levels, lines = None):
    """
    Keeps only the levels of the molecs in planet.gases that are in keep_levels. keep_levels is a dict: keep_levels[(gas, iso)] = levels_to_keep.
    If lines is not None checks for the lower levels of the transitions with the levels considered. These are kept even if not listed in keep_levels.
    """

    for gas in planet.gases:
        mol = planet.gases[gas]
        for iso in mol.all_iso:
            isomol = getattr(mol, iso)
            all_levels = copy.deepcopy(isomol.levels)
            for lev in all_levels:
                if not lev in keep_levels[(gas, iso)]:
                    print('erasing level {} of mol {} iso {}'.format(lev, isomol.mol, isomol.iso))
                    isomol.erase_level(lev)

    return


def track_all_levels(planet):
    track_levels = dict()
    for molnam in planet.gases.keys():
        mol = planet.gases[molnam]
        for iso in mol.all_iso:
            isomol = getattr(mol, iso)
            track_levels[(molnam, iso)] = isomol.levels

    return track_levels

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

    def update_parerror(self):
        for par, num in zip(self.params(), range(self.n_tot)):
            par.ret_error = np.sqrt(self.VCM[num,num])
        return

    def n_used_par(self):
        return sum([par.is_used for par in self.params()])

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

    def show_VCM(self):
        pl.imshow(self.VCM, interpolation='nearest')
        pl.colorbar()
        return

    def show_av_kernel(self):
        pl.imshow(self.av_kernel, interpolation='nearest')
        pl.colorbar()
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

    def plot_profile(self, label = None, fix_lat = None, logplot = False):
        if label is None:
            label = self.name
        self.profile().plot(label = label, fix_lat = fix_lat, logplot = logplot)
        return



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
        self.alts = list(alt_nodes)
        self.lats = list(lat_limits)

        alt_grid = sbm.AtmGrid('alt', atmosphere.grid.coords['alt'])

        if first_guess_profs is None:
            first_guess_profs = apriori_profs

        for apriori_prof, apriori_prof_err, first_guess_prof, lat in zip(apriori_profs, apriori_prof_errs, first_guess_profs, lat_limits):
            coso = LinearProfile_1D_new(name, alt_grid, alt_nodes, apriori_prof, apriori_prof_err)

            latbox = lat_box(lat_limits, lat)
            for cos in coso.set:
                numask = cos.maskgrid.merge(latbox)
                par = RetParam(name, (lat, cos.key), numask, cos.apriori, cos.apriori_err)
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


    def check_involved(self, parkey, coord_range):
        indp = self.alts.index(parkey[1])
        involved = True
        altz = coord_range['alt']

        if indp == len(self.alts)-1:
            pass
        elif altz[0] > self.alts[indp+1]:
            involved = False

        latz = coord_range['lat']
        indp = self.lats.index(parkey[0])
        if indp == len(self.lats)-1:
            if latz[1] < parkey[0]:
                involved = False
        elif latz[1] < parkey[0] or latz[0] > self.lats[indp+1]:
            involved = False

        return involved


class LinearProfile_1D_new(RetSet):
    """
    A profile constructed through linear interpolation of a set of params.
    """

    def __init__(self, name, alt_grid, alt_nodes, apriori_prof, apriori_prof_err, first_guess_prof = None):
        self.name = name
        self.set = []
        self.n_par = len(alt_nodes)
        self.alts = list(alt_nodes)

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

    def check_involved(self, parkey, coord_range):
        indp = self.alts.index(parkey)
        involved = True
        altz = coord_range['alt']

        if indp == len(self.alts)-1:
            pass
        elif altz[0] > self.alts[indp+1]:
            involved = False

        return involved

    def plot_points(self, with_err = True, plot_prof = False, label = None, logplot = False, linewidth = 2.0, color = None):
        if label is None:
            label = self.name
        alts = []
        parval = []
        parerr = []
        for par in self.set:
            alts.append(par.key)
            parval.append(par.value)
            parerr.append(par.ret_error)

        if with_err:
            pl.errorbar(parval, alts, xerr=parerr, linewidth=linewidth, label = label, color = color)
        else:
            if plot_prof:
                pio = pl.plot(parval, alts, color = color)
                color = pio[0].get_color()
            pl.scatter(parval, alts, label = label, color = color)

        if logplot:
            pl.xscale('log')

        return

    def plot_apriori_points(self, with_err = True, plot_prof = False, label = None, logplot = False, linewidth = 2.0, color = None):
        if label is None:
            label = self.name
        alts = []
        parval = []
        parerr = []
        for par in self.set:
            alts.append(par.key)
            parval.append(par.apriori)
            parerr.append(par.apriori_err)

        if with_err:
            pl.errorbar(parval, alts, xerr=parerr, linewidth=linewidth, label = label, color = color)
        else:
            if plot_prof:
                pio = pl.plot(parval, alts, color = color)
                color = pio[0].get_color()
            pl.scatter(parval, alts, label = label, color = color)

        if logplot:
            pl.xscale('log')

        return

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
        self.not_involved = False
        self.is_used = False
        return

    def set_not_involved(self):
        self.not_involved = True
        return

    def set_used(self):
        self.is_used = True
        return

    def set_involved(self):
        self.not_involved = False
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

    def erase_hires_deriv(self):
        self.hires_deriv = None
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

def lut_name_split(mol, iso, LTE, split):
    if LTE:
        name = 'LUT_csplit{:02d}_mol{:02d}_iso{:1d}_LTE'.format(split,mol,iso)
    else:
        name = 'LUT_csplit{:02d}_mol{:02d}_iso{:1d}_nonLTE'.format(split,mol,iso)

    return name

def lut_name_wsplits(mol, iso, LTE, n_split):
    if LTE:
        name = 'LUT_{:02d}splits_mol{:02d}_iso{:1d}_LTE'.format(n_split,mol,iso)
    else:
        name = 'LUT_{:02d}splits_mol{:02d}_iso{:1d}_nonLTE'.format(n_split,mol,iso)

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
        if self.wn_range != LUT.wn_range:
            print('Attention! The wn_range of the two LUTs is different!')
            raise ValueError('Incompatible LUTs, different wn_ranges: {} {}'.format(self.wn_range, LUT.wn_range))

        for lev_name in self.sets.keys():
            lev1 = self.sets[lev_name]
            lev2 = LUT.sets[lev_name]
            if not self.LTE:
                if not lev1.level.equiv(lev2.level.lev_string):
                    raise ValueError('Levels are different, cannot merge LUTs')
            lev1.add_file(lev2.filename, lev2.PTcouples)

        self.PTcouples += LUT.PTcouples
        return

    def make(self, spectral_grid, lines, PTcouples, export_levels = True, cartLUTs = None, control = True, n_threads = n_threads):
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
            lines_proc = None
            lines_proc = spcl.calc_shapes_lines(spectral_grid, lines, Temp, Pres, self.isomolec, n_threads = n_threads)

            print("PTcouple {} out of {}. P = {}, T = {}. Lineshapes calculated in {:5.1f} s, time from start {:7.1f} min".format(num,len(PTcouples),Pres,Temp,time.time()-time1,(time.time()-time0)/60.))

            if control:
                comm = 'echo "PTcouple {} out of {}. P = {}, T = {}. Lineshapes calculated in {:5.1f} s, time from start {:7.1f} min" >> control_spectrobot'.format(num,len(PTcouples),Pres,Temp,time.time()-time1,(time.time()-time0)/60.)
                os.system(comm)

            time1 = time.time()

            if not self.LTE:
                for lev in self.isomolec.levels:
                    #print('Building LutSet for level {} of mol {}, iso {}'.format(lev,self.mol,self.iso))
                    self.sets[lev].add_PT(spectral_grid, lines_proc, Pres, Temp, keep_memory = False, n_threads = n_threads)
            else:
                self.sets['all'].add_PT(spectral_grid, lines_proc, Pres, Temp, keep_memory = False, n_threads = n_threads)

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

    def add_split_file(self, filename):
        if not hasattr(self, 'splitfiles'):
            self.splitfiles = []

        self.splitfiles.append(filename)
        return

    def load_split(self, nsp):
        splitfi = open(self.splitfiles[nsp], 'rb')
        sett_nuovo = dict()
        for nam in self.sets:
            coso = pickle.load(splitfi)
            sett_nuovo[coso[0]] = coso[1]

        for nam in self.sets:
            self.sets[nam] = sett_nuovo[nam]
            for num in range(len(self.sets[nam].PTcouples)):
                set_ = self.sets[nam].sets[num]
                for ctype in set_:
                    if set_[ctype] is None: continue
                    set_[ctype].double_precision()
                    set_[ctype].restore_grid(self.sets[nam].spectral_grid, link_grid = True)

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

    def load_from_files(self, load_just_PT = False, spectral_grid = None, cartLUTs = '/home/fedefab/Scrivania/Research/Dotto/Spect_data/LUTs/'):
        """
        Loads from file just the data regarding level's LutSet. Better not to load all levels together due to memory limits.
        Adapted for more than one file.
        """

        self.PTcouples = []
        for filename in self.filenames:
            try:
                fileo = open(filename,'rb')
            except:
                ind = filename.rfind('/')
                filena = filename[ind+1:]
                filename = cartLUTs+filena
                fileo = open(filename,'rb')
            PTfil = pickle.load(fileo)
            self.PTcouples += PTfil

            if load_just_PT:
                fileo.close()
            else:
                for PT in PTfil:
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

        # try:
        if Pres <= np.min(Ps):
            #print(Pres)
            closest_P1 = np.min(Ps)
            closest_TA_ind = np.argmin(np.abs(Ts-Temp))
            closest_TB_ind = np.argsort(np.abs(Ts-Temp))[1]
            closest_TA = Ts[closest_TA_ind]
            closest_TB = Ts[closest_TB_ind]
            #print(closest_TA, closest_TB)

            ok1 = self.find(closest_P1,closest_TA)
            coeff_ok1 = self.sets[ok1]
            ok2 = self.find(closest_P1,closest_TB)
            coeff_ok2 = self.sets[ok2]
            set_ = dict()
            for ctype in ctypes:
                if coeff_ok1[ctype] is None or coeff_ok2[ctype] is None:
                    set_[ctype] = None
                    continue
                set_[ctype] = coeff_ok1[ctype].interpolate(coeff_ok2[ctype], Temp = Temp)
        elif Pres > np.min(Ps) and Pres <= np.max(Ps):
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
                if coeff_ok1[ctype] is None or coeff_ok2[ctype] is None or coeff_ok3[ctype] is None or coeff_ok4[ctype] is None:
                    set_[ctype] = None
                    continue
                coeff_ok13_cty = coeff_ok1[ctype].interpolate(coeff_ok3[ctype], Pres = Pres)
                coeff_ok24_cty = coeff_ok2[ctype].interpolate(coeff_ok4[ctype], Pres = Pres)
                set_[ctype] = coeff_ok13_cty.interpolate(coeff_ok24_cty, Temp = Temp)
        else:
            raise ValueError('Extrapolating in P')
        # except Exception as cazzillo:
        #     print('Unable to interpolate to pt couple {}'.format((Pres,Temp)))
        #     print('T set: {}'.format(Ts))
        #     print('P set: {}'.format(Ps))
        #     print('PT couples : {}'.format(self.PTcouples))
        #     raise cazzillo

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


    def add_PT(self, spectral_grid, lines, Pres, Temp, keep_memory = False, control = True, n_threads = n_threads):
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
            gigi.BuildCoeff(lines, Temp, Pres, preCalc_shapes = True, n_threads = n_threads)

            print('iiiii add_PT iiiiii {} {} {} {}'.format(ctype, np.max(gigi.spectrum),np.min(gigi.spectrum),gigi.integrate()))

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

    def __init__(self, filename, spectral_grid = None, indices = None):
        if indices is not None:
            self.indices = indices
        else:
            self.indices = []
        self.counter = 0
        self.remaining = 0
        self.filename = filename
        self.temp_file = None
        self.set = []
        self.spectral_grid = spectral_grid
        return

    def prepare_read(self, read_spectral_grid = True):
        #print('{} coefficients to be read sequentially..'.format(self.counter))
        if self.filename is None:
            raise ValueError('ERROR!: NO filename set for LutSet.')
        else:
            self.temp_file = open(self.filename, 'rb')
        self.remaining = self.counter
        if read_spectral_grid:
            self.spectral_grid = pickle.load(self.temp_file)
        return

    def prepare_export(self):
        """
        Opens the pic file for export, dumps PTcouples on top.
        """
        if self.filename is None:
            raise ValueError('ERROR!: NO filename set for LutSet.')
        else:
            self.temp_file = open(self.filename, 'wb')

        if self.spectral_grid is not None:
            pickle.dump(self.spectral_grid, self.temp_file, protocol = -1)

        return

    def finalize_IO(self):
        self.temp_file.close()
        self.temp_file = None
        return

    def add_dump(self, set_, no_spectral_grid = True):
        if no_spectral_grid:
            if type(set_) is dict:
                for cos in set_:
                    set_[cos].erase_grid()
            else:
                set_.erase_grid()
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
        if type(set_) is dict:
            for cos in set_:
                set_[cos].restore_grid(self.spectral_grid, link_grid = True)
        else:
            set_.restore_grid(self.spectral_grid, link_grid = True)

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
    sp_grids = []
    for fil in fileok:
        print(cartLUTs+fil)
        fio = open(cartLUTs+fil, 'rb')
        lut_fil = pickle.load(fio)
        sp_grids.append(lut_fil.spectral_grid)
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

    wn_ranges = [spgri.wn_range() for spgri in sp_grids]
    w1s = []
    w2s = []
    for [w1, w2] in wn_ranges:
        w1s.append(w1)
        w2s.append(w2)

    ok1 = np.all([sbm.isclose(w,w1s[0]) for w in w1s])
    ok2 = np.all([sbm.isclose(w,w2s[0]) for w in w2s])

    if ok1 and ok2:
        print('wn_range check ok')
    else:
        print('wn_ranges are different!')
        for fil, wnr in zip(fileok, wn_ranges):
            print(fil, wnr)

    return True, pt_to_do, pt_map, wn_ranges


def check_and_build_allluts(inputs, sp_grid, lines, molecs, atmosphere = None, PTcouples = None, LUTopt = dict(), check_wn_range = True):
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
            exists, pt_to_do, pt_map, wn_ranges = check_LUT_exists(PTcouples, inputs['cart_LUTS'], isomol.mol, isomol.iso, isomol.is_in_LTE)

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

                if not sbm.isclose(sp_grid.wn_range()[0],wn_ranges[0][0]) or not sbm.isclose(sp_grid.wn_range()[1],wn_ranges[0][1]):
                    print('ATTENTION building LUT for a different wn_range')
                    if check_wn_range:
                        raise ValueError('ATTENTION desired wn_range {} is different from the existing one {}'.format(sp_grid.wn_range(), wn_ranges[0]))

                LUT_isomol_new = makeLUT_nonLTE_Gcoeffs(sp_grid, lines, isomol, isomol.is_in_LTE, PTcouples = pt_to_do, cartLUTs = inputs['cart_LUTS'], n_threads = inputs['n_threads'], **LUTopt)

                LUT_isomol.merge(LUT_isomol_new)
            else:
                # fai la lut da zero
                # find_wnrange(sp_grid, isomol)
                LUT_isomol = makeLUT_nonLTE_Gcoeffs(sp_grid, lines, isomol, isomol.is_in_LTE, PTcouples = PTcouples, cartLUTs = inputs['cart_LUTS'], n_threads = inputs['n_threads'], **LUTopt)

            allLUTs[(molec.name, isomol.iso)] = LUT_isomol

    return allLUTs


def best_compressed_grid(simuls, thress = [1.e-3, 1.e-4, 1.e-5], factors = [5,20,100], skip_thres = 1.e-20, consider_derivatives = True, factor_minor = 10, thres_minor = 1.e-2, alg = 2):
    gridi_best = np.array([])
    maxo = 0.
    for singles in simuls:
        for tag in singles:
            if singles[tag].max() > maxo:
                maxo = singles[tag].max()

    for singles in simuls:
        for tag in singles:
            if singles[tag].max() < skip_thres:
                continue
            if singles[tag].max() < thres_minor*maxo:
                thress_tag = list(factor_minor*np.array(thress))
            else:
                thress_tag = thress
            if alg == 1:
                gridi = singles[tag].degrade_grid(thress = thress_tag, factors = factors, consider_derivatives = consider_derivatives).spectral_grid.grid
            elif alg == 2:
                gridi = singles[tag].degrade_grid2(thres = max(thress_tag), consider_derivatives = False).spectral_grid.grid
            else:
                gridi = singles[tag].degrade_grid3().spectral_grid.grid
            gridi_best = np.append(gridi_best, gridi)

    gridi_best = np.sort(np.unique(gridi_best))

    return gridi_best


def compress_LUTS(molecs, cartLUTs, n_threads, ram_max = 8., dim_tot = 20., low_thres = 1.e-30, new_grid = None):
    """
    Reads the luts in memory and splits them into smaller wn ranges. If new_grid is specified, the luts are first interpolated to the new grid. If the G_coeff considered is lower than low_thres, the zero flag is raised.
    """
    # read all luts and determine grid
    dim_tot = 20.
    ram_max = 8.

    n_split = np.ceil(dim_tot*n_threads/ram_max)

    for molec in molecs:
        for isoname in molec.all_iso:
            isomol = getattr(molec, isoname)
            exists, pt_to_do, pt_map, wn_ranges = check_LUT_exists([], cartLUTs, isomol.mol, isomol.iso, isomol.is_in_LTE)

            if exists:
                filos = pt_map[0][0]
                with open(filos, 'r') as coso:
                    LUT_isomol = pickle.load(coso)
                if len(pt_map) > 1:
                    for [fi,ptfi] in pt_map[1:]:
                        with open(fi, 'r') as coso:
                            LUT_isomol.merge(pickle.load(coso))
            else:
                raise ValueError('LUT does not exist!')

            for lev in isomol.levels:
                levello = getattr(isomol, lev)
                ok, lev_lut = LUT_isomol.find_lev(levello.lev_string)
                if not ok:
                    raise ValueError('mol {} iso {} Level {} not found'.format(isomolec.mol, isomolec.iso, levello.lev_string))

                LUT_isomol.sets[lev_lut].load_from_files()
                PTcouples = LUT_isomol.sets[lev_lut].PTcouples
                Pmin = min([P for [P,T] in PTcouples])
                Tmin = min([T for [P,T] in PTcouples if sbm.isclose(P, Pmin)])
                ind = LUT_isomol.sets[lev_lut].find(Pmin,Tmin)
                sett_min = LUT_isomol.sets[lev_lut].sets[ind]
                Pmax = max([P for [P,T] in PTcouples])
                Tmax = max([T for [P,T] in PTcouples if sbm.isclose(P, Pmax)])
                ind = LUT_isomol.sets[lev_lut].find(Pmax,Tmax)
                sett_max = LUT_isomol.sets[lev_lut].sets[ind]

                for nam in sett_min:
                    out1 = sett_min[nam].degrade_grid(thress = thress, factors = factors, consider_derivatives = consider_derivatives)
                    gridi = out1.spectral_grid.grid
                    gridi_best = np.append(gridi_best, gridi)
                    out2 = sett_max[nam].degrade_grid(thress = thress, factors = factors, consider_derivatives = consider_derivatives)
                    gridi2 = out2.spectral_grid.grid
                    gridi_best = np.append(gridi_best, gridi2)
                    # if sett_min[nam].max() > 0:
                    #     pl.figure(77)
                    #     pl.scatter(sett_min[nam].spectral_grid.grid, sett_min[nam].spectrum, s=1)
                    #     pl.scatter(out1.spectral_grid.grid, out1.spectrum, s=1)
                    #     pl.figure(78)
                    #     sett_min[nam].plot(label='min orig')
                    #     out1.plot(label='min deg')
                    #     sett_max[nam].plot(label='max orig')
                    #     out2.plot(label='max deg')
                    #     sys.exit()


                gridi_best = np.sort(np.unique(gridi_best))

    print('THIS IS NOT THE BEST GRID')
    return gridi_best

    # check split wrt memory available

    # create all compressed luts

    #return allLUTs
def split_and_compress_LUTS(spectral_grid, allLUTs, cartLUTs, n_threads, n_split = None, ram_max = 8., dim_tot = 20., low_thres = 1.e-30):
    """
    Reads the luts in memory and splits them into smaller wn ranges. If new_grid is specified, the luts are first interpolated to the new grid. If the G_coeff considered is lower than low_thres, the G_coeff is set equal to None.
    """
    # read all luts and determine grid
    if n_split is None:
        n_split = int(np.ceil(dim_tot*n_threads/ram_max))

    sp_grids = []
    i = 0
    len_split = int(np.ceil(1.0*len(spectral_grid.grid)/n_split))
    for nsp in range(n_split):
        nugri = spectral_grid.grid[i:i+len_split]
        gigi = copy.deepcopy(spectral_grid)
        gigi.grid = nugri
        sp_grids.append(gigi)
        i += len_split

    already_done = dict()
    for lutnam in allLUTs:
        stringa = lut_name_wsplits(allLUTs[lutnam].mol, allLUTs[lutnam].iso, allLUTs[lutnam].LTE, n_split)
        fileok = [fil for fil in os.listdir(cartLUTs) if stringa in fil]

        print(lutnam)
        print(fileok)
        if len(fileok) == 1:
            coso = open(cartLUTs+fileok[0],'r')
            allLUTs[lutnam] = pickle.load(coso)
            already_done[lutnam] = True
            print('già fatto', stringa)
            continue
        else:
            print('non ce', stringa)
            already_done[lutnam] = False

        splitfiles = []
        for nsp in range(n_split):
            nuname = lut_name_split(allLUTs[lutnam].mol, allLUTs[lutnam].iso, allLUTs[lutnam].LTE, nsp)
            nuovonome = cartLUTs+nuname+date_stamp()+'.pic'
            fio = open(nuovonome, 'wb')
            splitfiles.append(fio)
            allLUTs[lutnam].add_split_file(nuovonome)

        isomol = allLUTs[lutnam].isomolec
        if not allLUTs[lutnam].LTE:
            for lev in isomol.levels:
                levello = getattr(isomol, lev)
                ok, lev_lut = allLUTs[lutnam].find_lev(levello.lev_string)
                if not ok:
                    raise ValueError('mol {} iso {} Level {} not found'.format(isomolec.mol, isomolec.iso, levello.lev_string))

                allLUTs[lutnam].sets[lev_lut].load_from_files(cartLUTs = cartLUTs)

                for nsp, spgri, splitfile in zip(range(n_split), sp_grids, splitfiles):
                    nuset = copy.deepcopy(allLUTs[lutnam].sets[lev_lut])
                    splitset = []
                    for sett in nuset.sets:
                        settino = dict()
                        for cos in sett:
                            gigi = sett[cos].interp_to_grid(spgri)
                            if gigi.max() > 0.0:
                                gigi.erase_grid()
                                gigi.half_precision()
                                settino[cos] = gigi
                            else:
                                settino[cos] = None
                        splitset.append(settino)
                    nuset.sets = splitset
                    nuset.spectral_grid = spgri

                    pickle.dump([lev_lut, nuset], splitfile, protocol=-1)

                allLUTs[lutnam].sets[lev_lut].free_memory()

            for fio in splitfiles:
                fio.close()
        else:
            allLUTs[lutnam].sets['all'].load_from_files(cartLUTs = cartLUTs)

            for nsp, spgri, splitfile in zip(range(n_split), sp_grids, splitfiles):
                nuset = copy.deepcopy(allLUTs[lutnam].sets['all'])
                splitset = []
                for sett in nuset.sets:
                    settino = dict()
                    for cos in sett:
                        gigi = sett[cos].interp_to_grid(spgri)
                        if gigi.max() > 0.0:
                            gigi.erase_grid()
                            gigi.half_precision()
                            settino[cos] = gigi
                        else:
                            settino[cos] = None
                    splitset.append(settino)
                nuset.sets = splitset
                nuset.spectral_grid = spgri

                pickle.dump(['all', nuset], splitfile, protocol=-1)

            allLUTs[lutnam].sets['all'].free_memory()
            for fio in splitfiles:
                fio.close()

    for nam in allLUTs:
        if already_done[nam]:
            print('ce',nam)
            continue
        else:
            print('nonce',nam)
        nuname = lut_name_wsplits(allLUTs[nam].mol, allLUTs[nam].iso, allLUTs[nam].LTE, n_split)
        nuovonome = cartLUTs+nuname+date_stamp()+'.pic'
        fio = open(nuovonome, 'w')
        pickle.dump(allLUTs[nam], fio)
        fio.close()

    return allLUTs, n_split, sp_grids


def group_observations(pixels, ret_set):
    """
    Defines a set of LOSs suitable for calculating all required radtrans.
    """
    tg_points = [pix.get_tangent_point() for pix in pixels]
    spacecraft = [pix.Spacecraft() for pix in pixels]
    alt_tg = [po.Spherical()[2] for po in tg_points]
    lat_tg = [po.Spherical()[0] for po in tg_points]


    # potrei trovare punto di ingresso e uscita da atm anche

    pass


def calc_PT_couples_atmosphere(lines, molecs, atmosphere, pres_step_log = 0.4, temp_step = 5.0, max_pres = None, thres = 0.01, add_lowpres = True):
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
    Pres_0 = 1.e-8
    for [Pres, Temp] in PTcouples:
        dw, lw, wsh = lines[airbr].CheckWidths(Temp, Pres, min(mms))
        if lw < thres*dw:
            #print('Skippo pressure level: {} << {}'.format(lw,dw))
            if Pres > Pres_0:
                Pres_0 = Pres
            if Temp not in temps_lowpres:
                temps_lowpres.append(Temp)
        else:
            PTcouples_ok.append([Pres, Temp])

    for Temp in temps_lowpres:
        PTcouples_ok.insert(0, [Pres_0, Temp])

    if add_lowpres:
        presmin = np.exp(logpresMIN)
        for Temp in temps_lowpres:
            PTcouples_ok.insert(0, [presmin, Temp])

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

    LUT.make(spectral_grid, lines, PTcouples, export_levels = True, cartLUTs = cartLUTs, n_threads = n_threads)
    print(time.ctime())

    filename = cartLUTs+LUT.tag+date_stamp()+'.pic'
    filename = find_free_name(filename, maxnum = 10)
    LUT.export(filename)

    return LUT


def make_abscoeff_isomolec(wn_range_tot, isomolec, Temps, Press, LTE = True, allLUTs = None, useLUTs = False, lines = None, store_in_memory = False, tagLOS = None, cartDROP = None, track_levels = None, n_threads = n_threads):
    """
    Builds the absorption and emission coefficients for isomolec, both in LTE and non-LTE. If in non-LTE, isomolec levels have to contain the attribute local_vibtemp, produced by calling level.add_local_vibtemp(). If LTE is set to True, LTE populations are used.
    LUT is the object created by makeLUT_nonLTE_Gcoeffs(). Contains
    """

    time0 = time.time()

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
    abs_coeffs = AbsSetLOS(cartDROP+'abscoeff_'+tagg+'.pic', spectral_grid = spectral_grid)
    emi_coeffs = AbsSetLOS(cartDROP+'emicoeff_'+tagg+'.pic', spectral_grid = spectral_grid)
    if track_levels is not None:
        emi_coeffs_tracked = dict()
        abs_coeffs_tracked = dict()
        for lev in track_levels:
            tagg = tagLOS+'_mol_{}_iso_{}_{}'.format(isomolec.mol, isomolec.iso, lev)
            emi_coeffs_tracked[lev] = AbsSetLOS(cartDROP+'tracklevel_emicoeff_'+tagg+'.pic', spectral_grid = spectral_grid)
            abs_coeffs_tracked[lev] = AbsSetLOS(cartDROP+'tracklevel_abscoeff_'+tagg+'.pic', spectral_grid = spectral_grid)
    if store_in_memory:
        abs_coeffs.prepare_export()
        emi_coeffs.prepare_export()
        if track_levels is not None:
            for lev in track_levels:
                emi_coeffs_tracked[lev].prepare_export()
                abs_coeffs_tracked[lev].prepare_export()

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

    timooo = 0.0
    timuuu = 0.0
    timuuu_old = 0.0
    timuuu_fast = 0.0
    timaaa = 0.0
    timaaa2 = 0.0
    timhhh = 0.0

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
            intli = [lin.shape.integrate() for lin in lines_proc]
            print('shape',numh,max(intli),min(intli))
            #print(len(lines_proc))
            if not unidentified_lines:
                for lev in isomolec.levels:
                    #print('Siamo a mol {}, iso {}, lev {} bauuuuuuuuu'.format(isomolec.mol, isomolec.iso, lev))
                    levello = getattr(isomolec, lev)
                    set_tot[lev].add_PT(spectral_grid, lines_proc, Pres, Temp, n_threads = n_threads)
            else:
                #print('Siamo a mol {}, iso {}, all_levssss miaoooooooooooo'.format(isomolec.mol, isomolec.iso))
                set_tot['all'].add_PT(spectral_grid, lines_proc, Pres, Temp, n_threads = n_threads)

    else:
        if not unidentified_lines:
            for lev in isomolec.levels:
                levello = getattr(isomolec, lev)
                ok, lev_lut = LUTs.find_lev(levello.lev_string)
                if not ok:
                    raise ValueError('mol {} iso {} Level {} not found'.format(isomolec.mol, isomolec.iso, levello.lev_string))
                time1 = time.time()
                LUTs.sets[lev_lut].load_from_files()
                timaaa += time.time()-time1
                for Pres, Temp in zip(Press,Temps):
                    time1 = time.time()
                    set_ = LUTs.sets[lev_lut].calculate(Pres, Temp)
                    timooo += time.time()-time1
                    time1 = time.time()
                    set_tot[lev].add_dump(set_)
                    timaaa2 += time.time()-time1
                LUTs.sets[lev_lut].free_memory()
        else:
            LUTs.sets['all'].load_from_files()
            for Pres, Temp in zip(Press,Temps):
                time1 = time.time()
                set_ = LUTs.sets['all'].calculate(Pres, Temp)
                timooo += time.time()-time1
                set_tot['all'].add_dump(set_)
            LUTs.sets['all'].free_memory()

    print('     -  make abs part 1: {:5.1f} s'.format((time.time()-time0)))
    time0 = time.time()

    for val in set_tot.values():
        val.finalize_IO()
        #print('Finalizzzooooooooooo')

    print('     -  make abs part 1bis: {:5.1f} s'.format((time.time()-time0)))
    time0 = time.time()

    for nam, val in zip(set_tot.keys(), set_tot.values()):
        val.prepare_read()
        #print('Reading... -> '+nam)

    print('     -  make abs part 1tris: {:5.1f} s'.format((time.time()-time0)))
    time0 = time.time()

    for num in range(len(Temps)):
        time2 = time.time()
        #print('oyeeeeeeeeeee ', num)
        abs_coeff = prepare_spe_grid(wn_range_tot)
        emi_coeff = prepare_spe_grid(wn_range_tot)
        if track_levels is not None:
            emi_coeff_level = dict()
            abs_coeff_level = dict()
            for lev in track_levels:
                emi_coeff_level[lev] = prepare_spe_grid(wn_range_tot)
                abs_coeff_level[lev] = prepare_spe_grid(wn_range_tot)

        # THIS IS WITH LTE PARTITION FUNCTION!!
        Q_part = spcl.CalcPartitionSum(isomolec.mol, isomolec.iso, temp = Temps[num])
        timhhh += time.time()-time2

        if unidentified_lines:
            Gco = set_tot['all'].load_singlePT_from_file(spectral_grid)
            pop = 1 / Q_part
            abs_coeff += Gco['absorption']*pop
            abs_coeff -= Gco['ind_emission']*pop
            emi_coeff += Gco['sp_emission']*pop
        else:
            for lev in isomolec.levels:
                time2 = time.time()
                levello = getattr(isomolec, lev)
                if LTE:
                    vibt = Temps[num]
                else:
                    vibt = levello.local_vibtemp[num]
                #Gco = levello.Gcoeffs[num]
                time1 = time.time()
                Gco = set_tot[lev].load_singlePT_from_file(spectral_grid)
                timaaa += time.time()-time1
                #for key, val in zip(Gco.keys(), Gco.values()):
                    #print('iiiii make_abs iiiiii {} {} {}'.format(key, np.max(val.spectrum),np.min(val.spectrum)))

                pop = spcl.Boltz_ratio_nodeg(levello.energy, vibt) / Q_part
                timhhh += time.time()-time2
                time1 = time.time()
                # print(num, Temps[num], lev, vibt, Q_part, pop)
                # print(Gco['absorption'].integrate(), Gco['ind_emission'].integrate(), Gco['sp_emission'].integrate())
                abs_coeff += Gco['absorption']*pop
                abs_coeff -= Gco['ind_emission']*pop
                emi_coeff += Gco['sp_emission']*pop
                timuuu += time.time()-time1

                if track_levels is not None:
                    if lev in track_levels:
                        abs_coeff_level[lev] += Gco['absorption']*pop
                        abs_coeff_level[lev] -= Gco['ind_emission']*pop
                        emi_coeff_level[lev] += Gco['sp_emission']*pop

            # print(num, abs_coeff.integrate(), emi_coeff.integrate())

        if not store_in_memory:
            abs_coeffs.add_set(abs_coeff)
            emi_coeffs.add_set(emi_coeff)
            if track_levels is not None:
                for lev in track_levels:
                    emi_coeffs_tracked[lev].add_set(emi_coeff_level[lev])
                    abs_coeffs_tracked[lev].add_set(emi_coeff_level[lev])
        else:
            time1 = time.time()
            abs_coeffs.add_dump(abs_coeff)
            emi_coeffs.add_dump(emi_coeff)
            timaaa2 += time.time()-time1
            if track_levels is not None:
                for lev in track_levels:
                    emi_coeffs_tracked[lev].add_dump(emi_coeff_level[lev])
                    abs_coeffs_tracked[lev].add_dump(emi_coeff_level[lev])

    print('     -  make abs part 2: {:5.1f} s'.format((time.time()-time0)))
    time0 = time.time()

    print('      -   make_abs: LUT interp   ->   {:5.1f} s'.format(timooo))
    print('      -   make_abs: add   ->   {:5.2f} s'.format(timuuu))
    print('      -   make_abs: prima di add to spect   ->   {:5.1f} s'.format(timhhh))
    print('      -   make_abs: reading   ->   {:5.1f} s'.format(timaaa))
    print('      -   make_abs: writing   ->   {:5.1f} s'.format(timaaa2))

    if store_in_memory:
        abs_coeffs.finalize_IO()
        emi_coeffs.finalize_IO()
        if track_levels is not None:
            for lev in track_levels:
                emi_coeffs_tracked[lev].finalize_IO()
                abs_coeffs_tracked[lev].finalize_IO()

    print('     -  make abs part 3: {:5.1f} s'.format((time.time()-time0)))
    time0 = time.time()

    if track_levels is None:
        return abs_coeffs, emi_coeffs
    else:
        return abs_coeffs, emi_coeffs, emi_coeffs_tracked, abs_coeffs_tracked


def make_abscoeff_LUTS_fast(spectral_grid, isomolec, Temps, Press, LTE = True, tagLOS = None, allLUTs = None, cartDROP = None, store_in_memory = False, track_levels = None, time_control = False):
    """
    Works with compressed grid, keeps everything in memory.

    IMPORTANT: since the grid can be irregular, the same grid is required in input and in the LUTS.

    Another function in this module splits and compresses the grid in input and create the corresponding luts.
    """

    time0 = time.time()

    try:
        len(Press)
        len(Temps)
    except:
        Press = [Press]
        Temps = [Temps]

    if cartDROP is None:
        cartDROP = 'stuff_'+date_stamp()
        if not os.path.exists(cartDROP):
            os.mkdir(cartDROP)
        cartDROP += '/'

    LUTs = allLUTs[(isomolec.mol_name, isomolec.iso)]
    if LUTs is None:
        # Se sono fuori dal range spettrale di LUT leggo None
        if track_levels is None:
            return None, None
        else:
            return None, None, None, None

    if tagLOS is None:
        tagLOS = 'LOS'
    tagg = tagLOS+'_mol_{}_iso_{}'.format(isomolec.mol, isomolec.iso)
    abs_coeffs = AbsSetLOS(cartDROP+'abscoeff_'+tagg+'.pic', spectral_grid = spectral_grid)
    emi_coeffs = AbsSetLOS(cartDROP+'emicoeff_'+tagg+'.pic', spectral_grid = spectral_grid)
    if track_levels is not None:
        emi_coeffs_tracked = dict()
        abs_coeffs_tracked = dict()
        for lev in track_levels:
            tagg = tagLOS+'_mol_{}_iso_{}_{}'.format(isomolec.mol, isomolec.iso, lev)
            emi_coeffs_tracked[lev] = AbsSetLOS(cartDROP+'tracklevel_emicoeff_'+tagg+'.pic', spectral_grid = spectral_grid)
            abs_coeffs_tracked[lev] = AbsSetLOS(cartDROP+'tracklevel_abscoeff_'+tagg+'.pic', spectral_grid = spectral_grid)
    if store_in_memory:
        abs_coeffs.prepare_export()
        emi_coeffs.prepare_export()
        if track_levels is not None:
            for lev in track_levels:
                emi_coeffs_tracked[lev].prepare_export()
                abs_coeffs_tracked[lev].prepare_export()

    unidentified_lines = False
    if len(isomolec.levels) == 0:
        unidentified_lines = True
    #    print('acazuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu')

    if time_control: print('     -  make abs part 1: {:5.1f} s'.format((time.time()-time0)))
    time0 = time.time()

    timewrite = 0.0
    timecalc = 0.0
    timeadd = 0.0

    spe_zero = np.zeros(len(spectral_grid.grid), dtype = float)

    for Pres, Temp, num in zip(Press, Temps, range(len(Temps))):
        #print('oyeeeeeeeeeee ', num)
        abs_coeff = spcl.SpectralObject(spe_zero, spectral_grid, link_grid = True)
        emi_coeff = spcl.SpectralObject(spe_zero, spectral_grid, link_grid = True)
        if track_levels is not None:
            emi_coeff_level = dict()
            abs_coeff_level = dict()
            for lev in track_levels:
                emi_coeff_level[lev] = spcl.SpectralObject(spe_zero, spectral_grid, link_grid = True)
                abs_coeff_level[lev] = spcl.SpectralObject(spe_zero, spectral_grid, link_grid = True)

        # THIS IS WITH LTE PARTITION FUNCTION!!
        Q_part = spcl.CalcPartitionSum(isomolec.mol, isomolec.iso, temp = Temps[num])

        if unidentified_lines:
            time2 = time.time()
            Gco = LUTs.sets['all'].calculate(Pres, Temp)
            timecalc += time.time()-time2
            pop = 1 / Q_part
            time2 = time.time()
            if Gco['absorption'] is not None:
                abs_coeff += Gco['absorption']*pop
            if Gco['ind_emission'] is not None:
                abs_coeff -= Gco['ind_emission']*pop
            if Gco['sp_emission'] is not None:
                emi_coeff += Gco['sp_emission']*pop
            timeadd += time.time()-time2
        else:
            for lev in isomolec.levels:
                time2 = time.time()
                levello = getattr(isomolec, lev)
                if LTE:
                    vibt = Temps[num]
                else:
                    vibt = levello.local_vibtemp[num]

                time2 = time.time()
                ok, lev_lut = LUTs.find_lev(levello.lev_string)
                Gco = LUTs.sets[lev_lut].calculate(Pres, Temp)
                timecalc += time.time()-time2

                pop = spcl.Boltz_ratio_nodeg(levello.energy, vibt) / Q_part
                #print('{:4d} {:10s} {:10s} {:12.2f} {:15s} {:12.3f}'.format(num, lev, lev_lut, levello.energy, levello.lev_string, vibt))
                time2 = time.time()
                if Gco['absorption'] is not None:
                    abs_coeff += Gco['absorption']*pop
                if Gco['ind_emission'] is not None:
                    abs_coeff -= Gco['ind_emission']*pop
                if Gco['sp_emission'] is not None:
                    emi_coeff += Gco['sp_emission']*pop
                timeadd += time.time()-time2

                if track_levels is not None:
                    if lev in track_levels:
                        if Gco['absorption'] is not None:
                            abs_coeff_level[lev] += Gco['absorption']*pop
                        if Gco['ind_emission'] is not None:
                            abs_coeff_level[lev] -= Gco['ind_emission']*pop
                        if Gco['sp_emission'] is not None:
                            emi_coeff_level[lev] += Gco['sp_emission']*pop

        if not store_in_memory:
            abs_coeffs.add_set(abs_coeff)
            emi_coeffs.add_set(emi_coeff)
            if track_levels is not None:
                for lev in track_levels:
                    emi_coeffs_tracked[lev].add_set(emi_coeff_level[lev])
                    abs_coeffs_tracked[lev].add_set(emi_coeff_level[lev])
        else:
            time1 = time.time()
            abs_coeffs.add_dump(abs_coeff)
            emi_coeffs.add_dump(emi_coeff)
            timewrite += time.time()-time1
            if track_levels is not None:
                for lev in track_levels:
                    emi_coeffs_tracked[lev].add_dump(emi_coeff_level[lev])
                    abs_coeffs_tracked[lev].add_dump(emi_coeff_level[lev])

    if time_control: print('     -  make abs part 2: {:5.1f} s'.format((time.time()-time0)))
    time0 = time.time()


    if store_in_memory:
        time2 = time.time()
        abs_coeffs.finalize_IO()
        emi_coeffs.finalize_IO()
        timewrite += time.time()-time2
        if track_levels is not None:
            for lev in track_levels:
                emi_coeffs_tracked[lev].finalize_IO()
                abs_coeffs_tracked[lev].finalize_IO()

    if time_control: print('      -   make_abs: LUT interp   ->   {:5.1f} s'.format(timecalc))
    if time_control: print('      -   make_abs: add coeffs   ->   {:5.1f} s'.format(timeadd))
    if time_control: print('      -   make_abs: writing   ->   {:5.1f} s'.format(timewrite))

    if track_levels is None:
        return abs_coeffs, emi_coeffs
    else:
        return abs_coeffs, emi_coeffs, emi_coeffs_tracked, abs_coeffs_tracked



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


def comppix_to_pixels(comppix, nomefilebands, nomefilenoise):
    names = comppix[0].dtype.names

    wn_range = [min(comppix[0]['wl']), max(comppix[0]['wl'])]

    bands = sbm.read_bands(nomefilebands, wn_range = wn_range)
    noise = sbm.read_noise(nomefilenoise, wn_range = wn_range)

    nameconv = ['CUBO', 'YEAR', 'DIST', 'LAT', 'LON', 'SZA', 'PHANG', 'ALT', 'SSLAT', 'SSLON', 'OBSLAT', 'OBSLON', 'PIXELROT']
    nomiok = ['cube','year','dist','limb_tg_lat','limb_tg_lon','limb_tg_sza','phase_ang','limb_tg_alt','sub_solar_lat','sub_solar_lon','sub_obs_lat','sub_obs_lon','pixel_rot']
    dict_name = dict(zip(nameconv,nomiok))
    pix_set = []
    for pix in comppix:
        vals = dict()
        for nam,nunam in zip(nameconv, nomiok):
            coso = pix[nam]
            vals[nunam] = coso[0]

        gri = spcl.SpectralGrid(pix['wl'], units = 'nm')
        spet = spcl.SpectralIntensity(pix['spet'], gri, units = 'Wm2')
        spet.add_mask(pix['bbl'])
        spet.add_bands(bands)
        spet.add_noise(noise)
        vals['observation'] = spet

        pixo = sbm.VIMSPixel(vals.keys(), vals.values())
        pix_set.append(pixo)

    return pix_set


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


def inversion(inputs, planet, lines, bayes_set, pixels, wn_range = None, chi_threshold = 0.01, max_it = 10, lambda_LM = 0.1, L1_reg = False, radtran_opt = dict(), useLUTs = True, debugfile = None, save_hires = False, LUTopt = dict(), test = False, g3D = False):
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
        if save_hires:
            hiresfile = open(cartOUT+'hires_radtran.pic', 'wb')

        for pix, num in zip(pixels, range(len(pixels))):
            radtrans = []
            derivs = []
            print('pixel {} at tangent alt: {}'.format(num, pix.limb_tg_alt))
            time1 = time.time()

            linea_low = pix.low_LOS()
            linea0 = pix.LOS()
            linea_up = pix.up_LOS()
            if g3D:
                ssp = pix.sub_solar_point()
            else:
                ssp = None

            # Using 1D interpolation scheme for FOV integration
            # x_FOV = np.array([linea_low.tangent_altitude, linea_0.tangent_altitude, linea_up.tangent_altitude])
            zeroder = pix.observation*0.0

            linea_low.details()
            radtran_low = linea_low.radtran(wn_range, planet, lines, cartLUTs = inputs['cart_LUTS'], cartDROP = inputs['out_dir'], calc_derivatives = True, bayes_set = bayes_set, LUTS = LUTS, useLUTs = useLUTs, radtran_opt = radtran_opt, g3D = g3D, sub_solar_point = ssp)
            radtrans.append(radtran_low)
            deriv_ok = []
            for par in bayes_set.params():
                if par.not_involved:
                    deriv_ok.append(zeroder)
                    continue
                towl = par.hires_deriv
                lowres = towl.hires_to_lowres(pix.observation, spectral_widths = pix.observation.bands.spectrum)
                deriv_ok.append(lowres)
            derivs.append(deriv_ok)

            linea0.details()
            radtran = linea0.radtran(wn_range, planet, lines, cartLUTs = inputs['cart_LUTS'], cartDROP = inputs['out_dir'], calc_derivatives = True, bayes_set = bayes_set, LUTS = LUTS, useLUTs = useLUTs, radtran_opt = radtran_opt, g3D = g3D, sub_solar_point = ssp)
            radtrans.append(radtran)
            deriv_ok = []
            for par in bayes_set.params():
                if par.not_involved:
                    deriv_ok.append(zeroder)
                    continue
                towl = par.hires_deriv
                lowres = towl.hires_to_lowres(pix.observation, spectral_widths = pix.observation.bands.spectrum)
                deriv_ok.append(lowres)
            derivs.append(deriv_ok)

            if save_hires:
                pickle.dump([num, pix.limb_tg_alt, radtran], hiresfile)

            linea_up.details()
            radtran_up = linea_up.radtran(wn_range, planet, lines, cartLUTs = inputs['cart_LUTS'], cartDROP = inputs['out_dir'], calc_derivatives = True, bayes_set = bayes_set, LUTS = LUTS, useLUTs = useLUTs, radtran_opt = radtran_opt, g3D = g3D, sub_solar_point = ssp)
            radtrans.append(radtran_up)
            deriv_ok = []
            for par in bayes_set.params():
                if par.not_involved:
                    deriv_ok.append(zeroder)
                    continue
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

            print('pixel done in {:5.1f} min'.format((time.time()-time1)/60.))

        if save_hires:
            hiresfile.close()

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

        for par in bayes_set.params():
            par.hires_deriv = None
        if debugfile is not None:
            pickle.dump([num_it, obs, sims, bayes_set], debugfile)
        # Update the VMRs of the retrieved gases with the new values
        for gas in bayes_set.sets.keys():
            planet.gases[gas].add_clim(bayes_set.sets[gas].profile())

    return


def inversion_fast_limb(inputs, planet, lines, bayes_set, pixels, wn_range = None, sp_gri = None, chi_threshold = 0.01, max_it = 10, lambda_LM = 0.1, L1_reg = False, radtran_opt = dict(), debugfile = None, save_hires = True, LUTopt = dict(), test = False, use_tangent_sza = False, group_observations = False, nome_inv = '1', solo_simulation = False, invert_LOS_direction = False):
    """
    Main routine for retrieval. Fast version.
    """

    lines = check_lines_mols(lines, planet.gases.values())
    print('Begin inversion..')
    cartOUT = inputs['out_dir']

    pixels.sort(key = lambda x: x.limb_tg_alt)

    if sp_gri is None:
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

    alt_tg = []
    lat_tg = []
    pres_max = []
    tg_points = []
    for pix in pixels:
        linea_0 = pix.low_LOS()
        tg_point = linea_0.get_tangent_point()
        tg_points.append(tg_point)
        if 'max_pres' not in LUTopt:
            pres_max.append(planet.atmosphere.calc(tg_point, profname = 'pres'))
        alt_tg.append(tg_point.Spherical()[2])
        lat_tg.append(tg_point.Spherical()[0])

    if 'max_pres' not in LUTopt:
        pres_max = max(pres_max)
        LUTopt['max_pres'] = pres_max
    PTcoup_needed = calc_PT_couples_atmosphere(lines, planet.gases.values(), planet.atmosphere, **LUTopt)

    LUTS = check_and_build_allluts(inputs, sp_gri, lines, planet.gases.values(), PTcouples = PTcoup_needed, LUTopt = LUTopt)
    n_lut_tot = len(PTcoup_needed)
    print('{} PT couples needed'.format(n_lut_tot))

    sims = []
    obs = [pix.observation for pix in pixels]
    masks = [pix.observation.mask for pix in pixels]
    noise = [pix.observation.noise for pix in pixels]

    # FASE 0: decidere in quanti pezzi splittare le LUTS. le splitto. se tengo tutto raddoppia la dimensione su disco. butto via i Gcoeff nulli.
    n_threads = inputs['n_threads']
    LUTS, n_split, sp_grids = split_and_compress_LUTS(sp_gri, LUTS, inputs['cart_LUTS'], n_threads, n_split = inputs['n_split'])

    print(n_split)
    for sp_gr, u in zip(sp_grids, range(n_split)):
        print('split {}: {:5.1f} - {:5.1f}'.format(u,sp_gr.grid.min(),sp_gr.grid.max()))

    # lancio calc_radtran_steps e decido quante los calcolare davvero
    # ho una lista di los fittizie in uscita
    # se i pixels appartengono a cubi diversi o intersecano il pianeta in lats diverse devo separare in diversi gruppi
    # questo discorso è un po' pericoloso ad esempio per le derivate. bisogna stare attenti.
    # ad esempio che fare delle linee che intersecano più lat box? ehehehhe casino deh. NON esistono. Le metto o di qua o di là in base alla lat di tangenza. Fine.

    if group_observations:
        print('Group observations')
        #sim_LOSs = group_observations(pixels)
        sim_LOSs = [pix.LOS() for pix in pixels]
        first_los = pixels[0].low_LOS()
        if first_los.get_tangent_altitude() > pixels[0].limb_tg_alt:
            first_los = pixels[0].up_LOS()
        sim_LOSs.insert(0, first_los)
        last_los = pixels[-1].up_LOS()
        if last_los.get_tangent_altitude() < pixels[-1].limb_tg_alt:
            last_los = pixels[-1].low_LOS()
        sim_LOSs.append(last_los)
        #sim_LOSs = sim_LOSs[::-1]

        fszas = [pix.limb_tg_sza for pix in pixels]
        fszas.insert(0, pixels[0].limb_tg_sza)
        fszas.append(pixels[-1].limb_tg_sza)

        alts_sim = [los.get_tangent_altitude() for los in sim_LOSs]
        print(alts_sim)

        ssps = [pix.sub_solar_point() for pix in pixels]
        ssps.insert(0, pixels[0].sub_solar_point())
        ssps.append(pixels[-1].sub_solar_point())

        # ordering
        ordlos = np.argsort(np.array(alts_sim))
        sim_LOSs = list(np.array(sim_LOSs)[ordlos])
        ssps = list(np.array(ssps)[ordlos])
        alts_sim = list(np.sort(np.array(alts_sim)))
    else:
        sim_LOSs = []
        ssps = []
        fszas = []
        for pix in pixels:
            sim_LOSs.append(pix.low_LOS())
            sim_LOSs.append(pix.LOS())
            sim_LOSs.append(pix.up_LOS())
            ssps += 3*[pix.sub_solar_point()]
            fszas += 3*[pix.limb_tg_sza]

        alts_sim = [los.get_tangent_altitude() for los in sim_LOSs]

    for num in range(len(sim_LOSs)):
        sim_LOSs[num].tag = 'LOS{:03d}'.format(num)

    print(alts_sim)
    for pix in pixels:
        print(pix.limb_tg_lat, pix.limb_tg_lon, pix.limb_tg_alt)
    #sys.exit()

    observ_sample = pixels[0].observation
    spectral_widths = pixels[0].observation.bands.spectrum
    zeroder = observ_sample*0.0

    time0 = time.time()
    for num_it in range(max_it):
        print('we are at iteration: {}'.format(num_it))
        if save_hires:
            hiresfile = open(cartOUT+'hires_radtran_{}.pic'.format(nome_inv), 'wb')
            lowresfile = open(cartOUT+'lowres_radtran_{}.pic'.format(nome_inv), 'wb')

        ntot = 0
        nlos = len(sim_LOSs)
        proc_sims = []
        time03 = time.time()
        while ntot < nlos:
            losos = sim_LOSs[ntot:ntot+n_threads]
            sspsos = ssps[ntot:ntot+n_threads]
            fszasos = fszas[ntot:ntot+n_threads]
            n_proc = len(losos)
            time01 = time.time()
            print('Lancio {} procs'.format(n_proc))
            processi = []
            coda = []
            outputs = []
            for los, ssp, fsza, i in zip(losos, sspsos, fszasos, range(n_proc)):
                if invert_LOS_direction:
                    los.calc_atm_intersections(planet, LOS_order = 'photon')
                else:
                    los.calc_atm_intersections(planet)
                if use_tangent_sza:
                    los.szas = np.ones(len(los.intersections))*fsza
                else:
                    los.calc_SZA_along_los(planet, ssp)

                coda.append(Queue())
                args = (planet, lines)
                kwargs = {'queue': coda[i], 'calc_derivatives' : True, 'bayes_set' : bayes_set}
                kwargs.update(radtran_opt)
                processi.append(Process(target=los.calc_radtran_steps, args=args, kwargs=kwargs))
                processi[i].start()

            for i in range(n_proc):
                outputs.append(coda[i].get())

            for i in range(n_proc):
                processi[i].join()

            print('All processes ended')
            time02 = time.time()

            for los, out in zip(losos, outputs):
                if los.tag == out.tag:
                    proc_sims.append(out)
                else:
                    raise NameError('{} is not {}'.format(los.tag, out.tag))
                print(los.tag, len(out.radtran_steps['step']))
                #print(los.tag, len(los.radtran_steps['step']))
                #print(len(sim_LOSs[los.tag].radtran_steps['step']))

            ntot += n_proc

        print('calc_radtran_steps done in {} min'.format((time.time()-time03)/60.))
        sim_LOSs = proc_sims

        # fillo = open('calc_radtran_steps.pic', 'w')
        # for los, ssp in zip(sim_LOSs, ssps):
            # los.calc_SZA_along_los(planet, ssp)
            # los.calc_radtran_steps(planet, lines, calc_derivatives = True, bayes_set = bayes_set, **radtran_opt)
        # pickle.dump(sim_LOSs, fillo)
        # fillo.close()
        # fillo = open('calc_radtran_steps.pic', 'r')
        # sim_LOSs = pickle.load(fillo)
        # sim_LOSs = sim_LOSs[::-1]

        for num in range(len(sim_LOSs)):
            sim_LOSs[num].tag = 'LOS{:02d}'.format(num)
            print('{} - {} steps'.format(sim_LOSs[num].tag, len(sim_LOSs[num].radtran_steps['step'])))
        #sim_LOSs = sim_LOSs[::-1]

        radtrans = dict()
        derivs = dict()

        time00 = time.time()
        for nsp, sp_grid_split in zip(range(n_split), sp_grids):
            print('Split # {} of {}: {:5.1f} to {:5.1f} cm-1'.format(nsp, n_split, sp_grid_split.grid.min(), sp_grid_split.grid.max()))

            for nam in LUTS:
                LUTS[nam].load_split(nsp)

            time0 = time.time()
            hi_res = dict()
            ntot = 0
            nlos = len(sim_LOSs)

            time01 = time.time()
            while ntot < nlos:
                losos = sim_LOSs[ntot:ntot+n_threads]
                n_proc = len(losos)
                time01 = time.time()
                print('Lancio {} procs'.format(n_proc))
                processi = []
                coda = []
                outputs = []
                for los, i in zip(losos, range(n_proc)):
                    coda.append(Queue())
                    args = (sp_grid_split, planet)
                    kwargs = {'queue': coda[i], 'cartLUTs': inputs['cart_LUTS'], 'cartDROP' : inputs['out_dir'], 'calc_derivatives' : True, 'bayes_set' : bayes_set, 'LUTS' : LUTS, 'radtran_opt' : radtran_opt, 'store_abscoeff': False}
                    processi.append(Process(target=los.radtran_fast, args=args, kwargs=kwargs))
                    processi[i].start()

                for i in range(n_proc):
                    outputs.append(coda[i].get())

                for i in range(n_proc):
                    processi[i].join()

                print('All processes ended')
                time02 = time.time()

                for los, out in zip(losos,outputs):
                    radtran = out[0]
                    retsetmod = out[2]
                    hi_res[los.tag] = out[0]

                    # coda.append(Queue())
                    # args = (old_lowres, observ_sample)
                    # kwargs = {'queue': coda[i], 'spectral_widths': spectral_widths, 'cartDROP' : inputs['out_dir'], 'calc_derivatives' : True, 'bayes_set' : bayes_set, 'LUTS' : LUTS, 'radtran_opt' : radtran_opt, 'g3D' : g3D, 'sub_solar_point' : ssp, 'store_abscoeff': False}
                    # processi.append(Process(target=out_to_lowres, args=args, kwargs=kwargs))
                    # processi[i].start()

                    if los.tag in radtrans:
                        radtrans[los.tag] += radtran.hires_to_lowres(observ_sample, spectral_widths = spectral_widths)
                    else:
                        radtrans[los.tag] = radtran.hires_to_lowres(observ_sample, spectral_widths = spectral_widths)

                    for par, par_mod in zip(bayes_set.params(), retsetmod.params()):
                        # if par_mod.not_involved:
                        #     derivva = zeroder
                        #print(los.tag, par.nameset, par.key, los.involved_retparams[(par.nameset, par.key)])
                        if not los.involved_retparams[(par.nameset, par.key)]:
                            derivva = zeroder
                        else:
                            derivva = par_mod.hires_deriv.hires_to_lowres(observ_sample, spectral_widths = spectral_widths)
                            par.set_used()

                        kiave = (los.tag, par.nameset, par.key)
                        if kiave in derivs:
                            derivs[kiave] += derivva
                        else:
                            derivs[kiave] = derivva

                ntot += n_proc
                print('tempo tot: {:5.1f} min'.format((time.time()-time01)/60.))
                print('tempo single proc: {:5.1f} min'.format((time.time()-time02)/60.))

                print('split {}, {} to {} done'.format(nsp, losos[0].tag, losos[-1].tag))

            if save_hires:
                pickle.dump([nsp,hi_res], hiresfile)
            print('split {}: {} LOS done in {:5.1f} min'.format(nsp, len(sim_LOSs), (time.time()-time0)/60.))

        if save_hires:
            hiresfile.close()
        print('Tempo los sim all wn_range: {} min'.format((time.time()-time00)/60.))
        time0 = time.time()
        if solo_simulation:
            print('Solo simulation. returning..')
            return None

        sims = []
        if group_observations:
            radtrans_list = [radtrans[los.tag] for los in sim_LOSs]
            radtran_spline = make_radtran_spline(alts_sim, radtrans_list)
            deriv_splines = dict()
            for par in bayes_set.params():
                deriv_par = [derivs[(los.tag, par.nameset, par.key)] for los in sim_LOSs]
                deriv_splines[(par.nameset, par.key)] = make_radtran_spline(alts_sim, deriv_par)

            for pix, num in zip(pixels, range(len(pixels))):
                lineas = [pix.low_LOS(), pix.LOS(), pix.up_LOS()]

                alts_pix = np.array([lin.get_tangent_point().Spherical()[2] for lin in lineas])
                intens_FOV = np.array([radtran_spline(al) for al in alts_pix])
                sim_FOV_ok = FOV_integr_1D(intens_FOV, pix.pixel_rot)
                sims.append(sim_FOV_ok)

                derivs_FOV = []
                derivs_FOV.append(deriv_splines)
                for par in bayes_set.params():
                    ders = np.array([deriv_splines[(par.nameset, par.key)](al) for al in alts_pix])
                    der_FOV_ok = FOV_integr_1D(ders, pix.pixel_rot)
                    par.store_deriv(der_FOV_ok, num = num)
        else:
            for pix, num in zip(pixels, range(len(pixels))):
                loss = sim_LOSs[3*num:3*(num+1)]
                intens_FOV = np.array([radtrans[lo.tag] for lo in loss])
                sim_FOV_ok = FOV_integr_1D(intens_FOV, pix.pixel_rot)
                sims.append(sim_FOV_ok)

                for par in bayes_set.params():
                    ders = np.array([derivs[(los.tag, par.nameset, par.key)] for los in loss])
                    der_FOV_ok = FOV_integr_1D(ders, pix.pixel_rot)
                    par.store_deriv(der_FOV_ok, num = num)

        print('FOV interp done in {:5.1f} min'.format((time.time()-time0)/60.))

        #INVERSIONE
        chi = chicalc(obs, sims, noise, masks, bayes_set.n_used_par())

        for par in bayes_set.params():
            par.hires_deriv = None
        if debugfile is not None:
            if num_it == 0:
                pickle.dump([pixels, sim_LOSs], debugfile)
            pickle.dump([num_it, obs, sims, bayes_set, radtrans, derivs], debugfile)

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

        # Update the VMRs of the retrieved gases with the new values
        for gas in bayes_set.sets.keys():
            planet.gases[gas].add_clim(bayes_set.sets[gas].profile())

    return


def radtrans(inputs, planet, lines, pixels, wn_range = None, sp_gri = None, radtran_opt = dict(), save_hires = True, LUTopt = dict(), test = False, use_tangent_sza = False, group_observations = False, nome_inv = '1'):
    """
    Main routine for retrieval. Fast version.
    """

    lines = check_lines_mols(lines, planet.gases.values())
    print('Begin radtrans..')
    cartOUT = inputs['out_dir']

    pixels.sort(key = lambda x: x.limb_tg_alt)

    if sp_gri is None:
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

    alt_tg = []
    lat_tg = []
    pres_max = []
    tg_points = []
    for pix in pixels:
        linea_0 = pix.low_LOS()
        tg_point = linea_0.get_tangent_point()
        tg_points.append(tg_point)
        if 'max_pres' not in LUTopt:
            pres_max.append(planet.atmosphere.calc(tg_point, profname = 'pres'))
        alt_tg.append(tg_point.Spherical()[2])
        lat_tg.append(tg_point.Spherical()[0])

    if 'max_pres' not in LUTopt:
        pres_max = max(pres_max)
        LUTopt['max_pres'] = pres_max
    PTcoup_needed = calc_PT_couples_atmosphere(lines, planet.gases.values(), planet.atmosphere, **LUTopt)

    LUTS = check_and_build_allluts(inputs, sp_gri, lines, planet.gases.values(), PTcouples = PTcoup_needed, LUTopt = LUTopt)
    n_lut_tot = len(PTcoup_needed)
    print('{} PT couples needed'.format(n_lut_tot))

    sims = []
    obs = [pix.observation for pix in pixels]
    masks = [pix.observation.mask for pix in pixels]
    noise = [pix.observation.noise for pix in pixels]

    # FASE 0: decidere in quanti pezzi splittare le LUTS. le splitto. se tengo tutto raddoppia la dimensione su disco. butto via i Gcoeff nulli.
    n_threads = inputs['n_threads']
    LUTS, n_split, sp_grids = split_and_compress_LUTS(sp_gri, LUTS, inputs['cart_LUTS'], n_threads, n_split = inputs['n_split'])

    print(n_split)
    for sp_gr, u in zip(sp_grids, range(n_split)):
        print('split {}: {:5.1f} - {:5.1f}'.format(u,sp_gr.grid.min(),sp_gr.grid.max()))

    # lancio calc_radtran_steps e decido quante los calcolare davvero
    # ho una lista di los fittizie in uscita
    # se i pixels appartengono a cubi diversi o intersecano il pianeta in lats diverse devo separare in diversi gruppi
    # questo discorso è un po' pericoloso ad esempio per le derivate. bisogna stare attenti.
    # ad esempio che fare delle linee che intersecano più lat box? ehehehhe casino deh. NON esistono. Le metto o di qua o di là in base alla lat di tangenza. Fine.

    if group_observations:
        print('Group observations')
        #sim_LOSs = group_observations(pixels)
        sim_LOSs = [pix.LOS() for pix in pixels]
        first_los = pixels[0].low_LOS()
        if first_los.get_tangent_altitude() > pixels[0].limb_tg_alt:
            first_los = pixels[0].up_LOS()
        sim_LOSs.insert(0, first_los)
        last_los = pixels[-1].up_LOS()
        if last_los.get_tangent_altitude() < pixels[-1].limb_tg_alt:
            last_los = pixels[-1].low_LOS()
        sim_LOSs.append(last_los)

        fszas = [pix.limb_tg_sza for pix in pixels]
        fszas.insert(0, pixels[0].limb_tg_sza)
        fszas.append(pixels[-1].limb_tg_sza)

        alts_sim = [los.get_tangent_point().Spherical()[2] for los in sim_LOSs]
        print(alts_sim)

        ssps = [pix.sub_solar_point() for pix in pixels]
        ssps.insert(0, pixels[0].sub_solar_point())
        ssps.append(pixels[-1].sub_solar_point())

        # ordering
        ordlos = np.argsort(np.array(alts_sim))
        sim_LOSs = list(np.array(sim_LOSs)[ordlos])
        ssps = list(np.array(ssps)[ordlos])
        alts_sim = list(np.sort(np.array(alts_sim)))
    else:
        sim_LOSs = []
        ssps = []
        fszas = []
        for pix in pixels:
            sim_LOSs.append(pix.low_LOS())
            sim_LOSs.append(pix.LOS())
            sim_LOSs.append(pix.up_LOS())
            ssps += 3*[pix.sub_solar_point()]
            fszas += 3*[pix.limb_tg_sza]

        alts_sim = [los.get_tangent_point().Spherical()[2] for los in sim_LOSs]

    for num in range(len(sim_LOSs)):
        sim_LOSs[num].tag = 'LOS{:03d}'.format(num)

    print(alts_sim)
    for pix in pixels:
        print(pix.limb_tg_lat, pix.limb_tg_lon, pix.limb_tg_alt)
    #sys.exit()

    observ_sample = pixels[0].observation
    spectral_widths = pixels[0].observation.bands.spectrum
    zeroder = observ_sample*0.0

    time0 = time.time()
    print('simulation of the spectra')
    if save_hires:
        hiresfile = open(cartOUT+'hires_radtran_{}.pic'.format(nome_inv), 'wb')
        lowresfile = open(cartOUT+'lowres_radtran_{}.pic'.format(nome_inv), 'wb')

    ntot = 0
    nlos = len(sim_LOSs)
    proc_sims = []
    time03 = time.time()
    while ntot < nlos:
        losos = sim_LOSs[ntot:ntot+n_threads]
        sspsos = ssps[ntot:ntot+n_threads]
        fszasos = fszas[ntot:ntot+n_threads]
        n_proc = len(losos)
        time01 = time.time()
        print('Lancio {} procs'.format(n_proc))
        processi = []
        coda = []
        outputs = []
        for los, ssp, fsza, i in zip(losos, sspsos, fszasos, range(n_proc)):
            los.calc_atm_intersections(planet)
            if use_tangent_sza:
                los.szas = np.ones(len(los.intersections))*fsza
            else:
                los.calc_SZA_along_los(planet, ssp)

            coda.append(Queue())
            args = (planet, lines)
            kwargs = {'queue': coda[i], 'calc_derivatives' : False}
            kwargs.update(radtran_opt)
            processi.append(Process(target=los.calc_radtran_steps, args=args, kwargs=kwargs))
            processi[i].start()

        for i in range(n_proc):
            outputs.append(coda[i].get())

        for i in range(n_proc):
            processi[i].join()

        print('All processes ended')
        time02 = time.time()

        for los, out in zip(losos, outputs):
            if los.tag == out.tag:
                proc_sims.append(out)
            else:
                raise NameError('{} is not {}'.format(los.tag, out.tag))
            print(los.tag, len(out.radtran_steps['step']))
            #print(los.tag, len(los.radtran_steps['step']))
            #print(len(sim_LOSs[los.tag].radtran_steps['step']))

        ntot += n_proc

    print('calc_radtran_steps done in {} min'.format((time.time()-time03)/60.))
    sim_LOSs = proc_sims

    for num in range(len(sim_LOSs)):
        sim_LOSs[num].tag = 'LOS{:02d}'.format(num)
        print('{} - {} steps'.format(sim_LOSs[num].tag, len(sim_LOSs[num].radtran_steps['step'])))

    radtrans = dict()
    derivs = dict()
    single_rads = dict()
    for gas in planet.gases:
        for iso in planet.gases[gas].all_iso:
            single_rads[(gas, iso)] = dict()

    time00 = time.time()
    for nsp, sp_grid_split in zip(range(n_split), sp_grids):
        print('Split # {} of {}: {:5.1f} to {:5.1f} cm-1'.format(nsp, n_split, sp_grid_split.grid.min(), sp_grid_split.grid.max()))

        for nam in LUTS:
            LUTS[nam].load_split(nsp)

        time0 = time.time()
        hi_res = dict()
        ntot = 0
        nlos = len(sim_LOSs)

        time01 = time.time()
        while ntot < nlos:
            losos = sim_LOSs[ntot:ntot+n_threads]
            n_proc = len(losos)
            time01 = time.time()
            print('Lancio {} procs'.format(n_proc))
            processi = []
            coda = []
            outputs = []
            for los, i in zip(losos, range(n_proc)):
                coda.append(Queue())
                args = (sp_grid_split, planet)
                kwargs = {'queue': coda[i], 'cartLUTs': inputs['cart_LUTS'], 'cartDROP' : inputs['out_dir'], 'calc_derivatives' : False, 'LUTS' : LUTS, 'radtran_opt' : radtran_opt, 'store_abscoeff': False}
                processi.append(Process(target=los.radtran_fast, args=args, kwargs=kwargs))
                processi[i].start()

            for i in range(n_proc):
                outputs.append(coda[i].get())

            for i in range(n_proc):
                processi[i].join()

            print('All processes ended')
            time02 = time.time()

            for los, out in zip(losos,outputs):
                radtran = out[0]
                single_rad = out[1]
                hi_res[los.tag] = out[0]

                # coda.append(Queue())
                # args = (old_lowres, observ_sample)
                # kwargs = {'queue': coda[i], 'spectral_widths': spectral_widths, 'cartDROP' : inputs['out_dir'], 'calc_derivatives' : True, 'bayes_set' : bayes_set, 'LUTS' : LUTS, 'radtran_opt' : radtran_opt, 'g3D' : g3D, 'sub_solar_point' : ssp, 'store_abscoeff': False}
                # processi.append(Process(target=out_to_lowres, args=args, kwargs=kwargs))
                # processi[i].start()

                if los.tag in radtrans:
                    radtrans[los.tag] += radtran.hires_to_lowres(observ_sample, spectral_widths = spectral_widths)
                else:
                    radtrans[los.tag] = radtran.hires_to_lowres(observ_sample, spectral_widths = spectral_widths)

                for gasiso in single_rads:
                    if los.tag in single_rads[gasiso]:
                        single_rads[gasiso][los.tag] += single_rad[gasiso].hires_to_lowres(observ_sample, spectral_widths = spectral_widths)
                    else:
                        single_rads[gasiso][los.tag] = single_rad[gasiso].hires_to_lowres(observ_sample, spectral_widths = spectral_widths)

            ntot += n_proc
            print('tempo tot: {:5.1f} min'.format((time.time()-time01)/60.))
            print('tempo single proc: {:5.1f} min'.format((time.time()-time02)/60.))

            print('split {}, {} to {} done'.format(nsp, losos[0].tag, losos[-1].tag))

        if save_hires:
            pickle.dump([nsp,hi_res], hiresfile)
        print('split {}: {} LOS done in {:5.1f} min'.format(nsp, len(sim_LOSs), (time.time()-time0)/60.))

    if save_hires:
        hiresfile.close()
    print('Tempo los sim all wn_range: {} min'.format((time.time()-time00)/60.))

    sims = []
    if group_observations:
        radtrans_list = [radtrans[los.tag] for los in sim_LOSs]
        radtran_spline = make_radtran_spline(alts_sim, radtrans_list)

        for pix, num in zip(pixels, range(len(pixels))):
            lineas = [pix.low_LOS(), pix.LOS(), pix.up_LOS()]

            alts_pix = np.array([lin.get_tangent_point().Spherical()[2] for lin in lineas])
            intens_FOV = np.array([radtran_spline(al) for al in alts_pix])
            sim_FOV_ok = FOV_integr_1D(intens_FOV, pix.pixel_rot)
            sims.append(sim_FOV_ok)
    else:
        for pix, num in zip(pixels, range(len(pixels))):
            loss = sim_LOSs[3*num:3*(num+1)]
            intens_FOV = np.array([radtrans[lo.tag] for lo in loss])
            sim_FOV_ok = FOV_integr_1D(intens_FOV, pix.pixel_rot)
            sims.append(sim_FOV_ok)

    if save_hires:
        pickle.dump([pixels, sims, sim_LOSs, radtrans, single_rads], lowresfile)
        lowresfile.close()

    print('FOV interp done in {:5.1f} min'.format((time.time()-time0)/60.))

    return sims, radtrans, single_rads


def group_observations(pixels, lat_limits = None, alt_grid = None):
    """
    Builds a set of LOS for the forward model. pixels are grouped depending on observation cube and lat_band. For each group, a set of LOSs to calculate the radtran in that cube is given.
    """
    cubes = np.unique([pix])

    pass


def FOV_integr_1D(radtrans, pixel_rot = 0.0):
    """
    FOV integration interpolating with a spline the spectrum between the external los.
    """
    from scipy import integrate

    pixel_rot = abs(sbm.rad(pixel_rot))

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


def make_radtran_spline(alts, radtrans):
    """
    Creates a 2Dspline function to interpolate the los contributions at arbitrary altitude given some simulated LOS radtrans at fixed altitudes.
    """
    #print(x_integ)
    alts = np.array(alts)
    spectrums = np.array([rad.spectrum for rad in radtrans])
    grid = radtrans[0].spectral_grid.grid
    radsample = radtrans[0]

    intens_spl = spline2D(alts, radtrans[0].spectral_grid.grid, spectrums, kx=2, ky=2)

    def radtran_alt(x, funzspl = intens_spl, grid = grid, radsample = radsample):
        res = [float(intens_spl(x,ww)) for ww in grid]
        res = np.array(res)
        res_spe = copy.deepcopy(radsample)
        res_spe.spectrum = res
        return res_spe

    return radtran_alt


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

    chi = np.sum(((obs_vec-sim_vec)/noi_vec)**2)
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
