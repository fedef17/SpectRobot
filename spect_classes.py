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
import scipy
import warnings
from fparts import bd_tips_2003 as Q_calc
import copy
import spect_base_module as sbm

#parameters
Rtit = 2575.0 # km
Mtit = 1.3452e23 # kg
kb = 1.38065e-19 # boltzmann constant to use with P in hPa, n in cm-3, T in K (10^-4 * Kb standard)
c_R = 8.31446 # J K-1 mol-1
c_G = 6.67408e-11 # m3 kg-1 s-2
kbc = const.k/(const.h*100*const.c) # 0.69503

#####################################
# Define for the radiative transfer in cgs - DEFINED AS IN HITRAN
h_cgs = const.physical_constants['Planck constant'][0]*1.e7 # erg s
c_cgs = const.constants.c * 1.e2 # cm s-1
k_cgs = const.physical_constants['Boltzmann constant'][0]*1.e7 # erg K-1
c2 = h_cgs * c_cgs / k_cgs

### NAMES for SpectLine object
cose = ('Mol', 'Iso', 'Freq', 'Strength', 'A_coef', 'Air_broad', 'Self_broad', 'Energy_low', 'T_dep_broad', 'P_shift','Up_lev_str', 'Lo_lev_str', 'Q_num_up', 'Q_num_lo')
cose_hit = ('Mol', 'Iso', 'Freq', 'Strength', 'A_coef', 'Air_broad', 'Self_broad', 'Energy_low', 'T_dep_broad', 'P_shift','Up_lev_str', 'Lo_lev_str', 'Q_num_up', 'Q_num_lo', 'others', 'g_up', 'g_lo')

cose_mas = ('Up_lev_id', 'Lo_lev_id', 'Up_lev', 'Lo_lev')
#############################################################################

class SpectLine(object):
    """
    Spectral line.
    """

    def __init__(self, linea, nomi = None):
        if nomi is None:
            if type(linea) is dict:
                #print(type(linea))
                nomi = linea.keys()
            elif type(linea) is np.void:
                #print(type(linea))
                nomi = linea.dtype.names
            else:
                nomi = cose
                raise ValueError('Missing names for line quantities')
        else:
            linea = dict(zip(nomi,linea))

        if nomi != cose and nomi != cose_hit:
            print('------> WARNING: NON-Standard names assigned to SpectLine attributes <------')
            print('Assigned names: {}'.format(nomi))
            print('Standard names: {}'.format(cose))

        for nome in nomi:
            setattr(self,nome,linea[nome])

        for nome in cose_mas:
            setattr(self,nome,None)

        return


    def _Print_(self, ofile = None):
        if ofile is None:
            print('{:4d}{:2d}{:8.2f}{:10.3e}{:8.2f}{:15s}{:15s}'.format(self.Mol,self.Iso,self.Freq,self.Strength,self.Energy_low,self.Up_lev_str,self.Lo_lev_str))
        else:
            ofile.write('{:4d}{:2d}{:8.2f}{:10.3e}{:8.2f}{:15s}{:15s}'.format(self.Mol,self.Iso,self.Freq,self.Strength,self.Energy_low,self.Up_lev_str,self.Lo_lev_str))
        return


    def _ShowCalc_(self, T, P = 1, nlte_ratio = 1):
        pass


    def _CalcStrength_(self, T):
        """
        Calcs Line Strength at temperature T.
        """
        return CalcStrength_at_T(self.Mol, self.Iso, self.Strength, self.Freq, self.Energy_low, T)


    def _LinkToMolec_(self,IsoMolec):
        """
        IsoMolec has to be of the corresponding type in module spect_base_module.
        Sets the values of the two attr. "Up_lev_id" and "Lo_lev_id" with the internal names of the corresponding levels in IsoMolec. So that all level quantities can be accessed via Level = getattr(IsoMolec,Up_lev_id) and using Level..
        """
        for lev in IsoMolec.levels:
            Level = getattr(IsoMolec, lev)
            if np.any(np.array([self.Up_lev_str in string for string in [Level.level]+Level.simmetry])):
                # This is the right upper level!
                self.Up_lev_id = lev
                self.Up_lev = Level
            elif np.any(np.array([self.Lo_lev_str in string for string in [Level.level]+Level.simmetry])):
                # This is the right lower level!
                self.Lo_lev_id = lev
                self.Lo_lev = Level
            else:
                continue

        return

    def _Einstein_A_to_B(self):
        """
        Calculates the Einstein B
        """


class SpectralGrid(object):
    """
    This is the spectral grid. Some useful methods (conversions, e basta..).
    """

    def __init__(self, spectral_grid, units = 'nm'):
        self.grid = copy.deepcopy(spectral_grid)
        self.units = units
        print('ciao')

        return

    def convertto_nm(self):
        if self.units == 'nm':
            return self.grid
        elif self.units == 'mum':
            grid = self.grid*1.e3
            self.grid = copy.deepcopy(grid)
            self.units = 'nm'
            return grid
        elif self.units == 'cm_1':
            grid = 1.e7/self.grid
            self.grid = copy.deepcopy(grid[::-1])
            self.units = 'nm'
            return grid[::-1]
        elif self.units == 'hz':
            grid = const.constants.c/(1.e-9*self.grid)
            self.grid = copy.deepcopy(grid[::-1])
            self.units = 'nm'
            return grid[::-1]

    def convertto_cm_1(self):
        grid = self.convertto_nm()
        grid = 1.e7/self.grid
        self.grid = copy.deepcopy(grid[::-1])
        self.units = 'cm_1'
        return grid[::-1]

    def convertto_mum(self):
        grid = self.convertto_nm()*1.e-3
        self.grid = copy.deepcopy(grid)
        self.units = 'mum'
        return grid

    def convertto_hz(self):
        grid = self.convertto_nm()
        grid = const.constants.c/(1.e-9*self.grid)
        self.grid = copy.deepcopy(grid[::-1])
        self.units = 'hz'
        return grid[::-1]



class SpectralIntensity(object):
    """
    This is the spectral intensity. Some useful methods (conversions, integration, convolution, ..).
    """

    def __init__(self, intensity, spectral_grid, direction = None, units = 'Wm2'):
        self.intensity = copy.deepcopy(intensity)
        self.direction = copy.deepcopy(direction)
        self.spectral_grid = copy.deepcopy(spectral_grid)
        self.units = units

        return

    def convertto_Wm2(self):
        if self.units == 'Wm2':
            return self.intensity
        elif self.units == 'ergscm2':
            self.intensity = self.intensity*1.e-3
            self.units = 'Wm2'
            return self.intensity
        elif self.units == 'nWcm2':
            self.intensity = self.intensity*1.e-5
            self.units = 'Wm2'
            return self.intensity

    def convertto_ergscm2(self):
        intensity = self.convertto_Wm2()
        self.intensity = intensity*1.e3
        self.units = 'ergscm2'
        return self.intensity

    def convertto_nWcm2(self):
        intensity = self.convertto_Wm2()
        self.intensity = intensity*1.e5
        self.units = 'nWcm2'
        return self.intensity

    def convertto_nm(self):
        if self.spectral_grid.units == 'nm':
            return self.spectral_grid.grid, self.intensity
        elif self.spectral_grid.units == 'mum':
            self.intensity = self.intensity*1.e-3
            self.spectral_grid.convertto_nm()
            return self.spectral_grid.grid, self.intensity
        elif self.spectral_grid.units == 'cm_1':
            self.intensity = self.intensity*self.spectral_grid.grid**2*1.e-7
            self.intensity = self.intensity[::-1]
            self.spectral_grid.convertto_nm()
            return self.spectral_grid.grid, self.intensity
        elif self.spectral_grid.units == 'hz':
            self.intensity = self.intensity*self.spectral_grid.grid**2*1.e-9/const.constants.c
            self.intensity = self.intensity[::-1]
            self.spectral_grid.convertto_nm()
            return self.spectral_grid.grid, self.intensity

    def convertto_mum(self):
        self.convertto_nm()
        self.intensity = self.intensity*1.e3
        self.spectral_grid.convertto_mum()
        return self.spectral_grid.grid, self.intensity

    def convertto_cm_1(self):
        self.convertto_nm()
        self.intensity = self.intensity*self.spectral_grid.grid**2*1.e-7
        self.intensity = self.intensity[::-1]
        self.spectral_grid.convertto_cm_1()
        return self.spectral_grid.grid, self.intensity

    def convertto_hz(self):
        self.convertto_nm()
        self.intensity = self.intensity*self.spectral_grid.grid**2*1.e-9/const.constants.c
        self.intensity = self.intensity[::-1]
        self.spectral_grid.convertto_hz()
        return self.spectral_grid.grid, self.intensity


    def integrate(self, w1 = None, w2 = None, offset = None):
        """
        Integrates the spectrum from w1 to w2, subtracting the offset if set.
        """
        if w1 is None and w2 is None:
            cond = ~np.isnan(self.intensity)
        elif w1 is not None and w2 is None:
            cond = (~np.isnan(self.intensity)) & (self.spectral_grid.grid >= w1)
        elif w1 is None and w2 is not None:
            cond = (~np.isnan(self.intensity)) & (self.spectral_grid.grid <= w2)
        else:
            cond = (~np.isnan(self.intensity)) & (self.spectral_grid.grid <= w2) & (self.spectral_grid.grid >= w1)

        if offset is not None:
            spect = self.intensity - offset
        else:
            spect = self.intensity

        intt = np.trapz(spect[cond],x=self.spectral_grid.grid[cond])

        return intt


    def convolve_to_grid(self, new_spectral_grid, spectral_widths = None, conv_type = 'gaussian'):
        """
        Convolution of the spectrum to a different grid.
        """
        pass


#############################################################################

# def group_for_level(lines,levels,level_names):
#     """
#     Returns a dictionary. Each entry is the list for the selected level. All other lines are put in 'other'.
#     """


def read_mw_list(db_cart, nome = 'mw_list.dat'):
    """
    Reads list of MWs in the db_cart.
    """

    mw_list = open(db_cart+nome,'r')

    n_mws = int(mw_list.readline())

    mw_tags = []
    mw_ranges = []

    for i in range(n_mws):
        linea = mw_list.readline().split()
        mw_tags.append(linea[1])
        mw_ranges.append([float(linea[2]),float(linea[3])])

    return n_mws, mw_tags, mw_ranges


def read_line_database(nome_sp, mol = None, iso = None, up_lev = None, down_lev = None, db_format = 'HITRAN', freq_range = None, n_skip = 0, link_to_isomolecs = None, verbose = False):
    """
    Reads line spectral data.
    :param nome_sp: spectral data file
    :param mol: HITRAN molecule number
    :param iso: HITRAN iso number
    :param up_lev: Upper level HITRAN string
    :param down_lev: Lower level HITRAN string
    :param format: If 'gbb' the format of MAKE_MW is used in reading, if 'HITRAN' the HITRAN2012 format.
    returns:
    list of SpectLine objects.
    """

    if db_format == 'gbb':
        delim = (2, 1, 12, 10, 10, 6, 6, 10, 4, 8, 15, 15, 15, 15)
        cose = ('Mol', 'Iso', 'Freq', 'Strength', 'A_coef', 'Air_broad', 'Self_broad', 'Energy_low', 'T_dep_broad', 'P_shift', 'Up_lev_str', 'Lo_lev_str', 'Q_num_up', 'Q_num_lo')
        cose2 = 2 * 'i4,' + 8 * 'f8,' + 3 * '|S15,' + '|S15'
    elif db_format == 'HITRAN':
        delim = (2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 15, 19, 7, 7)
        cose = ('Mol', 'Iso', 'Freq', 'Strength', 'A_coef', 'Air_broad', 'Self_broad', 'Energy_low', 'T_dep_broad', 'P_shift', 'Up_lev_str', 'Lo_lev_str', 'Q_num_up', 'Q_num_lo', 'others', 'g_up', 'g_lo')
        cose2 = 'i4,i4' + 8 * ',f8' + 4 * ',|S15' + ',|S19' + 2 * ',f8'
    else:
        raise ValueError('Allowed values for db_format: {}, {}'.format('gbb','HITRAN'))

    infi = open(nome_sp, 'r')
    if n_skip == -1:
        sbm.trova_spip(infi)
    else:
        for i in range(n_skip):
            infi.readline()  # skip the first n_skip lines

    linee = np.genfromtxt(infi, delimiter=delim, dtype=cose2,names=cose)

    linee_ok = []

    #print('Creo lista di oggetti linea\n')
    for linea in linee:
        if verbose:
            print(linea['Mol'], linea['Iso'], linea['Freq'])
        if freq_range is not None:
            if linea['Freq'] < freq_range[0]:
                continue
            if linea['Freq'] > freq_range[1]:
                break
        if (linea['Mol'] == mol or mol is None) and (linea['Iso'] == iso or iso is None) and (linea['Up_lev_str'] == up_lev or up_lev is None) and (linea['Lo_lev_str'] == down_lev or down_lev is None):
            line = SpectLine(linea)
            if verbose:
                print(linea)
            if link_to_isomolecs is not None:
                IsoMolecol = [molecolo for molecolo in link_to_isomolecs if (molecolo.mol == mol and molecolo.iso == iso)]
                if len(IsoMolecol) > 1:
                    raise ValueError('Multiple levels corresponding to line! WTF?')
                line._LinkToMolec_(IsoMolecol)
            linee_ok.append(line)
    #print('Ho creato lista di oggetti linea??\n')

    infi.close()
    return linee_ok


def ImportPartitionSumTable(mol,iso):
    """
    Loads the tabulated values for the internal partition sum.
    Outputs: gi -> state independent degeneracy factor, T_grid -> temperature grid, Q_grid -> Q tabulated
    """
    import fparts_mod

    gi, T_grid, Q_grid = fparts_mod.bd_tips_2003(mol,iso)

    return gi, T_grid, Q_grid


def CalcPartitionSum(mol,iso,temp=296.0):
    """
    Calculates the internal partition sum for molecule mol (iso) at the chosen temperature. Default temp is 296 K as in hitran.
    """

    gi, T_grid, Q_grid = ImportPartitionSumTable(mol, iso)

    x1 = T_grid[T_grid <= temp][-2:]
    x2 = T_grid[T_grid > temp][:2]
    x = np.hstack([x1,x2])

    qg1 = Q_grid[T_grid <= temp][-2:]
    qg2 = Q_grid[T_grid > temp][:2]
    qg = np.hstack([qg1,qg2])

    poli = scipy.interpolate.lagrange(x,qg)

    q = poli(temp)

    return q


def CalcStrength_at_T(mol, iso, S_ref, w, E_low, T, T_ref=296.):
    """
    Calculates line strength at temperature T given the standard line strength S_ref at T_ref.
    Default for HITRAN format with T_ref = 296 K.
    :param S_ref: Reference line strength at T_ref. (cm-1/mol cm-2)
    :param w: Central line wavenumber. (cm-1)
    :param E_low: Energy of the lower level. (cm-1)
    :param T: Temperature at which we want S(T).
    :param T_ref: Reference temperature for S_ref. Default 296 K.
    :return: S(T)
    """

    def fu_exp(mol,iso,E_low,w,T):
        g_nuc, Q_T = Q_calc(mol, iso, T)
        li = mt.exp(-E_low/(kbc*T)) * (1 - mt.exp(-w/(kbc*T))) / Q_T
        return li

    S = S_ref * fu_exp(mol, iso, E_low, w, T)/fu_exp(mol, iso, E_low, w, T_ref)

    return S


def Einstein_A_to_B(A_coeff, wavenumber, units = 'cm3ergs'):
    """
    Calculates the Eintein B coeff for induced absorption. B is expressed in m^3/J*s and is defined so that in the expression for the light absorption it appears with the radiation density rho (not with the number of photons, other possible definition for B).
    """
    nu = convertto_hz(wavenumber, 'cm_1')

    fact = 8*np.pi*const.constants.h*nu**3/const.constants.c**3
    B_coeff = A_coeff/fact

    if units == 'm3Js':
        pass
    elif units == 'cm3ergs':
        B_coeff = B_coeff * 1.e6 * 1.e-7

    return B_coeff


def Einstein_A_to_LineStrength_vardavas(A_coeff, wavenumber, temp, Q_part, g_lower, E_lower, g_upper):
    """
    Calculates the line strength in units cm-1/cm-2 (HITRAN units).
    """

    B_coeff = Einstein_A_to_B(A_coeff, wavenumber, units = 'cm3ergs')
    E_upper = E_lower + wavenumber
    print(E_lower,wavenumber,E_upper)
    print(Boltz_pop_at_T(E_lower, temp, g_lower, Q_part))
    print(1 - g_lower*Boltz_ratio_nodeg(E_upper,temp)/(g_upper*Boltz_ratio_nodeg(E_lower,temp)))
    print(B_coeff)

    S = h_cgs*wavenumber/c_cgs * Boltz_pop_at_T(E_lower, temp, g_lower, Q_part) * (1 - g_lower*Boltz_ratio_nodeg(E_upper,temp)/(g_upper*Boltz_ratio_nodeg(E_lower,temp))) * B_coeff

    return S


def Einstein_A_to_LineStrength(A_coeff, wavenumber, temp, Q_part, g_lower, E_lower):
    """
    Calculates the line strength at LTE in units cm-1/s-1.
    """
    B_coeff = Einstein_A_to_B(A_coeff, wavenumber, units = 'm3Js')
    print(E_lower,wavenumber)
    print(Boltz_pop_at_T(E_lower, temp, g_lower, Q_part))
    print(unomenoexp(-const.constants.h*const.constants.c*1.e2*wavenumber/(const.constants.Boltzmann*temp)))
    print(B_coeff)

    S = B_coeff * const.constants.h * wavenumber * 1.e2 * Boltz_pop_at_T(E_lower, temp, g_lower, Q_part) * unomenoexp(-const.constants.h*const.constants.c*1.e2*wavenumber/(const.constants.Boltzmann*temp))

    return S


def Einstein_A_to_LineStrength_hitran(A_coeff, wavenumber, temp, Q_part, g_upper, E_lower, iso_ab = 1.0):
    """
    Calculates the line strength at LTE in units cm-1/(mol cm-2).
    """
    B_coeff = Einstein_A_to_B(A_coeff, wavenumber, units = 'cm3ergs')
    # print(E_lower,wavenumber)
    # print(Boltz_pop_at_T(E_lower, temp, g_lower, Q_part))
    # print(unomenoexp(-const.constants.h*const.constants.c*1.e2*wavenumber/(const.constants.Boltzmann*temp)))
    print('eja',h_cgs,c_cgs,k_cgs,c2)
    print(B_coeff)

    S = iso_ab * A_coeff * g_upper * np.exp(-c2*E_lower/temp)*(1-np.exp(-c2*wavenumber/temp)) / (8 * np.pi * c_cgs * wavenumber**2 * Q_part)
    #S = B_coeff * const.constants.h * wavenumber * 1.e2 * Boltz_pop_at_T(E_lower, temp, g_lower, Q_part) * unomenoexp(-const.constants.h*const.constants.c*1.e2*wavenumber/(const.constants.Boltzmann*temp))

    return S



def unomenoexp(x):
    return (1-np.exp(x))

def Boltz_pop_at_T(wavenumber, temp, g_level, Q_part):
    """
    Boltzmann statistics LTE population of level.
    """

    enne = g_level * Boltz_ratio_nodeg(wavenumber, temp) / Q_part

    return enne

def Boltz_ratio_nodeg(wavenumber, temp):
    ratio = np.exp(-wavenumber/(k_cgs*temp))
    return ratio


def convert_cm_1_to_J(w):
    J = const.constants.h*1.e2*const.constants.c*w
    return J

def convert_cm_1_to_eV(w):
    eV = convert_cm_1_to_J(w)/const.constants.eV
    return J

def convert_J_to_eV(J):
    eV = J/const.constants.eV
    return J

def convertto_nm(w, units):
    if units == 'nm':
        return w
    elif units == 'mum':
        w = w*1.e3
        return w
    elif units == 'cm_1':
        w = 1.e7/w
        return w
    elif units == 'hz':
        w = const.constants.c/(1.e-9*w)
        return w

def convertto_cm_1(w, units):
    w = convertto_nm(w, units)
    w = 1.e7/w
    return w

def convertto_mum(w, units):
    w = convertto_nm(w, units)
    w = w*1.e-3
    return w

def convertto_hz(w, units):
    w = convertto_nm(w, units)
    w = const.constants.c/(1.e-9*w)
    return w


def BB(T,w):
    """
    Black body at temp. T in units of nW/(cm2*cm-1).
    :param T: Temperature
    :param w: Wavenumber (cm-1)
    :return:
    """
    rc1 = 1.1904e-3
    rhck = 1.4388

    BB = rc1 * w**3 / (mt.exp(w*rhck/T)-1)

    return BB


def BB_nm(T,w):
    """
    Black body at temp. T in units of W/(m2*nm).
    :param T: Temperature
    :param w: Wavelength (nm)
    :return:
    """
    rc1 = 1.1904
    rhck = 1.4388

    if T*w > 5e5 :
        BB = rc1 * mt.pow((1.e4/w),5) / (mt.exp(1.e7/w*rhck/T)-1)
    else:
        BB = rc1 * mt.pow((1.e4/w),5) * mt.exp(-1.e7/w*rhck/T)

    return BB

##########################################################################
