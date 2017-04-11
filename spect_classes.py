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
from fparts import bd_tips_2003 as Q_calc

#parameters
Rtit = 2575.0 # km
Mtit = 1.3452e23 # kg
kb = 1.38065e-19 # boltzmann constant to use with P in hPa, n in cm-3, T in K (10^-4 * Kb standard)
c_R = 8.31446 # J K-1 mol-1
c_G = 6.67408e-11 # m3 kg-1 s-2
kbc = const.k/(const.h*100*const.c) # 0.69503

### NAMES for SpectLine object
cose = ('Mol', 'Iso', 'Freq', 'Strength', 'A_coef', 'Air_broad', 'Self_broad', 'Energy_low', 'T_dep_broad', 'P_shift','Up_lev_str', 'Lo_lev_str', 'Q_num_up', 'Q_num_lo')

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

        if nomi != cose:
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


def read_line_database(nome_sp, mol = None, iso = None, up_lev = None, down_lev = None, db_format = 'gbb', n_skip = 3, link_to_isomolecs = None):
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
    elif db_format == 'HITRAM':
        delim = (2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 15)
    else:
        raise ValueError('Allowed values for db_format: {}, {}'.format('gbb','HITRAN'))

    infi = open(nome_sp, 'r')
    if n_skip == 0:
        sbm.trova_spip(infi)
    else:
        for i in range(n_skip):
            infi.readline()  # skip the first n_skip lines

    cose = ('Mol', 'Iso', 'Freq', 'Strength', 'A_coef', 'Air_broad', 'Self_broad', 'Energy_low', 'T_dep_broad', 'P_shift', 'Up_lev_str', 'Lo_lev_str', 'Q_num_up', 'Q_num_lo')
    cose2 = 2 * 'i4,' + 8 * 'f8,' + 3 * '|S15,' + '|S15'
    linee = np.genfromtxt(infi, delimiter=delim, dtype=cose2,names=cose)

    linee_ok = []

    #print('Creo lista di oggetti linea\n')
    for linea in linee:
        if (linea['Mol'] == mol or mol is None) and (linea['Iso'] == iso or iso is None) and (linea['Up_lev_str'] == up_lev or up_lev is None) and (linea['Lo_lev_str'] == down_lev or down_lev is None):
            line = SpectLine(linea)
            if link_to_isomolecs is not None:
                IsoMolecol = [molecolo for molecolo in link_to_isomolecs if (molecolo.mol == mol and molecolo.iso == iso)]
                if len(IsoMolecol) > 1:
                    raise ValueError('Multiple levels corresponding to line! WTF?')
                line._LinkToMolec_(IsoMolecol)
            linee_ok.append(line)
    #print('Ho creato lista di oggetti linea??\n')

    infi.close()
    return linee_ok


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
