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
from scipy.interpolate import lagrange
import warnings
import copy
import spect_base_module as sbm
import lineshape
import time
from multiprocessing import Process, Queue
import cPickle as pickle
#from numba import jit

n_threads = 8

# Fortran code parameters
imxsig = 13010
imxlines = 40000
imxsig_long = 2000000

#parameters
Rtit = 2575.0 # km
Mtit = 1.3452e23 # kg
kb = 1.38065e-19 # boltzmann constant to use with P in hPa, n in cm-3, T in K (10^-4 * Kb standard)
c_R = 8.31446 # J K-1 mol-1
c_G = 6.67408e-11 # m3 kg-1 s-2
kbc = const.k/(const.h*100*const.c) # 0.69503

T_ref = 296.0
hpa_to_atm = 0.00098692326671601

#####################################
# Define for the radiative transfer in cgs - DEFINED AS IN HITRAN
h_cgs = const.physical_constants['Planck constant'][0]*1.e7 # erg s
c_cgs = const.constants.c * 1.e2 # cm s-1
k_cgs = const.physical_constants['Boltzmann constant'][0]*1.e7 # erg K-1
c2 = h_cgs * c_cgs / k_cgs

### NAMES for SpectLine object
cose = ('Mol', 'Iso', 'Freq', 'Strength', 'A_coeff', 'Air_broad', 'Self_broad', 'E_lower', 'T_dep_broad', 'P_shift','Up_lev_str', 'Lo_lev_str', 'Q_num_up', 'Q_num_lo')
cose_hit = ('Mol', 'Iso', 'Freq', 'Strength', 'A_coeff', 'Air_broad', 'Self_broad', 'E_lower', 'T_dep_broad', 'P_shift','Up_lev_str', 'Lo_lev_str', 'Q_num_up', 'Q_num_lo', 'others', 'g_up', 'g_lo')

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


    def Print(self, ofile = None):
        if ofile is None:
            print('{:4d}{:2d}{:8.2f}{:10.3e}{:8.2f}{:15s}{:15s}{:8.3f}{:8.3f}'.format(self.Mol,self.Iso,self.Freq,self.Strength,self.E_lower,self.Up_lev_str,self.Lo_lev_str,self.g_lo,self.g_up))
        else:
            ofile.write('{:4d}{:2d}{:8.2f}{:10.3e}{:8.2f}{:15s}{:15s}{:8.3f}{:8.3f}'.format(self.Mol,self.Iso,self.Freq,self.Strength,self.E_lower,self.Up_lev_str,self.Lo_lev_str,self.g_lo,self.g_up))
        return


    def ShowCalc(self, T, P = 1, nlte_ratio = 1):
        pass


    def CalcStrength(self, T):
        """
        Calcs Line Strength at temperature T.
        """
        return CalcStrength_at_T(self.Mol, self.Iso, self.Strength, self.Freq, self.E_lower, T)


    def LinkToMolec(self, isomolec):
        """
        IsoMolec has to be of the corresponding type in module spect_base_module.
        Sets the values of the two attr. "Up_lev_id" and "Lo_lev_id" with the internal names of the corresponding levels in IsoMolec. So that all level quantities can be accessed via Level = getattr(IsoMolec,Up_lev_id) and using Level..
        """
        if isomolec is None:
            return False

        self.Up_lev_id = None
        self.E_vib_up = None
        self.Lo_lev_id = None
        self.E_vib_lo = None

        for lev in isomolec.levels:
            Level = getattr(isomolec, lev)
            if Level.minimal_level_string() == self.minimal_level_string_up():
                self.Up_lev_id = lev
                self.E_vib_up = Level.energy
            elif Level.minimal_level_string() == self.minimal_level_string_lo():
                self.Lo_lev_id = lev
                self.E_vib_lo = Level.energy
            else:
                continue

        #print(self.Up_lev_id, self.E_vib_up, self.Lo_lev_id, self.E_vib_lo)
        if self.Up_lev_id is None or self.Lo_lev_id is None:
            return False
        else:
            return True

    def Einstein_A_to_B(self):
        """
        Calculates the Einstein B
        """
        B = Einstein_A_to_B(self.A_coeff, self.Freq, units = 'cm3ergcm2')

        return B


    def CheckWidths(self, Temp, Pres, MM):
        """
        Returns the pressure and doppler half widths for the line.
        """
        Pres_atm = convert_to_atm(Pres, units='hPa')
        p_shift = self.P_shift * Pres_atm # Pressure shift

        lw = Lorenz_width(Temp, Pres_atm, self.T_dep_broad, self.Air_broad)
        dw = Doppler_width(Temp, MM, self.Freq)

        return dw, lw, p_shift


    def MakeShapeLine(self, Temp, Pres, grid = None, MM = None, Strength = 1.0, verbose = False, keep_memory = False):
        """
        Calls routine makeshape.
        """
        if MM is None:
            MM = sbm.find_molec_metadata(self.Mol, self.Iso)['iso_MM']

        if grid is None:
            sp_step = 5.e-4
            grid = np.arange(-imxsig*sp_step/2,imxsig*sp_step/2,sp_step, dtype = float)
            grid = SpectralGrid(grid+self.Freq, units = 'cm_1')

        Pres_atm = convert_to_atm(Pres, units='hPa')
        wn_0 = self.Freq + self.P_shift * Pres_atm # Pressure shift
        if(verbose): print('Shifto da {} a {}'.format(self.Freq, wn_0))

        lw = Lorenz_width(Temp, Pres_atm, self.T_dep_broad, self.Air_broad)
        dw = Doppler_width(Temp, MM, self.Freq)
        if(verbose): print('Lorenz width = {}; Doppler width = {}'.format(lw,dw))

        #shape = MakeShape_py(wn_arr, self.Freq, lw, dw, Strength = Strength)

        try:
            shape = MakeShape(grid, self.Freq, lw, dw, Strength = Strength)
        except Exception as cazzillo:
            self.Print()
            print(dw, Temp, MM, self.Freq)
            raise cazzillo

        if keep_memory:
            self.shape = shape

        return shape #, lw, dw

    def CalcStrength_nonLTE(self, T_vib_lower, T_vib_upper, Q_part):
        """
        Calculates the line strength S in nonLTE conditions.
        """
        S = Einstein_A_to_LineStrength_nonLTE(self.A_coeff, self.Freq, self.E_lower, T_vib_lower, T_vib_upper, self.g_lo, self.g_up, Q_part)

        return S

    def CalcStrength_from_Einstein(self, Temp, Q_part = None, iso_ab = None, isomolec = None, T_vib_lower = None, T_vib_upper = None):
        """
        Calculates the line strength S in nonLTE conditions, starting from the Einstein A coeff.
        """
        if T_vib_lower is None:
            T_vib_lower = Temp
        if T_vib_upper is None:
            T_vib_upper = Temp

        if Q_part is None:
            Q_part = CalcPartitionSum(self.Mol, self.Iso, temp = Temp)

        if iso_ab is None:
            iso_ab = sbm.find_molec_metadata(self.Mol, self.Iso)['iso_ratio']

        G_coeffs = self.Calc_Gcoeffs(Temp, isomolec = isomolec)

        try:
            E_vib_lo = self.E_vib_lo
            E_vib_up = self.E_vib_up
        except:
            E_vib_lo = 0.0
            E_vib_up = 0.0

        try:
            S_ab = (G_coeffs['absorption']*Boltz_ratio_nodeg(E_vib_lo, T_vib_lower) - G_coeffs['ind_emission']*Boltz_ratio_nodeg(E_vib_up, T_vib_upper))/Q_part

            S_em = G_coeffs['sp_emission']*Boltz_ratio_nodeg(E_vib_up, T_vib_upper)/Q_part
        except Exception as cazzillo:
            self.Print()
            print(self.E_vib_lo, self.E_vib_up, T_vib_upper, T_vib_lower)
            raise cazzillo

        return iso_ab*S_ab, iso_ab*S_em


    def Calc_Gcoeffs(self, Temp, isomolec = None):
        ctypes = ['sp_emission','ind_emission','absorption']
        values = []

        ok = self.LinkToMolec(isomolec)

        if not ok:
            #raise RuntimeWarning('Line is unidentified! setting 0.0 as level_energy')
            lev_energy_lo = 0.0
            lev_energy_up = 0.0
        else:
            lev_energy_lo = self.E_vib_lo
            lev_energy_up = self.E_vib_up

        G_co = Einstein_A_to_Gcoeff_spem(self, Temp, lev_energy_up)
        values.append(G_co)
        G_co = Einstein_A_to_Gcoeff_indem(self, Temp, lev_energy_up)
        values.append(G_co)
        G_co = Einstein_A_to_Gcoeff_abs(self, Temp, lev_energy_lo)
        values.append(G_co)

        G_coeffs = dict(zip(ctypes,values))

        self.G_coeffs = G_coeffs

        return G_coeffs

    def minimal_level_string_up(self):
        minstr, _, _ = sbm.extract_quanta_HITRAN(self.Mol, self.Iso, self.Up_lev_str)
        return minstr

    def minimal_level_string_lo(self):
        minstr, _, _ = sbm.extract_quanta_HITRAN(self.Mol, self.Iso, self.Lo_lev_str)
        return minstr


class SpectralGrid(object):
    """
    This is the spectral grid. Some useful methods (conversions, e basta..).
    """

    def __init__(self, spectral_grid, units = 'nm'):
        self.grid = copy.deepcopy(spectral_grid)
        self.units = units
        if len(spectral_grid) > imxsig_long:
            raise ValueError('Grid longer that the max value imxsig_long set to {}'.format(imxsig_long))
        return

    def step(self):
        return self.grid[1]-self.grid[0]

    def len_wn(self):
        return len(self.grid)

    def wn_range(self):
        return [self.min_wn(), self.max_wn()]

    def min_wn(self):
        return min(self.grid)

    def max_wn(self):
        return max(self.grid)

    def half_precision(self):
        """
        Converts the grid to half precision (float16) for saving.
        """
        self.grid = self.grid.astype(np.float16)
        return

    def double_precision(self):
        """
        Converts the grid to double precision (float).
        """
        self.grid = self.grid.astype(float)
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


class SpectralObject(object):
    """
    This is the general class for all spectral quantities, that is for distributions in the frequency/wavelength/wavenumber domani. Some useful methods (conversions, integration, convolution, ..).
    """

    def __init__(self, spectrum, spectral_grid, direction = None, units = ''):
        self.spectrum = copy.deepcopy(spectrum)
        self.direction = copy.deepcopy(direction)
        self.spectral_grid = copy.deepcopy(spectral_grid)
        self.units = units

        return

    def __getitem__(self, key):
        #print(key[0], key[1])
        cond = (self.spectral_grid.grid > key[0]-self.spectral_grid.step()/2.0) & (self.spectral_grid.grid < key[1]+self.spectral_grid.step()/2.0)
        if not np.any(cond):
            return None
        spectral_grid = SpectralGrid(self.spectral_grid.grid[cond], units = self.units)
        coso = copy.deepcopy(self)
        coso.spectral_grid = spectral_grid
        coso.spectrum = self.spectrum[cond]
        return coso

    def __add__(self, obj2):
        coso = copy.deepcopy(self)
        if isinstance(obj2, SpectralObject):
            if len(obj2.spectrum) == len(self.spectrum):
                coso.spectrum += obj2.spectrum
            else:
                coso.add_to_spectrum(obj2)
        else:
            coso.spectrum += obj2
        return coso

    def __sub__(self, obj2):
        coso = copy.deepcopy(self)
        if isinstance(obj2, SpectralObject):
            if len(obj2.spectrum) == len(self.spectrum):
                coso.spectrum -= obj2.spectrum
            else:
                coso.add_to_spectrum(obj2, Strength=-1.0)
        else:
            coso.spectrum -= obj2
        return coso

    def __mul__(self, obj2):
        coso = copy.deepcopy(self)
        if isinstance(obj2, SpectralObject):
            coso.spectrum *= obj2.spectrum
        else:
            coso.spectrum *= obj2
        return coso

    def __div__(self, obj2):
        coso = copy.deepcopy(self)
        if isinstance(obj2, SpectralObject):
            coso.spectrum /= obj2.spectrum
        else:
            coso.spectrum /= obj2
        return coso

    # def __truediv__(self, obj2):
    #     coso = copy.deepcopy(self)
    #     if type(obj2) is SpectralObject:
    #         coso.spectrum /= obj2.spectrum
    #     else:
    #         coso.spectrum /= obj2
    #     return coso

    def max(self):
        return np.max(self.spectrum)

    def min(self):
        return np.min(self.spectrum)

    def n_points(self):
        return len(self.spectrum)

    def add_mask(self, mask):
        """
        Adds a mask (an array of bool usually) to the obj.
        """
        self.mask = mask
        return

    def multiply(self, factor, save = True):
        """
        Simply multiplies all the spectrum by factor.
        """
        if save:
            self.spectrum = self.spectrum * factor
            return
        else:
            coso_new = factor * self.spectrum
            coso = copy.deepcopy(self)
            coso.spectrum = coso_new
            return coso

    def exp_elementwise(self, exp_factor, save = False):
        coso_new = np.exp(self.spectrum * exp_factor)
        if save:
            self.spectrum = coso_new
            return
        else:
            coso = copy.deepcopy(self)
            coso.spectrum = coso_new
            return coso


    def erase_grid(self):
        """
        Erases the spectral_grid to save disk space.
        """
        self.spectral_grid = None
        return

    def restore_grid(self, spectral_grid):
        """
        Erases the spectral_grid to save disk space.
        """
        self.spectral_grid = copy.deepcopy(spectral_grid)
        return

    def half_precision(self):
        """
        Converts the spectrum to half precision (np.float16) for saving.
        """
        # print(self.spectrum.dtype)
        # print(np.max(self.spectrum))
        # cos = np.frexp(self.spectrum)
        # esp = cos[1]
        # mant = np.float32()
        # self.spectrum = copy.deepcopy(cos.astype(np.float16))
        self.spectrum = self.spectrum.astype(np.float32)
        # print(self.spectrum.dtype)
        # print(np.max(self.spectrum))
        if self.spectral_grid is not None:
            self.spectral_grid.half_precision()
        return

    def double_precision(self):
        """
        Converts the spectrum to double precision (float).
        """
        self.spectrum = self.spectrum.astype(float)
        if self.spectral_grid is not None:
            self.spectral_grid.double_precision()
        return

    def compress_spectrum(self, threshold = 1.e-25):
        """
        Converts the spectrum to scipy.sparse.csr_matrix. threshold should be carefully evaluated in order to avoid errors in the spectrum computation. Deve essere threshold*column_density << 1 (not more than 0.001 I'd say).
        """
        coso = self.spectrum
        coso[self.spectrum < threshold] = 0.0
        self.spectrum = scipy.sparse.csr_matrix(coso)
        return

    def convert_grid_to(self, units):
        if units == 'nm':
            out = self.convertto_nm()
        elif units == 'mum':
            out = self.convertto_mum()
        elif units == 'cm_1':
            out = self.convertto_cm_1()
        elif units == 'hz':
            out = self.convertto_hz()
        else:
            raise ValueError('Cannot recognize units {}'.format(units))

        return out

    def convertto_nm(self):
        if self.spectral_grid.units == 'nm':
            return self.spectral_grid.grid, self.spectrum
        elif self.spectral_grid.units == 'mum':
            self.spectrum = self.spectrum*1.e-3
            self.spectral_grid.convertto_nm()
            return self.spectral_grid.grid, self.spectrum
        elif self.spectral_grid.units == 'cm_1':
            self.spectrum = self.spectrum*self.spectral_grid.grid**2*1.e-7
            self.spectrum = self.spectrum[::-1]
            self.spectral_grid.convertto_nm()
            return self.spectral_grid.grid, self.spectrum
        elif self.spectral_grid.units == 'hz':
            self.spectrum = self.spectrum*self.spectral_grid.grid**2*1.e-9/const.constants.c
            self.spectrum = self.spectrum[::-1]
            self.spectral_grid.convertto_nm()
            return self.spectral_grid.grid, self.spectrum

    def convertto_mum(self):
        self.convertto_nm()
        self.spectrum = self.spectrum*1.e3
        self.spectral_grid.convertto_mum()
        return self.spectral_grid.grid, self.spectrum

    def convertto_cm_1(self):
        self.convertto_nm()
        self.spectrum = self.spectrum*self.spectral_grid.grid**2*1.e-7
        self.spectrum = self.spectrum[::-1]
        self.spectral_grid.convertto_cm_1()
        return self.spectral_grid.grid, self.spectrum

    def convertto_hz(self):
        self.convertto_nm()
        self.spectrum = self.spectrum*self.spectral_grid.grid**2*1.e-9/const.constants.c
        self.spectrum = self.spectrum[::-1]
        self.spectral_grid.convertto_hz()
        return self.spectral_grid.grid, self.spectrum


    def integrate(self, w1 = None, w2 = None, offset = None):
        """
        Integrates the spectrum from w1 to w2, subtracting the offset if set.
        """
        if w1 is None and w2 is None:
            cond = ~np.isnan(self.spectrum)
        elif w1 is not None and w2 is None:
            cond = (~np.isnan(self.spectrum)) & (self.spectral_grid.grid >= w1)
        elif w1 is None and w2 is not None:
            cond = (~np.isnan(self.spectrum)) & (self.spectral_grid.grid <= w2)
        else:
            cond = (~np.isnan(self.spectrum)) & (self.spectral_grid.grid <= w2) & (self.spectral_grid.grid >= w1)

        if offset is not None:
            spect = self.spectrum - offset
        else:
            spect = self.spectrum

        intt = np.trapz(spect[cond],x=self.spectral_grid.grid[cond])

        return intt

    def convolve_to_grid(self, new_spectral_grid, spectral_widths = None, conv_type = 'gaussian', n_sigma = 5.):
        """
        Convolution of the spectrum to a different grid.
        """
        new_len = len(new_spectral_grid.grid)
        if spectral_widths is None:
            weed = new_spectral_grid.step()
        else:
            weed = max(spectral_widths)

        sp_step_old = self.spectral_grid.step()
        n_points = int(n_sigma*weed/sp_step_old)
        lin_grid = np.arange(-n_points*sp_step_old,n_points*sp_step_old,sp_step_old, dtype = float)

        if spectral_widths is None:
            gigi = gaussian(lin_grid, 0.0, new_spectral_grid.step())
            spectral_widths = [None]*new_len

        spectrum = np.zeros(new_len, dtype = float)

        for num, freq, wid in zip(range(new_len), new_spectral_grid.grid, spectral_widths):
            if wid is not None:
                gigi = gaussian(lin_grid, 0.0, wid)

            ind_ok, fr_grid_ok = closest_grid(self.spectral_grid, freq)
            lin_grid_ok = SpectralGrid(lin_grid+fr_grid_ok, units = 'cm_1')

            spect_old = self[lin_grid_ok.grid[0],lin_grid_ok.grid[-1]]

            if len(spect_old.spectrum) < len(gigi):
                zero = SpectralObject(np.zeros(len(lin_grid_ok.grid)), lin_grid_ok)
                zero.add_to_spectrum(spect_old)
                spect_old = zero

            spectrum[num] = conv_single(spect_old, gigi, new_spectral_grid.step())
        convolved = copy.deepcopy(self)
        convolved.spectral_grid = copy.deepcopy(new_spectral_grid)
        convolved.spectrum = copy.deepcopy(spectrum)
        #convolved = SpectralObject(spectrum, new_spectral_grid)
        return convolved

    def interp_to_regular_grid(self):
        n_po = len(self.spectral_grid.grid)
        griniu = np.linspace(np.min(self.spectral_grid.grid), np.max(self.spectral_grid.grid), n_po)
        griniu = SpectralGrid(griniu, units = self.spectral_grid.units)
        niuspe = np.interp(griniu.grid, self.spectral_grid.grid, self.spectrum)
        self.spectrum = niuspe
        self.spectral_grid = griniu
        return

    def add_to_spectrum(self, spectrum2, Strength = None, sumcheck = 10.):
        """
        Spectrum2 is a SpectralObject with a spectral_grid which is entirely or partially included in the self.spectral_grid, with the SAME STEP.
        """
        if not sbm.isclose(self.spectral_grid.step(), spectrum2.spectral_grid.step()):
            #raise ValueError("Different steps {} and {}, can't add the two spectra, convolve first".format(self.spectral_grid.step(),spectrum2.spectral_grid.step()))
            print("Different steps {} and {}, can't add the two spectra, convolve first".format(self.spectral_grid.step(),spectrum2.spectral_grid.step()))

        len2 = len(spectrum2.spectrum)
        grid_intersect = np.intersect1d(self.spectral_grid.grid, spectrum2.spectral_grid.grid)
        spino = self.spectral_grid.step()/10.

        if len(grid_intersect) == 0:
            #print('No intersection between the two spectra, doing nothing')
            return
        else:
            ok = (self.spectral_grid.grid > spectrum2.spectral_grid.grid[0]-spino) & (self.spectral_grid.grid < spectrum2.spectral_grid.grid[-1]+spino)
            ok2 = (spectrum2.spectral_grid.grid > self.spectral_grid.grid[0]-spino) & (spectrum2.spectral_grid.grid < self.spectral_grid.grid[-1]+spino)
            #print(type(ok), type(ok2))
            #print(ok)
            #print(ok2)
            if Strength is not None:
                self.spectrum[ok] += Strength*spectrum2.spectrum[ok2]
            else:
                # print(np.sum(ok))
                # print(np.sum(ok2))
                # print(len(self.spectrum))
                # print(len(spectrum2.spectrum))
                # print(len(self.spectral_grid.grid))
                # print(len(spectrum2.spectral_grid.grid))
                #sys.exit()
                self.spectrum[ok] += spectrum2.spectrum[ok2]

            check_grid_diff = self.spectral_grid.grid[ok]-spectrum2.spectral_grid.grid[ok2]
            griddifsum = np.sum(np.abs(check_grid_diff))
            #if griddifsum > self.spectral_grid.step()/sumcheck:
                #print(griddifsum, len(self.spectral_grid.grid))
                # pl.ion()
                # pl.figure(78)
                # pl.plot(self.spectral_grid.grid[ok], spectrum2.spectral_grid.grid[ok2])
                # pl.plot(self.spectral_grid.grid[ok], self.spectral_grid.grid[ok])
                #print("Problem with the spectral grids! Check or lower the threshold for sumcheck")

            return griddifsum


    def multiply_elementwise(self, spectrum2, save = True):
        """
        Multiplies each element of self.spectrum by the corresponding element of spectrum2.
        """
        if len(self.spectrum) != len(spectrum2.spectrum):
            raise ValueError('The two spectra have different lengths!')

        if save:
            self.spectrum = spectrum2.spectrum * self.spectrum
        else:
            coso_new = spectrum2.spectrum * self.spectrum
            coso = copy.deepcopy(self)
            coso.spectrum = coso_new
            return coso

        return


    def divide_elementwise(self, spectrum2, save = True):
        """
        Divides each element of self.spectrum by the corresponding element of spectrum2.
        """
        if len(self.spectrum) != len(spectrum2.spectrum):
            raise ValueError('The two spectra have different lengths!')

        if save:
            self.spectrum = self.spectrum/spectrum2.spectrum
        else:
            coso_new = self.spectrum/spectrum2.spectrum
            coso = copy.deepcopy(self)
            coso.spectrum = coso_new
            return coso

        return


    def sum_scalar(self, scalar):
        self.spectrum = self.spectrum + scalar
        return self.spectrum


    def add_lines_to_spectrum(self, lines, Strengths = None, fix_length = imxsig, n_threads = n_threads):
        """
        Sums a set of lines to the spectrum, using a fast routine in fortran. All shapes are filled with zeros till fix_length dimension.

        To be added:
            - adaptation to large set of lines (for now max = 20000)
        """
        #print('Inside add_lines_to_spectrum.. summing up all the lines!')

        n_lines = len(lines)
        #print('Preparing to sum {} lines...'.format(len(lines)))
        if n_lines == 0:
            return self.spectrum

        if n_lines > imxlines:
            raise ValueError('{} are too many lines!! Increase the thresold imxlines (now {}) or decrease num of lines..'.format(n_lines,imxlines))
        if self.n_points() > imxsig_long:
            raise ValueError('The input spectrum is too long!! Increase the thresold imxsig_long or decrease num of wn..')

        time0 = time.time()

        lines_ok = []
        if Strengths is not None:
            for line, strength in zip(lines,Strengths):
                line2 = line.multiply(strength, save = False)
                lines_ok.append(line2)

        lines = lines_ok

        processi = []
        coda = []
        outputs = []

        nlinst = n_lines/n_threads

        for i in range(n_threads):
        #    print('Lancio proc: {}'.format(i))
            linee = lines[nlinst*i:nlinst*(i+1)]
            if i == n_threads-1:
                linee = lines[nlinst*i:]
            coda.append(Queue())
            processi.append(Process(target=self.prepare_fortran_sum,args=(linee, i, coda[i])))
            processi[i].start()

        for i in range(n_threads):
            outputs.append(coda[i].get())

        for i in range(n_threads):
            processi[i].join()

        initarr = np.array(outputs[0][1])
        finarr = np.array(outputs[0][2])
        matrix = outputs[0][0]

        for output, i in zip(outputs[1:],range(1,n_threads)):
            initarr = np.append(initarr, np.array(output[1]))
            finarr = np.append(finarr, np.array(output[2]))
            matrix = np.vstack([matrix, output[0]])

        del outputs
        del lines

        mancano = imxlines-len(finarr)
        matrzero = np.zeros((mancano,imxsig), dtype=float, order='F')
        matrix = np.vstack([matrix, matrzero])

        initarr = np.append(initarr, np.zeros(mancano, dtype=int))
        finarr = np.append(finarr, np.zeros(mancano, dtype=int))

        spettro = np.zeros(imxsig_long, dtype=float)
        spettro[:self.n_points()] = self.spectrum

        time_fort = time.time()
        #print('The python pre-routine took {} s to prepare {} lines for the sum'.format(time.time()-time0,n_lines))

        somma = lineshape.sum_all_lines(spettro, matrix, initarr, finarr, n_lines, self.n_points())
        #print('The fortran routine took {} s to sum {} lines'.format(time.time()-time_fort,n_lines))

        self.spectrum = somma[:self.n_points()]

        return self.spectrum


    def prepare_fortran_sum(self, lines, i, coda, fix_length = imxsig):
        """
        Subprocess of the call add_lines_to_spectrum. Each process calls this routine that produces a slice of the final matrix.
        """

        spino = self.spectral_grid.step()/10.
        n_lines = len(lines)

        matrix = np.zeros((n_lines,fix_length), dtype = float, order='F')
        init = []
        fin = []

        for line, iii in zip(lines,range(n_lines)):
            ok = np.where(((self.spectral_grid.grid > line.spectral_grid.grid[0]-spino) & (self.spectral_grid.grid < line.spectral_grid.grid[-1]+spino)))
            ok2 = np.where(((line.spectral_grid.grid > self.spectral_grid.grid[0]-spino) & (line.spectral_grid.grid < self.spectral_grid.grid[-1]+spino)))

            ok = np.ravel(ok)
            ok2 = np.ravel(ok2)

            ok_ini = ok[0]+1 # FORTRAN VECTORS START FROM 1!!!!!
            ok_fin = ok[-1]+1

            n_zeri = fix_length - (ok_fin - ok_ini +1)

            if n_zeri > 0:
                zeros = np.zeros(n_zeri, dtype=float)

                if ok_fin + n_zeri < self.n_points():
                    # appiccico zeri a dx
                    lineaa = np.append(line.spectrum[ok2], zeros)
                    ok_fin += n_zeri
                else:
                    # appiccico zeri a sx e sposto ok_ini
                    lineaa = np.append(zeros, line.spectrum[ok2])
                    ok_ini -= n_zeri
            else:
                lineaa = line.spectrum

            matrix[iii,:] = lineaa
            init.append(ok_ini)
            fin.append(ok_fin)

        initarr = np.array(init)
        finarr = np.array(fin)

        coda.put([matrix,initarr,finarr])

        return

    def plot(self, label = None):
        pl.plot(self.spectral_grid.grid, self.spectrum, label = label)
        return

    def norm_plot(self, label = None):
        pl.plot(self.spectral_grid.grid, self.spectrum/np.max(self.spectrum), label = label)
        return

#@jit
def conv_single(spect, window, step):
    convolution = np.trapz(spect.spectrum*window, x = spect.spectral_grid.grid)
    return convolution


class SpectralIntensity(SpectralObject):
    """
    This is the spectral intensity. Some useful methods (conversions, integration, convolution, ..).
    """

    def __init__(self, intensity, spectral_grid, direction = None, units = 'ergscm2'):
        self.spectrum = copy.deepcopy(intensity)
        self.direction = copy.deepcopy(direction)
        self.spectral_grid = copy.deepcopy(spectral_grid)
        self.units = units

        return

    def hires_to_lowres(self, lowres_obs, spectral_widths = None):
        self.convert_grid_to(lowres_obs.spectral_grid.units)
        self.interp_to_regular_grid()
        lowres = self.convolve_to_grid(lowres_obs.spectral_grid, spectral_widths = spectral_widths)
        lowres.convertto(lowres_obs.units)
        return lowres

    def convertto(self, new_units):
        if new_units == 'Wm2':
            self.convertto_Wm2()
        elif new_units == 'ergscm2':
            self.convertto_ergscm2()
        elif new_units == 'nWcm2':
            self.convertto_nWcm2()
        else:
            raise ValueError('No method for units '+new_units)

        return self.spectrum


    def convertto_Wm2(self):
        if self.units == 'Wm2':
            return self.spectrum
        elif self.units == 'ergscm2':
            self.spectrum = self.spectrum*1.e-3
            self.units = 'Wm2'
            return self.spectrum
        elif self.units == 'nWcm2':
            self.spectrum = self.spectrum*1.e-5
            self.units = 'Wm2'
            return self.spectrum

    def convertto_ergscm2(self):
        spectrum = self.convertto_Wm2()
        self.spectrum = spectrum*1.e3
        self.units = 'ergscm2'
        return self.spectrum

    def convertto_nWcm2(self):
        spectrum = self.convertto_Wm2()
        self.spectrum = spectrum*1.e5
        self.units = 'nWcm2'
        return self.spectrum

    def add_noise(self, noise):
        self.noise = copy.deepcopy(noise)
        return

    def add_bands(self, bands):
        self.bands = copy.deepcopy(bands)
        return



class SpectralGcoeff(SpectralObject):
    """
    This is the class to represent the G_abs, G_spem and G_indem level-specific coeffs that build up the absorption coefficient in non-LTE. These are needed to build the LUTs in non-LTE.

    <<< IMPORTANT: Set as level string the part of the string containing the quanta, not the simmetry of the levels. Instead it will not recognize levels with same quanta and different symmetry. Example: for level "0 0 1 2 1F2" set minimal_level_string = "0 0 1 2" >>>>>
    """

    def __init__(self, ctype, spectral_grid, mol, iso, MM, minimal_level_string, unidentified_lines = False, spectrum = None, Pres = None, Temp = None):
        self.mol = mol
        self.iso = iso
        self.MM = MM
        if not unidentified_lines:
            self.lev_string = minimal_level_string
            self.unidentified_lines = False
        else:
            self.unidentified_lines = True
        self.ctype = ctype
        self.spectral_grid = copy.deepcopy(spectral_grid)
        if spectrum is None:
            self.spectrum = np.zeros(len(spectral_grid.grid), dtype = float)
        else:
            self.spectrum = spectrum

        if Pres is not None and Temp is not None:
            self.pres = Pres
            self.temp = Temp

        return

    def BuildCoeff(self, lines, Temp, Pres, n_threads = n_threads, preCalc_shapes = False, debug = False):
        """
        Calculates the G_coeff for the selected level, using the proper lines among lines.
        If preCalc_shapes is set as True, the lines in input shall already contain the calculated shape as attribute. Default is False. Setting preCalc_shapes as True speeds up the calculation by a factor of three when calculating LUTs.
        """

        ctypes = ['sp_emission','ind_emission','absorption']

        if len(lines) == 0:
            #print('No lines here, returning..')
            self.temp = Temp
            self.pres = Pres
            return self.spectrum

        if preCalc_shapes:
            #print('Using precalculated shapes..')
            try:
                gigi = lines[0].shape
                gigi = lines[0].G_coeffs
            except Exception as cazzillo:
                print(cazzillo)
                raise ValueError('preCalc_shapes is set as True but the lines do not contain the attribute << shapes >>. Are you sure you precalculated the line shapes? Run calc_shapes_lines on your line set first.')
            lines_new = lines
        else:
            #print('Calculating shapes..')
            lines_new = self.calc_shapes(lines, Temp, Pres)

        #print(len(lines_new))
        #print('aaaaaaaaaaazzzulegnaaaaaaaaaaaaaaaa')

        if not self.unidentified_lines:
            if self.ctype == 'sp_emission' or self.ctype == 'ind_emission':
                lin_ok_levup = [lin for lin in lines_new if lin.Mol == self.mol and lin.Iso == self.iso and self.lev_string == lin.minimal_level_string_up()]
                shapes_tot = [lin.shape for lin in lin_ok_levup]
                G_coeffs_tot = [lin.G_coeffs[self.ctype] for lin in lin_ok_levup]

                if debug:
                    print(self.lev_string, self.mol, self.iso)
                    print('NAUUiii------------- EMM')
                    print(self.ctype)

                    coil = sbm.find_molec_metadata(6, 1)
                    Qch4 = CalcPartitionSum(6, 1, temp = 296.0)

                    freqs = np.array([lin.Freq for lin in lin_ok_levup])
                    ch4 = pickle.load(open('./ch4_iso1_LTE.pic','r'))

                    isoab = coil['iso_ratio']

                    for lev in ch4.iso_1.levels:
                        levello = getattr(ch4.iso_1, lev)
                        if levello.equiv(self.lev_string):
                            E_vib = levello.energy
                            pop = Boltz_ratio_nodeg(E_vib, Temp)/Qch4
                            for lin,Gco in zip(lin_ok_levup, G_coeffs_tot):
                                stren = isoab*Gco*pop
                                stren_ab, stren_em = lin.CalcStrength_from_Einstein(Temp, Qch4, iso_ab = isoab, isomolec = ch4.iso_1)
                                print(('{:12.3f}'+5*'{:12.3e}'+2*'{:12.3f}').format(lin.Freq, stren_ab, stren_em, stren, stren/stren_ab, stren_em/stren_ab, lin.g_lo, lin.g_up))

            elif self.ctype == 'absorption':
                lin_ok_levlo = [lin for lin in lines_new if lin.Mol == self.mol and lin.Iso == self.iso and self.lev_string == lin.minimal_level_string_lo()]
                shapes_tot = [lin.shape for lin in lin_ok_levlo]
                G_coeffs_tot = [lin.G_coeffs[self.ctype] for lin in lin_ok_levlo]
                if debug:
                    print(self.lev_string, self.mol, self.iso)
                    print('NAUUiii------------- ABSS')
                    print(self.ctype)

                    coil = sbm.find_molec_metadata(6, 1)
                    Qch4 = CalcPartitionSum(6, 1, temp = Temp)

                    freqs = np.array([lin.Freq for lin in lin_ok_levlo])
                    ch4 = pickle.load(open('./ch4_iso1_LTE.pic','r'))

                    isoab = coil['iso_ratio']

                    for lev in ch4.iso_1.levels:
                        levello = getattr(ch4.iso_1, lev)
                        if levello.equiv(self.lev_string):
                            E_vib = levello.energy
                            pop = Boltz_ratio_nodeg(E_vib, Temp)/Qch4
                            for lin,Gco in zip(lin_ok_levlo, G_coeffs_tot):
                                stren = isoab*Gco*pop
                                stren_ab, stren_em = lin.CalcStrength_from_Einstein(Temp, Qch4, iso_ab = isoab, isomolec = ch4.iso_1)
                                print(('{:12.3f}'+5*'{:12.3e}'+2*'{:12.3f}').format(lin.Freq, stren_ab, stren_em, stren, stren/stren_ab, stren_em/stren_ab, lin.g_lo, lin.g_up))
            #     shapes_tot = [lin.shape for lin in lines_new if (self.lev_string == lin.minimal_level_string_up() and lin.Mol == self.mol and lin.Iso == self.iso)]
            #     G_coeffs_tot = [lin.G_coeffs[self.ctype] for lin in lines_new if (self.lev_string == lin.minimal_level_string_up() and lin.Mol == self.mol and lin.Iso == self.iso)]
            # elif self.ctype == 'absorption':
            #     shapes_tot = [lin.shape for lin in lines_new if (self.lev_string == lin.minimal_level_string_lo() and lin.Mol == self.mol and lin.Iso == self.iso)]
            #     G_coeffs_tot = [lin.G_coeffs[self.ctype] for lin in lines_new if (self.lev_string == lin.minimal_level_string_lo() and lin.Mol == self.mol and lin.Iso == self.iso)]
            else:
                raise ValueError('ctype has to be one among {}, {} and {}. {} not recognized'.format(ctypes[0],ctypes[1],ctypes[2],self.ctype))
        else:
            try:
                shapes_tot = [lin.shape for lin in lines_new if (lin.Mol == self.mol and lin.Iso == self.iso)]
                G_coeffs_tot = [lin.G_coeffs[self.ctype] for lin in lines_new if (lin.Mol == self.mol and lin.Iso == self.iso)]
            except:
                raise ValueError('ctype has to be one among {}, {} and {}. {} not recognized'.format(ctypes[0],ctypes[1],ctypes[2],self.ctype))

        if len(shapes_tot) > 0:
            maxi = [np.max(hap.spectrum*gco) for hap,gco in zip(shapes_tot, G_coeffs_tot)]
            mini = [np.min(hap.spectrum*gco) for hap,gco in zip(shapes_tot, G_coeffs_tot)]
            #print('cazzzzuuuuuuuuuuuuuuuu {} {}'.format(max(maxi),min(mini)))
            self.add_lines_to_spectrum(shapes_tot, Strengths = G_coeffs_tot)
            maxi = [np.max(hap.spectrum*gco) for hap,gco in zip(shapes_tot, G_coeffs_tot)]
            mini = [np.min(hap.spectrum*gco) for hap,gco in zip(shapes_tot, G_coeffs_tot)]
            #print('cazzzzuuuuuuuuuuuuuuuu {} {}'.format(max(maxi),min(mini)))

            if debug: print('iiiiiiiiiiiiiiiiiiiiii {} {} {}'.format(self.ctype, np.max(self.spectrum),np.min(self.spectrum)))

        self.temp = Temp
        self.pres = Pres

        return self.spectrum


    def calc_shapes(self, lines, Temp, Pres, isomolec):
        """
        Calculates the line shapes for the selected type of coefficient.
        """

        lines_new = calc_shapes_lines(self.spectral_grid, lines, Temp, Pres, isomolec)

        return lines_new

    def interpolate(self, coeff2, Pres = None, Temp = None):
        """
        Interpolates self with another Gcoeff coeff2, considering the parameters given. The two coeffs should either have the same pressure (and in this case the desired Temp is given in the call) or the same temperature (Pres is given in the call).
        """
        if not sbm.isclose(self.temp, coeff2.temp) and not sbm.isclose(self.pres, coeff2.pres):
            raise ValueError('The two coeffs have both different temperatures and pressures! cannot interpolate')

        if Pres is not None:
            if not sbm.isclose(self.temp, coeff2.temp):
                raise ValueError('The two coeffs have different temperatures! You should specify the interpolation temperature, not the pressure')
            w1, w2 = sbm.weight(Pres, self.pres, coeff2.pres, itype='exp')
            new_spect = w1 * self.spectrum + w2 * coeff2.spectrum
            gigi = SpectralGcoeff(self.ctype, self.spectral_grid, self.mol, self.iso, self.MM, self.lev_string, spectrum = new_spect, Pres = Pres, Temp = self.temp)
        elif Temp is not None:
            if not sbm.isclose(self.pres, coeff2.pres):
                raise ValueError('The two coeffs have different pressures! You should specify the interpolation pressure, not the temperature')
            w1, w2 = sbm.weight(Temp, self.temp, coeff2.temp, itype='lin')
            new_spect = w1 * self.spectrum + w2 * coeff2.spectrum
            gigi = SpectralGcoeff(self.ctype, self.spectral_grid, self.mol, self.iso, self.MM, self.lev_string, spectrum = new_spect, Pres = self.pres, Temp = Temp)

        return gigi


def calc_shapes_lines(wn_arr, lines, Temp, Pres, isomolec):
    """
    Calculates the line shapes and attaches new attributes "shape" and "Gcoeffs" to the line objects. line.Gcoeffs is a dict with three elements: sp_emission, ind_emission and absorption. Multiplying line.shape by line.Gcoeffs one has the three SpectralGcoeffs contributions of each line.
    """

    #if not isomolec.is_in_LTE:
    if len(isomolec.levels) > 0:
        oks = []
        for line in lines:
            oks.append(line.LinkToMolec(isomolec))
        lines = [lin for lin, ok in zip(lines,oks) if ok]

    time0 = time.time()

    processi = []
    coda = []
    outputs = []

    for i in range(n_threads):
        coda.append(Queue())
        processi.append(Process(target=do_for_th_calc,args=(wn_arr, lines, Temp, Pres, isomolec, i, coda[i])))
        processi[i].start()

    for i in range(n_threads):
        outputs.append(coda[i].get())

    for i in range(n_threads):
        processi[i].join()

    lines_new = []

    for output in outputs:
        lines_new += output

    return lines_new


def do_for_th_calc(wn_arr, linee_tot, Temp, Pres, isomolec, i, coda):
    """
    Single process called by calc_shapes_lines.
    """
    time0 = time.time()
    step_nlin = len(linee_tot)/n_threads
    linee = linee_tot[step_nlin*i:step_nlin*(i+1)]
    if i == n_threads-1:
        linee = linee_tot[step_nlin*i:]

    #print('Hey! Questo è il ciclo {} con {} linee su {}!'.format(i,len(linee),len(linee_tot)))

    lines_new = PrepareCalcShapes(wn_arr, linee, Temp, Pres, isomolec)

    #print('Ciclo {} concluso in {} s!'.format(i,time.time()-time0))

    coda.put(lines_new)

    return


def PrepareCalcShapes(wn_arr, linee_mol, Temp, Pres, isomolec):
    """
    Core of the calculation in do_for_th_calc.
    """

    sp_step = wn_arr.step()
    lin_grid = np.arange(-imxsig*sp_step/2,imxsig*sp_step/2,sp_step, dtype = float)

    time0 = time.time()
    time_100 = time.time()

    #print(type(isomolec.MM), isomolec.MM)

    for ii,lin in zip(range(len(linee_mol)),linee_mol):
        ind_ok, fr_grid_ok = closest_grid(wn_arr, lin.Freq)
        lin_grid_ok = SpectralGrid(lin_grid+fr_grid_ok, units = 'cm_1')

        lin.MakeShapeLine(Temp, Pres, grid = lin_grid_ok, MM = isomolec.MM, keep_memory = True)
        lin.Calc_Gcoeffs(Temp, isomolec = isomolec)

    #print('Made {} lines in {} s'.format(len(linee_mol),time.time()-time0))

    return linee_mol


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


def read_line_database(nome_sp, mol = None, iso = None, up_lev = None, down_lev = None, fraction_to_keep = None, db_format = 'HITRAN', freq_range = None, n_skip = 0, link_to_isomolecs = None, verbose = False):
    """
    Reads line spectral data.
    :param nome_sp: spectral data file
    :param mol: HITRAN molecule number
    :param iso: HITRAN iso number
    :param up_lev: Upper level HITRAN string
    :param down_lev: Lower level HITRAN string
    :param format: If 'gbb' the format of MAKE_MW is used in reading, if 'HITRAN' the HITRAN2012 format.
    :param fraction_to_keep: default is 1.0. If lower keeps just that fraction of the lines, selected for line_strength. Could be problematic in non-LTE studies.
    returns:
    list of SpectLine objects.
    """

    if db_format == 'gbb':
        delim = (2, 1, 12, 10, 10, 6, 6, 10, 4, 8, 15, 15, 15, 15)
        cose = ('Mol', 'Iso', 'Freq', 'Strength', 'A_coeff', 'Air_broad', 'Self_broad', 'E_lower', 'T_dep_broad', 'P_shift', 'Up_lev_str', 'Lo_lev_str', 'Q_num_up', 'Q_num_lo')
        cose2 = 2 * 'i4,' + 8 * 'f8,' + 3 * '|S15,' + '|S15'
    elif db_format == 'HITRAN':
        delim = (2, 1, 12, 10, 10, 5, 5, 10, 4, 8, 15, 15, 15, 15, 19, 7, 7)
        cose = ('Mol', 'Iso', 'Freq', 'Strength', 'A_coeff', 'Air_broad', 'Self_broad', 'E_lower', 'T_dep_broad', 'P_shift', 'Up_lev_str', 'Lo_lev_str', 'Q_num_up', 'Q_num_lo', 'others', 'g_up', 'g_lo')
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
        if verbose: print(linea['Mol'], linea['Iso'], linea['Freq'])
        if freq_range is not None:
            if linea['Freq'] < freq_range[0]:
                continue
            if linea['Freq'] > freq_range[1]:
                break
        if (linea['Mol'] == mol or mol is None) and (linea['Iso'] == iso or iso is None) and (linea['Up_lev_str'] == up_lev or up_lev is None) and (linea['Lo_lev_str'] == down_lev or down_lev is None):
            line = SpectLine(linea)
            if verbose: print(linea)
            if link_to_isomolecs is not None:
                IsoMolecol = [molecolo for molecolo in link_to_isomolecs if (molecolo.mol == mol and molecolo.iso == iso)]
                if len(IsoMolecol) > 1:
                    raise ValueError('Multiple levels corresponding to line! WTF?')
                line.LinkToMolec(IsoMolecol)
            linee_ok.append(line)
    #print('Ho creato lista di oggetti linea??\n')

    if fraction_to_keep is not None:
        essesss = [lin.Strength for lin in linee_ok]
        essort = np.sort(np.array(essesss))[int(fraction_to_keep*(len(linee_ok)-1))]

        linee_sel = [lin for lin in linee_ok if lin.Strength >= essort]
        #print('The threshold for line Strength is {}. Selected {} out of {} lines.'.format(essort,len(linee_sel),len(linee_ok)))
    else:
        linee_sel = linee_ok

    infi.close()
    return linee_sel


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

    poli = lagrange(x,qg)

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


def Einstein_A_to_B(A_coeff, wavenumber, units = 'cm3ergcm2'):
    """
    Calculates the Eintein B coeff for induced absorption. B is expressed in m^3/J*s and is defined so that in the expression for the light absorption it appears with the radiation density rho (not with the number of photons, other possible definition for B).
    """
    nu = convertto_hz(wavenumber, 'cm_1')

    fact = 8*np.pi*const.constants.h*nu**3/const.constants.c**3
    fact_2 = 2*h_cgs*c_cgs**2*wavenumber**3
    fact_3 = 2*h_cgs*nu**3/c_cgs**2
    #print(h_cgs,wavenumber,fact_2,c_cgs)

    if units == 'm3Js': # SI units
        B_coeff = A_coeff/fact
    elif units == 'cm3ergcm2': # my units (freq in cm-1)
        B_coeff = A_coeff/fact_2
    elif units == 'cm3ergs': # Vardavas units (freq in Hz)
        B_coeff = A_coeff/fact_3

    return B_coeff


def Einstein_B21_to_B12(B_21, g_1, g_2):
    """
    Calculates the inverse Eintein B coeff for induced absorption.
    """

    B_12 = B_21*g_2/g_1

    return B_12


def Einstein_A_to_LineStrength_nonLTE(A_coeff, wavenumber, E_lower, T_vib_lower, T_vib_upper, g_lower, g_upper, Q_part, iso_ab = 1.0):
    """
    Calculates the line strength in non-LTE.
    """

    B_21 = Einstein_A_to_B(A_coeff, wavenumber, units = 'cm3ergcm2')
    B_12 = Einstein_B21_to_B12(B_21, g_lower, g_upper)
    E_upper = E_lower + wavenumber

    fact = g_lower/g_upper * Boltz_pop_at_T(E_upper, T_vib_upper, g_upper, Q_part) / Boltz_pop_at_T(E_lower, T_vib_lower, g_lower, Q_part)

    pop_low = Boltz_pop_at_T(E_lower, T_vib_lower, g_lower, Q_part)

    S = iso_ab * h_cgs*wavenumber*c_cgs * pop_low * (1 - fact) * B_12 / (4*np.pi)

    return S


def Einstein_A_to_Gcoeff_abs(line, Temp, E_vib):
    """
    Calculates the absorption contribution to the G_coeff for a SINGLE LINE with the level as LOWER level of the transition.
    G_coeff = hcw/(4pi) * \sum_lines(B_12*Phi(T,P)) --> the sum is on all the lines with LEVEL as LOWER level of the transition.
    """

    B_21 = line.Einstein_A_to_B()

    B_12 = Einstein_B21_to_B12(B_21, line.g_lo, line.g_up)
    rot_pop = line.g_lo * Boltz_ratio_nodeg(line.E_lower-E_vib, Temp)

    G_co = h_cgs*c_cgs*line.Freq * rot_pop * B_12 / (4*np.pi)

    return G_co


def Einstein_A_to_Gcoeff_indem(line, Temp, E_vib):
    """
    Calculates the induced emission contribution to the G_coeff for a SINGLE LINE with the level as UPPER level of the transition.
    G_coeff = hcw/(4pi) * \sum_lines(B_21*Phi(T,P)) --> the sum is on all the lines with LEVEL as UPPER level of the transition.
    """
    B_21 = line.Einstein_A_to_B()

    rot_pop = line.g_up * Boltz_ratio_nodeg(line.E_lower+line.Freq-E_vib, Temp)
    G_co = h_cgs*c_cgs*line.Freq * rot_pop * B_21 / (4*np.pi)

    return G_co


def Einstein_A_to_Gcoeff_spem(line, Temp, E_vib):
    """
    Calculates the spontaneous emission contribution to the G_coeff for a SINGLE LINE with the level as UPPER level of the transition.
    G_coeff = hcw/(4pi) * \sum_lines(A_21*Phi(T,P)) --> the sum is on all the lines with LEVEL as UPPER level of the transition.
    """
    rot_pop = line.g_up * Boltz_ratio_nodeg(line.E_lower+line.Freq-E_vib, Temp)
    G_co = h_cgs*c_cgs*line.Freq * rot_pop * line.A_coeff / (4*np.pi)

    return G_co


def Einstein_A_to_LineStrength_hitran(A_coeff, wavenumber, temp, Q_part, g_upper, E_lower, iso_ab = 1.0):
    """
    Calculates the line strength at LTE in units cm-1/(mol cm-2).
    """

    S = iso_ab * A_coeff * g_upper * np.exp(-c2*E_lower/temp)*(1-np.exp(-c2*wavenumber/temp)) / (8 * np.pi * c_cgs * wavenumber**2 * Q_part)

    return S


def Boltz_pop_at_T(wavenumber, temp, g_level, Q_part):
    """
    Boltzmann statistics LTE population of level.
    """

    enne = g_level * Boltz_ratio_nodeg(wavenumber, temp) / Q_part

    return enne


def Boltz_ratio_nodeg(wavenumber, temp):
    ratio = np.exp(-c2*wavenumber/temp)
    return ratio


def Calc_BB(spectral_grid, T, units = 'ergscm2'):
    """
    Returns a SpectralIntensity object with a Planck spectrum at temp T in the range considered.
    """

    spectrum = 2*h_cgs*c_cgs**2 * (spectral_grid.grid)**3 / (np.exp(c2*spectral_grid.grid/T)-1)
    BBfu = SpectralIntensity(spectrum, spectral_grid, units = 'ergscm2')

    if units != 'ergscm2':
        BBfu.convertto(units)

    return BBfu


def Lorentz_shape(wn,wn_0,lw):
    """
    Returns a lorentzian shape on the np.array of wavenumbers wn, with central wavenumber wn_0 and width lw.
    """

    ls = 1/np.pi * lw/(lw**2 + (wn-wn_0)**2)

    return ls


def Doppler_shape(wn,wn_0,dw):
    """
    Returns a gaussian shape on the np.array of wavenumbers wn, with central wavenumber wn_0 and width dw.
    """

    ds = mt.sqrt(mt.log(2.0)/(np.pi*dw**2)) * np.exp(-(wn-wn_0)**2*mt.log(2.0)/dw**2)

    return ds


def gaussian(arr,mu,sig):
    """
    normalized gaussian function on array.
    """
    fac = 1/(sig*np.sqrt(2.*np.pi))

    gauss = fac*np.exp(-0.5*((arr-mu)/sig)**2)

    return gauss


def closest_grid(wn_arr,wn_0):
    """
    Returns the value and the index of the grid point closest to wn_0.
    """
    ind = np.argmin(np.abs(wn_arr.grid-wn_0))

    return ind, wn_arr.grid[ind]


def Lorenz_width(Temp, Pres_atm, T_dep_broad, Air_broad, Self_broad = 0.0, Self_pres_atm=0.0):
    """
    Pressure in atmospheres.
    rlhalf=rhw0(mline,imw) * (rp1/rp0h) * (rt0h/rt1)**rexph(mline,imw)
    """
    lw = (T_ref/Temp)**T_dep_broad * (Air_broad*(Pres_atm-Self_pres_atm) + Self_broad*Self_pres_atm)

    return lw

def Doppler_width(Temp,MM,wn_0):
    """
    MM is the molecular mass in atomic units.
    !!! ATTENZIONE !!!
    As input for the original humliv_bb dw/sqrt(ln2) is taken, not dw!
    *  rdsqln = rdhalf / sqrt(ln 2)
    """

    dw = wn_0/c_cgs * mt.sqrt(2*const.Avogadro*k_cgs*Temp*mt.log(2.0)/MM)

    return dw



def MakeShape(wn_arr, wn_0, lw, dw, Strength = 1.0):
    """
    Returns the line shape array as a SpectralObject class.
    --------------------------------------------------------------------------------------------
    PAY ATTENTION: while the inputs for MakeShape are lw and dw, the humliv_bb routine takes lw and dw/sqrt(ln2). Moreover, the lineshape is multiplied by a factor (fac below) which is maybe useful for optimization.
    However, the shape returned by MakeShape is normalized -> integral(shape) = 1.
    """
    fac = float(dw*mt.sqrt(np.pi/mt.log(2.0)))

    y = lineshape.humliv_bb(wn_arr.grid,1,len(wn_arr.grid),wn_0,lw,dw/mt.sqrt(mt.log(2.0)))

    #### PUT 1 HERE!! FORTRAN ARRAYS START FROM ONEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE -> 1

    y = Strength*y/fac

    shape = SpectralObject(y, wn_arr)
    #print('IIIIIIII ', shape.integrate())

    return shape


def MakeShape_py(wn_arr, wn_0, lw, dw, Strength = 1.0):
    """
    Returns the line shape array as a SpectralObject class.
    """
    fac = float(dw*mt.sqrt(np.pi/mt.log(2.0)))

    Lorentz = Lorentz_shape(wn_arr.grid-wn_0, 0.0, lw)
    Doppler = Doppler_shape(wn_arr.grid-wn_0, 0.0, dw)

    step = wn_arr.step()

    y2 = np.convolve(Doppler, Lorentz, mode = 'same')*step*Strength

    shape = SpectralObject(y2, wn_arr)

    return shape


def convert_to_atm(Pres, units = 'hPa'):
    """
    Converts pressure to atmospheres.
    """
    if units == 'hPa':
        Pres_atm = Pres * hpa_to_atm

    return Pres_atm


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
