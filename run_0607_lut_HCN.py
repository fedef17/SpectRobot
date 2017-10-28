#!/usr/bin/python
# -*- coding: utf-8 -*-

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
import copy
import time
import spect_main_module as smm
from multiprocessing import Process, Queue

### Program for CH4 HCN and C2H2 climatology on TITAN from VIMS spectra

input_file = 'inputs_spect_robot_LUTS.in'

cartatm = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/Tit_atm/climat_0607_manuel/p-T_clima2_07-07/'

cart_LUTS = '/media/hd_B/Spect_data/LUTs/'

hit12_25 = '/home/fedefab/Scrivania/Research/Dotto/Spect_data/HITRAN/HITRAN2012_2-5mu.par'

keys = 'cart_atm cart_LUTS hitran_db n_threads test'
keys = keys.split()
itype = [str, str, str, int, bool]
defaults = [cartatm, cart_LUTS, hit12_25, 8, False]
inputs = sbm.read_inputs(input_file, keys, itype = itype, defaults = defaults)

time0 = time.time()

### LOADING PLANET
print('Loading planet...')

planet = sbm.Titan(1500.)

# planet = pickle.load(open(inputs['cart_LUTS']+'planet_1D_chc_e_LTEgases.pic', 'r'))
planet = pickle.load(open('planet_1D_chc_e_LTEgases.pic', 'r'))

db_file = inputs['hitran_db']

########################################################

LUTopt = dict()
LUTopt['max_pres'] = 0.1 # hPa circa 120 km
LUTopt['temp_step'] = 5.
LUTopt['pres_step_log'] = 1.0

wn_ranges = dict()
wn_ranges['HCN'] = [3200.,3400.]
wn_ranges['C2H2'] = [3175.,3375.]
wn_ranges['CH4'] = [2825.,3225.]

# for gas in planet.gases:
#     print(gas)
#     linee = spcl.read_line_database(db_file, freq_range = wn_ranges[gas])
#     linee = smm.check_lines_mols(linee, [planet.gases[gas]])
#     abs_coeff = smm.prepare_spe_grid(wn_ranges[gas])
#     sp_grid = abs_coeff.spectral_grid
#
#     LUTS = smm.check_and_build_allluts(inputs, sp_grid, linee, [planet.gases[gas]], atmosphere = planet.atmosphere, LUTopt = LUTopt)

ch4 = planet.gases['CH4']
ch4.del_iso('iso_2')
ch4.del_iso('iso_3')
linee = spcl.read_line_database(db_file, freq_range = wn_ranges['CH4'])
linee = smm.check_lines_mols(linee, [ch4])

PTcouples = smm.calc_PT_couples_atmosphere(linee, [ch4], planet.atmosphere, **LUTopt)

nuca = '/work/localuser/fedef/SPECT_ROBOT_RUN/CH4_newband/'
linee += spcl.read_line_database(nuca+'CH4_corrected_sel_0020.dat', freq_range = wn_ranges['CH4'])
linee += spcl.read_line_database(nuca+'CH4_corrected_sel_0012.dat', freq_range = wn_ranges['CH4'])
linee += spcl.read_line_database(nuca+'CH4_corrected_sel_0111.dat', freq_range = wn_ranges['CH4'])

abs_coeff = smm.prepare_spe_grid(wn_ranges['CH4'])
sp_grid = abs_coeff.spectral_grid

LUTS = smm.check_and_build_allluts(inputs, sp_grid, linee, [ch4], atmosphere = planet.atmosphere, LUTopt = LUTopt)

nuca2 = '/work/localuser/fedef/SPECT_ROBOT_RUN/HCN_newband/'
linee = spcl.read_line_database(nuca2+'HCN_new_hitcomplete.dat', freq_range = wn_ranges['HCN'])
hcn = planet.gases['HCN']
linee = smm.check_lines_mols(linee, [hcn])

abs_coeff = smm.prepare_spe_grid(wn_ranges['HCN'])
sp_grid = abs_coeff.spectral_grid

LUTS = smm.check_and_build_allluts(inputs, sp_grid, linee, [hcn], atmosphere = planet.atmosphere, LUTopt = LUTopt)

#PTtest = [[3.0, 170.], [4.0, 160.], [2.0, 110.]]
# PTtest = [[3.0, 170.], [4.0, 160.]]
#LUTS = smm.check_and_build_allluts(inputs, sp_grid, linee, planet.gases.values(), PTcouples = PTtest, LUTopt = LUTopt)

print(time.ctime())
print('CIAO!')
