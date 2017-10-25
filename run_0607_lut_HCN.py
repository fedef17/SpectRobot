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

planet = pickle.load(open(inputs['cart_LUTS']+'planet_3D_chc_e_LTEgases.pic', 'r'))

db_file = inputs['hitran_db']

########################################################

LUTopt = dict()
LUTopt['max_pres'] = 2.0 # hPa circa 120 km
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

linee = spcl.read_line_database(db_file, freq_range = wn_ranges['CH4'])
linee = smm.check_lines_mols(linee, [planet.gases['CH4']])

nuca = '/work/localuser/fedef/SPECT_ROBOT_RUN/CH4_newband/'
#nuca = '/home/fedefab/Scrivania/Research/Dotto/Spect_data/CH4_newband/'
linee_0020 = spcl.read_line_database(nuca+'P4mP2_prediction_0020-0010')
linee_0012 = spcl.read_line_database(nuca+'P4mP2_prediction_0012-0002')
linee_0111 = spcl.read_line_database(nuca+'P4mP2_prediction_0111-0101')

linee_0020_sel = [lin for lin in linee_0020 if lin.Strength > 1.e-29]
linee_0012_sel = [lin for lin in linee_0020 if lin.Strength > 1.e-29]
linee_0111_sel = [lin for lin in linee_0020 if lin.Strength > 1.e-28]

linee += linee_0020_sel
linee += linee_0012_sel
linee += linee_0111_sel

abs_coeff = smm.prepare_spe_grid(wn_ranges['CH4'])
sp_grid = abs_coeff.spectral_grid

LUTS = smm.check_and_build_allluts(inputs, sp_grid, linee, [planet.gases['CH4']], atmosphere = planet.atmosphere, LUTopt = LUTopt)

#PTtest = [[3.0, 170.], [4.0, 160.], [2.0, 110.]]
# PTtest = [[3.0, 170.], [4.0, 160.]]
#LUTS = smm.check_and_build_allluts(inputs, sp_grid, linee, planet.gases.values(), PTcouples = PTtest, LUTopt = LUTopt)

print(time.ctime())
print('CIAO!')
