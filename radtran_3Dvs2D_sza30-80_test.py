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
from memory_profiler import profile

time0 = time.time()
print(time.ctime())

input_file = 'inputs_spect_robot_3D.in'

cart_LUTS = '/media/hd_B/Spect_data/LUTs/'

hit12_25 = '/home/fedefab/Scrivania/Research/Dotto/Spect_data/HITRAN/HITRAN2012_2-5mu.par'

keys = 'cart_atm cart_observed cart_tvibs cart_tvibs2 cart_tvibs3 cart_LUTS out_dir hitran_db n_threads test n_split cart_input_1D cart_inputs'
keys = keys.split()
itype = [str, str, str, str, str, str, str, str, int, bool, int, str, str]
defaults = [cart_atm, None, cart_tvibs, cart_tvibs2, cart_tvibs3, cart_LUTS, None, hit12_25, 8, False, 5, cart_input_1D, None]
inputs = sbm.read_inputs(input_file, keys, itype = itype, defaults = defaults, verbose = True)

if not os.path.exists(inputs['cart_LUTS']):
    raise MemoryError('Disk not mounted or wrong path: '+inputs['cart_LUTS'])

if not os.path.exists(inputs['out_dir']):
    raise MemoryError('Disk not mounted or wrong path: '+inputs['out_dir'])

sbm.check_free_space(inputs['cart_LUTS'])
sbm.check_free_space(inputs['out_dir'])

time0 = time.time()

### LOADING PLANET
print('Loading planet...')

planet1D = pickle.load(open(inputs['cart_inputs']+'planet_1D.pic'))
planet3D = pickle.load(open(inputs['cart_inputs']+'planet_3D.pic'))

wn_range = [2850.,3450.]
wn_range_obs = [spcl.convertto_nm(wn_range[1], 'cm_1')+10., spcl.convertto_nm(wn_range[0], 'cm_1')-10.]
print(wn_range_obs)

radtran_opt = dict()
radtran_opt['max_T_variation'] = 5.
radtran_opt['max_Plog_variation'] = 1.

print('Loading lines...')
db_file = inputs['hitran_db']
linee = spcl.read_line_database(db_file, freq_range = wn_range)

# planet = pickle.load(open(inputs['cart_tvibs']+'planet.pic'))

linee = smm.check_lines_mols(linee, planet3D.gases.values())
smm.keep_levels_wlines(planet3D, linee)
smm.keep_levels_wlines(planet1D, linee)

LUTopt = dict()
LUTopt['max_pres'] = 2.0 # hPa circa 200 km
LUTopt['temp_step'] = 5.
LUTopt['pres_step_log'] = 1.0

sp_gri = smm.prepare_spe_grid(wn_range).spectral_grid

PTcoup_needed = smm.calc_PT_couples_atmosphere(linee, planet.gases.values(), planet.atmosphere, **LUTopt)

LUTS = smm.check_and_build_allluts(inputs, sp_gri, linee, planet.gases.values(), PTcouples = PTcoup_needed, LUTopt = LUTopt)

# sys.exit()

fil = open(inputs['cart_inputs']+'pix7418_sza80.pic','r')
pixels = pickle.load(fil)
fil.close()

for pix in pixels:
    print('Masking CH4 R branch')
    gri = pix.observation.spectral_grid.grid
    cond = (gri > 3190.) & (gri < 3295.)
    pix.observation.mask[cond] = 0
    pix.observation.noise = copy.deepcopy(pix.observation)
    pix.observation.noise.spectrum = 2.e-8*np.ones(len(pix.observation.spectrum))

for i in range(20):
    print('\n')

bay1 = copy.deepcopy(baybau1D)
time0 = time.time()
teag = '2Dvs3D_sza80_szavar'
dampa = open(inputs['out_dir']+'./out_'+teag+'.pic','wb')
result = smm.inversion_fast_limb(inputs, planet3D, linee, bay1, pixels, wn_range = wn_range, radtran_opt = radtran_opt, debugfile = dampa, LUTopt = LUTopt, nome_inv = teag)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))


for i in range(20):
    print('\n')

bay2 = copy.deepcopy(baybau1D)
time0 = time.time()
teag = '2Dvs3D_sza80_noszavar'
dampa = open(inputs['out_dir']+'./out_'+teag+'.pic','wb')
result = smm.inversion_fast_limb(inputs, planet3D, linee, bay2, pixels, wn_range = wn_range, radtran_opt = radtran_opt, debugfile = dampa, LUTopt = LUTopt, use_tangent_sza = True, nome_inv = teag)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))


fil = open(inputs['cart_inputs']+'pix7418_sza30.pic','r')
pixels = pickle.load(fil)
fil.close()

for pix in pixels:
    print('Masking CH4 R branch')
    gri = pix.observation.spectral_grid.grid
    cond = (gri > 3190.) & (gri < 3295.)
    pix.observation.mask[cond] = 0
    pix.observation.noise = copy.deepcopy(pix.observation)
    pix.observation.noise.spectrum = 2.e-8*np.ones(len(pix.observation.spectrum))


for i in range(20):
    print('\n')

bay1 = copy.deepcopy(baybau1D)
time0 = time.time()
teag = '2Dvs3D_sza30_szavar'
dampa = open(inputs['out_dir']+'./out_'+teag+'.pic','wb')
result = smm.inversion_fast_limb(inputs, planet3D, linee, bay1, pixels, wn_range = wn_range, radtran_opt = radtran_opt, debugfile = dampa, LUTopt = LUTopt, nome_inv = teag)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))


for i in range(20):
    print('\n')

bay2 = copy.deepcopy(baybau1D)
time0 = time.time()
teag = '2Dvs3D_sza30_noszavar'
dampa = open(inputs['out_dir']+'./out_'+teag+'.pic','wb')
result = smm.inversion_fast_limb(inputs, planet3D, linee, bay2, pixels, wn_range = wn_range, radtran_opt = radtran_opt, debugfile = dampa, LUTopt = LUTopt, use_tangent_sza = True, nome_inv = teag)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))

print(time.ctime())
