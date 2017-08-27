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

planet = sbm.Titan()

lat_bands = ['SP','subPS','TS','EQ','TN','subPN','NP']
lat_ext = [-90.,-75.,-60.,-30.,30.,60.,75.,90.]
lat_c = [(cos+cos2)/2.0 for cos,cos2 in zip(lat_ext[:-1],lat_ext[1:])]

temps = []
press = []
for i,band,minl,maxl in zip(range(len(lat_bands)),lat_bands,lat_ext[:-1],lat_ext[1:]):
    print('Band {} from lat {} to lat {}'.format(band,minl,maxl))
    z2,T,P = sbm.read_input_atm_man(inputs['cart_atm']+'pt_clima2_06-07_lat{:1d}.prf'.format(i+1))
    temps.append(T)
    press.append(P)

#grid = np.meshgrid(lat_c,z2)
grid = sbm.AtmGrid(['lat', 'alt'], [lat_ext[:-1], z2])

TT = np.vstack(temps)
PP = np.vstack(press)

Atm = sbm.AtmProfile(grid, TT, 'temp', ['box','lin'])
Atm.add_profile(PP, 'pres', ['box','exp'])

planet.add_atmosphere(Atm)

### LOADING LINES
print('Loading lines...')

db_file = inputs['hitran_db']
wn_range = [3175.,3450.]

#linee = spcl.read_line_database(db_file)
linee = spcl.read_line_database(db_file, freq_range = wn_range)

### LOADING MOLECULES
print('Loading molecules...')

hcn = sbm.Molec(23, 'HCN')
hcn.add_all_iso_from_HITRAN(linee, add_levels = True)
hcn.link_to_atmos(Atm)
#hcn.add_clim(atm_gases_old['HCN'])

c2h2 = sbm.Molec(26, 'C2H2')
c2h2.add_all_iso_from_HITRAN(linee, add_levels = True)
c2h2.link_to_atmos(Atm)
#c2h2.add_clim(atm_gases_old['C2H2'])

print('hcn')
for iso in hcn.all_iso:
    print(iso)
    isomol = getattr(hcn, iso)
    print(isomol.levels)

print('c2h2')
for iso in hcn.all_iso:
    print(iso)
    isomol = getattr(c2h2, iso)
    print(isomol.levels)

planet.add_gas(hcn)
planet.add_gas(c2h2)

# pickle.dump(planet, open(inputs['cart_LUTS']+'planet_hcn_c2h2.pic', 'w'))
# planet = pickle.load(open(inputs['cart_LUTS']+'planet_hcn_c2h2.pic', 'r'))


linee = [lin for lin in linee if lin.Freq >= wn_range[0] and lin.Freq <= wn_range[1]]

if inputs['test']:
    print('Keeping ONLY 10 linee for testing')
    linee = linee[:10]

########################################################

LUTopt = dict()
LUTopt['max_pres'] = 5.0 # hPa circa 120 km

abs_coeff = smm.prepare_spe_grid(wn_range)
sp_grid = abs_coeff.spectral_grid

# LUTS = smm.check_and_build_allluts(inputs, sp_grid, linee, planet.gases.values(), atmosphere = planet.atmosphere, LUTopt = LUTopt)
PTtest = [[3.0, 170.], [4.0, 160.]]
LUTS = smm.check_and_build_allluts(inputs, sp_grid, linee, planet.gases.values(), PTcouples = PTtest, LUTopt = LUTopt)

print(time.ctime())
print('CIAO!')
