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

input_file = 'inputs_spect_robot.in'

cartatm = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/Tit_atm/climat_0607_manuel/p-T_clima2_07-07/'

cart_new_ch4 = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/clima_Maya_22_May_2017/'

cart_LUTS = '/home/fedefab/Scrivania/Research/Dotto/Spect_data/LUTs/'

hit08_25 = '/home/fedefab/Scrivania/Research/Dotto/Spect_data/HITRAN/HITRAN2012_2-5mu.par'

keys = 'cart_atm cart_molecs_new cart_LUTS hitran_db n_threads test'
keys = keys.split()
itype = [str, str, str, str, int, bool]
defaults = [cartatm, cart_new_ch4, cart_LUTS, hit08_25, 8, False]
inputs = sbm.read_inputs(input_file, keys, itype = itype, defaults = defaults)

### LOADING LINES
print('Loading lines...')

time0 = time.time()

db_file = inputs['hitran_db']
wn_range = [2900.,3200.]

#linee = spcl.read_line_database(db_file)
linee = spcl.read_line_database(db_file, freq_range = wn_range)

inputs['test'] = False

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

grid = np.meshgrid(lat_c,z2)

TT = np.vstack(temps)
PP = np.vstack(press)

Atm = sbm.AtmProfile(TT, grid, gridname=['Lat','Alt (km)'], interp = ['box','lin'], profname='temp')
Atm.add_profile(PP, 'pres', interp = ['box','exp'])

planet.add_atmosphere(Atm)

### LOADING MOLECULES
print('Loading molecules...')

ch4 = sbm.Molec(6, 'CH4')

ch4.add_iso(1, LTE = False)
alts_vib, molecs, levels, energies, vib_ok = sbm.read_tvib_manuel(inputs['cart_molecs_new']+'vt_ch4__092_2006-07_clima2_00.0e_20.0_Cas_Voy_v3_v10_0061')

ch4.link_to_atmos(Atm)

print('qui')

ch4.iso_1.add_levels(levels, energies, add_fundamental = True)
ch4.iso_1.add_simmetries_levels(linee)

ch4.add_iso(2, LTE = False)
alts_vib, molecs, levels, energies, vib_ok = sbm.read_tvib_manuel(inputs['cart_molecs_new']+'vt_ch4__092_2006-07_clima2_00.0e_20.0_Cas_Voy_v3_v10_0062')
ch4.iso_2.add_levels(levels, energies, add_fundamental = True)
ch4.iso_2.add_simmetries_levels(linee)

print(levels, energies)

#ch4.add_iso(3, MM = 17, ratio = 6.158e-4, LTE = True)
ch4.add_all_iso_from_HITRAN(linee)

# hcn = sbm.Molec(23, 'HCN')
# hcn.add_all_iso_from_HITRAN(linee)
# hcn.link_to_atmos(atm_old)
# hcn.add_clim(atm_gases_old['HCN'])
#
# c2h2 = sbm.Molec(26, 'C2H2')
# c2h2.add_all_iso_from_HITRAN(linee)
# c2h2.link_to_atmos(atm_old)
# c2h2.add_clim(atm_gases_old['C2H2'])

planet.add_gas(ch4)
# planet.add_gas(hcn)
# planet.add_gas(c2h2)

pickle.dump(planet, open(inputs['cart_molecs_new']+'ch4_new_ref.pic','w'))

if inputs['test']:
    print('Keeping ONLY 1000 linee for testing')
    linee = linee[:1000]

#planet = pickle.load(open(inputs['cart_molecs_new']+'ch4_old_ref.pic','r'))

#sys.exit()
##########################################################
##########################################################
###########################################################

abs_coeff = smm.prepare_spe_grid(wn_range)
LUTS = smm.makeLUT_nonLTE_Gcoeffs(abs_coeff.spectral_grid, linee, planet.gases.values(), planet.atmosphere, pres_step_log = 0.4, cartLUTs = inputs['cart_LUTS'], n_threads = inputs['n_threads'], test = inputs['test'])
#
pickle.dump(LUTS, open(inputs['cart_LUTS']+'_allLUTS'+smm.date_stamp(),'w') )
print(time.ctime())
print('CIAO!')
