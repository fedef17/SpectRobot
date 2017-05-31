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

cart_old_ch4 = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/Old_vts/INP_TEST_OLDs/'

cart_LUTS = '/home/fedefab/Scrivania/Research/Dotto/Spect_data/LUTs/'

hit08_25 = '/home/fedefab/Scrivania/Research/Dotto/Spect_data/HITRAN/HITRAN08_2-5mu.par'

keys = 'cart_atm cart_molecs cart_LUTS hitran_db n_threads test'
keys = keys.split()
itype = [str, str, str, str, int, bool]
defaults = [cartatm, cart_old_ch4, cart_LUTS, hit08_25, 8, False]
inputs = sbm.read_inputs(input_file, keys, itype = itype, defaults = defaults)

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

temp_old = sbm.read_input_prof_gbb(inputs['cart_molecs'] + 'in_temp.dat', 'temp')
pres_old = sbm.read_input_prof_gbb(inputs['cart_molecs'] + 'in_pres.dat', 'pres')
zold = np.linspace(0.,1500.,151)

ch4 = sbm.Molec(6, 'CH4', MM=12)

ch4.add_iso(1, MM = 16.04, ratio = 0.9883)
alts_vib, molecs, levels, energies, vib_ok = sbm.read_tvib_manuel(inputs['cart_molecs']+'vt_ch4__029_2006_t15_10.2s_29.9_vmrA2_v10_0061')

atm_old = sbm.AtmProfile(np.interp(alts_vib,zold,temp_old),np.array(alts_vib),profname='temp')
atm_old.add_profile(np.exp(np.interp(alts_vib,zold,np.log(pres_old))), 'pres', interp = 'exp')
ch4.link_to_atmos(atm_old)

print('qui')

ch4.iso_1.add_levels(levels, energies, vibtemps=vib_ok, add_fundamental = True, T_kin = atm_old.temp)

ch4.add_iso(2, MM = 17, ratio = 0.0111)
alts_vib, molecs, levels, energies, vib_ok = sbm.read_tvib_manuel(inputs['cart_molecs']+'vt_ch4__029_2006_t15_10.2s_29.9_vmrA2_v10_0062')
ch4.iso_2.add_levels(levels, energies, vibtemps=vib_ok, add_fundamental = True, T_kin = atm_old.temp)

ch4.add_iso(3, MM = 17, ratio = 6.158e-4, LTE = True)

pickle.dump(ch4, open(inputs['cart_molecs']+'ch4_old_ref.pic','w'))

planet.add_gas(ch4)

### LOADING LINES
print('Loading lines...')

time0 = time.time()

db_file = inputs['hitran_db']
wn_range = [2800.,3500.]
linee = spcl.read_line_database(db_file, freq_range = wn_range)
linee = [lin for lin in linee if lin.Mol == 6]

if inputs['test']:
    print('Keeping ONLY 1000 lines for testing')
    linee = linee[:1000]

abs_coeff = smm.prepare_spe_grid(wn_range)
LUTS = smm.makeLUT_nonLTE_Gcoeffs(abs_coeff.spectral_grid, linee, planet.gases.values(), planet.atmosphere, pres_step_log = 0.2, cartLUTs = inputs['cart_LUTS'], n_threads = inputs['n_threads'], test = inputs['test'])

pickle.dump(LUTS, open(inputs['cart_LUTS']+'_allLUTS'+smm.date_stamp(),'w') )
print(time.ctime())
print('CIAO!')
