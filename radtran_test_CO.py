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

with warnings.catch_warnings(record=True) as w:
    # Cause all warnings to always be triggered.
    warnings.simplefilter("error")

### Program for CH4 HCN and C2H2 climatology on TITAN from VIMS spectra

time0 = time.time()

input_file = 'inputs_spect_robot.in'

cart_test = '/home/fedefab/Scrivania/Research/Dotto/Reports/Code_validation/INP_FILES_29_25s_sza30/'
#cart_test = '/home/fedefab/Scrivania/Research/Dotto/Reports/Code_validation/INP_FILES_29_25s_sza30/INP_HOMO/'


cart_LUTS = '/media/hd_B/Spect_data/LUTs/'

cart = '/media/hd_B/Spect_data/Stuff/'
if not os.path.exists(cart):
    raise MemoryError('Disk not mounted or wrong path: '+cart)

out_dir = cart+'stuff_'+smm.date_stamp()
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
out_dir += '/'

hit08_25 = '/home/fedefab/Scrivania/Research/Dotto/Spect_data/HITRAN/HITRAN08_2-5mu.par'

hit12_25 = '/home/fedefab/Scrivania/Research/Dotto/Spect_data/HITRAN/HITRAN2012_2-5mu.par'

keys = 'cart_atm cart_molecs cart_LUTS out_dir hitran_db n_threads test'
keys = keys.split()
itype = [str, str, str, str, str, int, bool]
defaults = [cart_test, cart_test, cart_LUTS, out_dir, hit12_25, 8, False]
inputs = sbm.read_inputs(input_file, keys, itype = itype, defaults = defaults)

if not os.path.exists(inputs['cart_LUTS']):
    raise MemoryError('Disk not mounted or wrong path: '+cart)

if not os.path.exists(inputs['out_dir']):
    raise MemoryError('Disk not mounted or wrong path: '+cart)

sbm.check_free_space(inputs['cart_LUTS'])
sbm.check_free_space(inputs['out_dir'])

### LOADING LINES
print('Loading lines...')

time0 = time.time()

db_file = inputs['hitran_db']
wn_range = [2050.,2250.]

linee = spcl.read_line_database(db_file)
#linee = spcl.read_line_database(db_file, freq_range = wn_range)

### LOADING PLANET
print('Loading planet...')

planet = sbm.Titan()

### LOADING MOLECULES
print('Loading molecules...')

n_alt_max = 101

temp_old = sbm.read_input_prof_gbb(inputs['cart_molecs'] + 'in_temp_co_ref_07_25s.dat', 'temp', n_alt_max = n_alt_max)
pres_old = sbm.read_input_prof_gbb(inputs['cart_molecs'] + 'in_pres_co_ref_07_25s.dat', 'pres', n_alt_max = n_alt_max)

zold = np.linspace(0.,10*(n_alt_max-1),n_alt_max)
alt_gri = sbm.AtmGrid('alt', zold)

atm_old = sbm.AtmProfile(alt_gri, temp_old, 'temp', 'lin')
atm_old.add_profile(pres_old, 'pres', interp = 'exp')
planet.add_atmosphere(atm_old)

filetvi = inputs['cart_molecs']+'vt_co__07_25s_sza30_vmr04_7.62.1_0050'
#filetvi = inputs['cart_molecs']+'vt_HOMO.dat'
nlte_molecs = sbm.add_nLTE_molecs_from_tvibmanuel(planet, filetvi, linee = linee)#, extend_to_alt = 1500.)

atm_gases_old = sbm.read_input_prof_gbb(inputs['cart_molecs'] + 'in_vmr_prof.dat', 'vmr', n_alt_max = n_alt_max)

for gas in atm_gases_old:
    atm_gases_old[gas] = sbm.AtmProfile(alt_gri, atm_gases_old[gas], profname='vmr', interp = 'lin')

# hcn = sbm.Molec(23, 'HCN')
# hcn.add_all_iso_from_HITRAN(linee)
# hcn.link_to_atmos(atm_old)
# hcn.add_clim(atm_gases_old['HCN'])
# planet.add_gas(hcn)
#
# c2h2 = sbm.Molec(26, 'C2H2')
# c2h2.add_all_iso_from_HITRAN(linee)
# c2h2.link_to_atmos(atm_old)
# c2h2.add_clim(atm_gases_old['C2H2'])
# planet.add_gas(c2h2)

print(' ')
print(atm_gases_old.keys())
print(' ')

for molec in nlte_molecs.values():
    molec.link_to_atmos(atm_old)
    try:
        molec.add_clim(atm_gases_old[molec.name])
        planet.add_gas(molec)
    except:
        for n in range(30): print(' ')
        print('ATTENZZZZIONEEE: gas {} not found in input vmr profiles'.format(molec.name))
        time.sleep(5)

#pickle.dump(planet, open(inputs['cart_molecs']+'co_old_ref_nonLTE.pic','w'))
# planet = pickle.load(open(inputs['cart_molecs']+'co_old_ref_nonLTE.pic'))

planetmols = [gas.mol for gas in planet.gases.values()]

linee = [lin for lin in linee if lin.Freq >= wn_range[0] and lin.Freq <= wn_range[1] and lin.Mol in planetmols]

# max_lines = 100
# if inputs['test'] and len(linee) > max_lines:
#     print('Keeping ONLY 50 strongest lines for testing')
#     essesss = [lin.Strength for lin in linee]
#     essort = np.sort(np.array(essesss))[-1*max_lines]
#     linee_sel = [lin for lin in linee if lin.Strength >= essort]
#     linee = linee_sel


#linee = [lin for lin in linee if lin.Iso <= 3 and lin.Mol == 5]

for lin in linee:
    lin.Print()

print(len(linee))
print(planet.gases)

####################### BUILDING ABS_COEFFFFF

pixels = smm.read_input_observed(cart_test)
loss = []
pl.ion()
for pix in pixels[:1]:
    linea1 = pix.LOS(verbose = True)
    loss.append(linea1)

    # linea1.calc_atm_intersections(planet)
    # pl.plot([p.Spherical()[2] for p in linea1.intersections])

    linea1.calc_radtran_steps(planet, linee, max_Plog_variation = 2.0, max_opt_depth = 10.0, max_T_variation = 5.0)

    print('Tangent ALT! :' ,linea1.tangent_altitude)
    time.sleep(3)

    print('Ci sono {} steps'.format(len(linea1.radtran_steps['step'])))
    time.sleep(7)

    #pl.ion()
    radtran = linea1.radtran(wn_range, planet, linee, cartLUTs = inputs['cart_LUTS'], cartDROP = inputs['out_dir'])
    #pl.legend()

    pickle.dump(radtran, open('./radtran_CO_test_LOSVERO{:04d}_intero_trueatm_newCurGod.pic'.format(int(pix.limb_tg_alt)),'w'))

    # intens = radtran[0]
    # pl.figure(42)
    # intens.plot()
    # pl.grid()

tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))

print('Tempo una LOS: {} min'.format(tot_time/180.))

    # new_grid = smm.prepare_spe_grid(wn_range, sp_step = 25.e-4)
    # spet = intens.convolve_to_grid(new_grid.spectral_grid)
    #
    # pickle.dump(spet, open('./radtran_CO_test_LOSVERO{:04d}_lowres.pic'.format(int(pix.limb_tg_alt)),'w'))
    #
    # pl.figure(892)
    # spet.plot()
    # pl.grid()
