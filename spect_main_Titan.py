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

hit12_25 = '/home/fedefab/Scrivania/Research/Dotto/Spect_data/HITRAN/HITRAN2012_2-5mu.par'

keys = 'cart_atm cart_molecs cart_LUTS hitran_db n_threads test'
keys = keys.split()
itype = [str, str, str, str, int, bool]
defaults = [cartatm, cart_old_ch4, cart_LUTS, hit12_25, 8, False]
inputs = sbm.read_inputs(input_file, keys, itype = itype, defaults = defaults)

### LOADING LINES
print('Loading lines...')

time0 = time.time()

db_file = inputs['hitran_db']
wn_range = [2950.,2980.]

#linee = spcl.read_line_database(db_file)
linee = spcl.read_line_database(db_file, freq_range = wn_range)

inputs['test'] = False

### LOADING PLANET
print('Loading planet...')

planet = sbm.Titan()

"""
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

"""

### LOADING MOLECULES
print('Loading molecules...')

n_alt_max = 121

temp_old = sbm.read_input_prof_gbb(inputs['cart_molecs'] + 'in_temp.dat', 'temp', n_alt_max = n_alt_max)
pres_old = sbm.read_input_prof_gbb(inputs['cart_molecs'] + 'in_pres.dat', 'pres', n_alt_max = n_alt_max)
zold = np.linspace(0.,10*(n_alt_max-1),n_alt_max)

ch4 = sbm.Molec(6, 'CH4')

ch4.add_iso(1, LTE = False)
alts_vib, molecs, levels, energies, vib_ok = sbm.read_tvib_manuel(inputs['cart_molecs']+'vt_ch4__029_2006_t15_10.2s_29.9_vmrA2_v10_0061', n_alt_max = 2*n_alt_max-1)

atm_old = sbm.AtmProfile(np.interp(alts_vib,zold,temp_old),np.array(alts_vib),profname='temp')
atm_old.add_profile(np.exp(np.interp(alts_vib,zold,np.log(pres_old))), 'pres', interp = 'exp')

planet.add_atmosphere(atm_old)

atm_gases_old = sbm.read_input_prof_gbb(inputs['cart_molecs'] + 'in_vmr_prof.dat', 'vmr', n_alt_max = n_alt_max)

for gas in atm_gases_old:
    atm_gases_old[gas] = sbm.AtmProfile(np.interp(alts_vib,zold,atm_gases_old[gas]),np.array(alts_vib),profname='vmr')
print(atm_gases_old.viewitems())

ch4.link_to_atmos(atm_old)
ch4.add_clim(atm_gases_old['CH4'])

print('qui')

ch4.iso_1.add_levels(levels, energies, vibtemps=vib_ok, add_fundamental = True, T_kin = atm_old.temp)
ch4.iso_1.add_simmetries_levels(linee)


ch4.add_iso(2, LTE = False)
alts_vib, molecs, levels, energies, vib_ok = sbm.read_tvib_manuel(inputs['cart_molecs']+'vt_ch4__029_2006_t15_10.2s_29.9_vmrA2_v10_0062')
ch4.iso_2.add_levels(levels, energies, vibtemps=vib_ok, add_fundamental = True, T_kin = atm_old.temp)
ch4.iso_2.add_simmetries_levels(linee)

ch4.add_all_iso_from_HITRAN(linee, n_max = 3)

print(levels, energies)

#ch4.add_iso(3, MM = 17, ratio = 6.158e-4, LTE = True)
# ch4 = None
# ch4 = sbm.Molec(6, 'CH4')

# ch4.link_to_atmos(atm_old)
# ch4.add_clim(atm_gases_old['CH4'])

# hcn = sbm.Molec(23, 'HCN')
# hcn.add_all_iso_from_HITRAN(linee)
# hcn.link_to_atmos(atm_old)
# hcn.add_clim(atm_gases_old['HCN'])
#
# c2h2 = sbm.Molec(26, 'C2H2')
# c2h2.add_all_iso_from_HITRAN(linee)
# c2h2.link_to_atmos(atm_old)
# c2h2.add_clim(atm_gases_old['C2H2'])
#
# planet.add_gas(hcn)
# planet.add_gas(c2h2)
planet.add_gas(ch4)

# pickle.dump(planet, open(inputs['cart_molecs']+'ch4_old_ref.pic','w'))

#linee = [lin for lin in linee if lin.Freq >= wn_range[0] and lin.Freq <= wn_range[1]]

if inputs['test']:
    print('Keeping ONLY 1000 linee for testing')
    linee = linee[:1000]

#planet = pickle.load(open(inputs['cart_molecs']+'ch4_old_ref.pic','r'))

#sys.exit()
##########################################################
##########################################################
###########################################################

# abs_coeff = smm.prepare_spe_grid(wn_range)
# LUTS = smm.makeLUT_nonLTE_Gcoeffs(abs_coeff.spectral_grid, linee, planet.gases.values(), planet.atmosphere, pres_step_log = 0.2, cartLUTs = inputs['cart_LUTS'], n_threads = inputs['n_threads'], test = inputs['test'])
#
# pickle.dump(LUTS, open(inputs['cart_LUTS']+'_allLUTS'+smm.date_stamp(),'w') )
# print(time.ctime())
# print('CIAO!')



####################### BUILDING ABS_COEFFFFF

# IN LTE: check che il abs_coeff in LTE fatto dai Gcoeff venga uguale a quello fatto dalla line strength

# IN NON-LTE: check tra il nostro e quello del gbb a 600 km
t600 = ch4.atmosphere.calc(600., profname = 'temp')
p600 = ch4.atmosphere.calc(600., profname = 'pres')
print('PRESSURE: {}, TEMPERATURE: {}'.format(p600,t600))

for iso in ch4.all_iso:
    isomol = getattr(ch4, iso)
    print('Calculating mol {}, iso {}. Mol in LTE? {}'.format(isomol.mol,isomol.iso,isomol.is_in_LTE))
    for lev in isomol.levels:
        print(lev)
        levello = getattr(isomol, lev)
        print(levello.lev_string)
        print(levello.simmetry)
        if not isomol.is_in_LTE:
            #print(levello.vibtemp)
            tvi = levello.vibtemp.calc(600.)
            levello.add_local_vibtemp(tvi)
            print(levello.local_vibtemp)


pickle.dump([t600, p600, ch4], open('./local_vibtemp_ch4.pic','w'))


Temps = [175.0, 175.0, 175.0, 175.0, 175.0]
Press = [0.001, 10.0, 1.0, 0.1, 0.01]

airbr = np.argmax(np.array([lin.Air_broad for lin in linee]))

for T, P in zip(Temps,Press):
    print(T, P)
    dw, lw, wsh = linee[airbr].CheckWidths(T, P, ch4.iso_1.MM)
    print(dw, lw, wsh, lw/dw, wsh/dw)

    abs_coeff_tot = smm.prepare_spe_grid(wn_range)
    emi_coeff_tot = smm.prepare_spe_grid(wn_range)
    abs_coeffs_iso = dict()

    for iso in ch4.all_iso:
        isomol = getattr(ch4, iso)
        abs_coeffs, emi_coeffs = smm.make_abscoeff_isomolec(wn_range, isomol, T, P, lines = linee, LTE = isomol.is_in_LTE, cartLUTs = inputs['cart_LUTS'])
        abs_coeffs_iso[iso] = abs_coeffs[0]
        iso_ab = isomol.ratio
        for ab, em in zip(abs_coeffs,emi_coeffs):
            abs_coeff_tot.add_to_spectrum(ab, Strength = iso_ab)
            emi_coeff_tot.add_to_spectrum(em, Strength = iso_ab)

    piu = int(T)
    pou = int(100.*P)
    print(T, P)
    print(int(T), int(100.*P))
    print(piu,pou)
    nom = './abscoeff_ch4_T{:03d}K_P{:04d}Pa.pic'.format(piu, pou)
    print(nom)
    pickle.dump([abs_coeff_tot, emi_coeff_tot, abs_coeffs_iso], open(nom,'w'))

sys.exit()

abs_coeff_tot = smm.prepare_spe_grid(wn_range)
emi_coeff_tot = smm.prepare_spe_grid(wn_range)
hcn = planet.gases['HCN']
for iso in hcn.all_iso:
    isomol = getattr(hcn, iso)
    abs_coeffs, emi_coeffs = smm.make_abscoeff_isomolec(wn_range, isomol, t600, p600, lines = linee, LTE = isomol.is_in_LTE, cartLUTs = inputs['cart_LUTS'])
    iso_ab = isomol.ratio
    for ab, em in zip(abs_coeffs,emi_coeffs):
        abs_coeff_tot.add_to_spectrum(ab, Strength = iso_ab)
        emi_coeff_tot.add_to_spectrum(em, Strength = iso_ab)

pickle.dump([abs_coeff_tot, emi_coeff_tot], open('./validation_abscoeff_hcnLTE_600km_3.pic','w'))

abs_coeff_tot = smm.prepare_spe_grid(wn_range)
emi_coeff_tot = smm.prepare_spe_grid(wn_range)
c2h2 = planet.gases['C2H2']
for iso in c2h2.all_iso:
    isomol = getattr(c2h2, iso)
    for levi in isomol.levels:
        print(levi)
        levello = getattr(isomol, levi)
        print(levello.energy, levello.lev_string)
    abs_coeffs, emi_coeffs = smm.make_abscoeff_isomolec(wn_range, isomol, t600, p600, lines = linee, LTE = isomol.is_in_LTE, cartLUTs = inputs['cart_LUTS'])
    iso_ab = isomol.ratio
    for ab, em in zip(abs_coeffs,emi_coeffs):
        abs_coeff_tot.add_to_spectrum(ab, Strength = iso_ab)
        emi_coeff_tot.add_to_spectrum(em, Strength = iso_ab)

pickle.dump([abs_coeff_tot, emi_coeff_tot], open('./validation_abscoeff_c2h2LTE_600km_3.pic','w'))

#planet = pickle.load(open(inputs['cart_molecs']+'ch4_old_ref.pic','r'))

spacecraft = sbm.Coords([1.e5,600.,0],s_ref='Cartesian')
second = sbm.Coords([90,0,600],s_ref='Spherical')

linea1 = sbm.LineOfSight(spacecraft, second)
linea1.details()

# Winter Solstice NORTH
#ssp = sbm.Coords(np.array([-26.,90,0]),s_ref='Spherical')

steplos = 120.0 # km!
point1 = linea1.calc_atm_intersections(planet, delta_x = steplos)
print(len(point1))
#psza1 = linea1.calc_SZA_along_los(planet,ssp)

pt1 = linea1.calc_along_LOS(planet.atmosphere, profname = 'temp', set_attr = True)
print(pt1)
pp1 = linea1.calc_along_LOS(planet.atmosphere, profname = 'pres', set_attr = True)
print(pp1)

for gas in planet.gases:
    for iso in planet.gases[gas].all_iso:
        isomol = getattr(planet.gases[gas], iso)
        print('Gas {}, iso {}, lev {}'.format(isomol.mol, isomol.iso, isomol.levels))
    conc_gas = linea1.calc_abundance(planet, gas, set_attr = True)

opt_depth = linea1.calc_optical_depth(wn_range, planet, linee, step = steplos, cartLUTs = inputs['cart_LUTS'])
pickle.dump(opt_depth, open('./opt_dep_600_CONFRONTO.pic','w'))

# pl.ion()
# pl.figure(17)
# pl.plot(abs_coeff_tot.spectral_grid.grid, abs_coeff_tot.spectrum)
# pl.figure(18)
# pl.plot(emi_coeff_tot.spectral_grid.grid, emi_coeff_tot.spectrum)
# pl.show()
