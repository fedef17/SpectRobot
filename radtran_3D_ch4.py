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

#warnings.simplefilter("error")

### Program for CH4 HCN and C2H2 climatology on TITAN from VIMS spectra

time0 = time.time()

input_file = 'inputs_spect_robot_3D.in'

cart_atm = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/Tit_atm/climat_0607_manuel/p-T_clima2_07-07/'

cart_LUTS = '/media/hd_B/Spect_data/LUTs/'

cart_tvibs = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/clima_Maya_22_May_2017/'

cart_tvibs2 = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/clima2_vts_hcn/'

cart_tvibs3 = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/clima2_vts_c2h2/'

hit12_25 = '/home/fedefab/Scrivania/Research/Dotto/Spect_data/HITRAN/HITRAN2012_2-5mu.par'

keys = 'cart_atm cart_observed cart_tvibs cart_tvibs2 cart_tvibs3 cart_LUTS out_dir hitran_db n_threads test'
keys = keys.split()
itype = [str, str, str, str, str, str, str, str, int, bool]
defaults = [cart_atm, None, cart_tvibs, cart_tvibs2, cart_tvibs3, cart_LUTS, None, hit12_25, 8, False]
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

planet = sbm.Titan(1500.)

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

### LOADING MOLECULES
print('Loading molecules...')

n_alt_max = 151

atm_gases_old = sbm.read_input_prof_gbb(inputs['cart_atm'] + 'in_vmr_prof.dat', 'vmr', n_alt_max = n_alt_max)

zold = np.linspace(0.,10*(n_alt_max-1),n_alt_max)
gridvmr = sbm.AtmGrid(['lat', 'alt'], [lat_ext[:-1], zold])
for gas in atm_gases_old:
    coso2d = np.array((len(lat_ext)-1)*[atm_gases_old[gas]])
    atm_gases_old[gas] = sbm.AtmProfile(gridvmr, coso2d, profname='vmr', interp = ['box','lin'])

nlte_molecs = sbm.add_nLTE_molecs_from_tvibmanuel_3D(planet, inputs['cart_tvibs'], formato = 'Maya')

nlte_molecs2 = sbm.add_nLTE_molecs_from_tvibmanuel_3D(planet, inputs['cart_tvibs2'], formato = 'Manuel', correct_levstring = True)
for molec in nlte_molecs2.items():
    nlte_molecs[molec[0]] = copy.deepcopy(molec[1])

nlte_molecs3 = sbm.add_nLTE_molecs_from_tvibmanuel_3D(planet, inputs['cart_tvibs3'], formato = 'Manuel2')
for molec in nlte_molecs3.items():
    nlte_molecs[molec[0]] = copy.deepcopy(molec[1])

for molec in nlte_molecs.values():
    molec.link_to_atmos(Atm)
    try:
        molec.add_clim(atm_gases_old[molec.name])
        planet.add_gas(molec)
    except:
        for n in range(30): print(' ')
        print('ATTENZZZZIONEEE: gas {} not found in input vmr profiles'.format(molec.name))
        time.sleep(5)

##### SETTING THE BAYESSET:
baybau = smm.BayesSet(tag = 'test_CH4_HCN_C2H2_3D')
alt_nodes = np.arange(350., 1050., 100.)
lat_limits = lat_ext[:-1]

apriori_profs = []
cososo = atm_gases_old['CH4']
for lat in lat_limits:
    prf = []
    for alt in alt_nodes:
        prf.append(cososo.calc([lat,alt]))
    apriori_profs.append(prf)
apriori_profs = np.array(apriori_profs)
#apriori_profs = atm_gases_old['CH4'].vmr
apriori_prof_errs = 0.7*apriori_profs
set_ = smm.LinearProfile_2D('CH4', planet.atmosphere, alt_nodes, lat_limits, apriori_profs, apriori_prof_errs)
baybau.add_set(set_)


apriori_profs = []
cososo = atm_gases_old['HCN']
for lat in lat_limits:
    prf = []
    for alt in alt_nodes:
        prf.append(cososo.calc([lat,alt]))
    apriori_profs.append(prf)
apriori_profs = np.array(apriori_profs)
#apriori_profs = atm_gases_old['CH4'].vmr
apriori_prof_errs = 0.7*apriori_profs
set_ = smm.LinearProfile_2D('HCN', planet.atmosphere, alt_nodes, lat_limits, apriori_profs, apriori_prof_errs)
baybau.add_set(set_)


apriori_profs = []
cososo = atm_gases_old['C2H2']
for lat in lat_limits:
    prf = []
    for alt in alt_nodes:
        prf.append(cososo.calc([lat,alt]))
    apriori_profs.append(prf)
apriori_profs = np.array(apriori_profs)
#apriori_profs = atm_gases_old['CH4'].vmr
apriori_prof_errs = 0.7*apriori_profs
set_ = smm.LinearProfile_2D('C2H2', planet.atmosphere, alt_nodes, lat_limits, apriori_profs, apriori_prof_errs)
baybau.add_set(set_)


### updating the profile of gases in bayesset
for gas in baybau.sets.keys():
    planet.gases[gas].add_clim(baybau.sets[gas].profile())

### LOADING LINES

# max_lines = 10
# if len(linee) > max_lines:
#     print('Keeping ONLY 50 strongest lines for testing')
#     essesss = [lin.Strength for lin in linee]
#     essort = np.sort(np.array(essesss))[-1*max_lines]
#     linee_sel = [lin for lin in linee if lin.Strength >= essort]
#     linee = linee_sel
#
#     print(len(linee))
#     print(planet.gases)

wn_range = [2850.,3450.]
wn_range_obs = [spcl.convertto_nm(wn_range[1], 'cm_1')+10., spcl.convertto_nm(wn_range[0], 'cm_1')-10.]
print(wn_range_obs)

pixels = smm.read_input_observed(inputs['cart_observed'], wn_range = wn_range_obs)
pickle.dump(pixels, open('pixels_3D_test.pic','w'))

print(len(pixels))

dampa = open('./debuh_yeah.pic','wb')

radtran_opt = dict()
radtran_opt['max_T_variation'] = 5.
radtran_opt['max_Plog_variation'] = 1.

# prova ad alta quota
pix_ok = []
for pix in pixels:
    #print(pix.limb_tg_alt)
    if pix.limb_tg_alt < 1050. and pix.limb_tg_alt > 950.:
        pix_ok.append(pix)
        break

for pix in pixels:
    #print(pix.limb_tg_alt)
    if pix.limb_tg_alt < 750. and pix.limb_tg_alt > 650.:
        pix_ok.append(pix)
        break

for pix in pixels:
    #print(pix.limb_tg_alt)
    if pix.limb_tg_alt < 450. and pix.limb_tg_alt > 350.:
        pix_ok.append(pix)
        break

pix = pix_ok[0]
linea = pix.LOS()
ssp = pix.sub_solar_point()
linea.calc_SZA_along_los(planet, ssp)

prof = planet.atmosphere.get('pres')

time0 = time.time()
for i in range(20):
    prof.calc([72.,34.])
print('Tempo calc 2D: {}'.format((time.time()-time0)/20))

prof2 = planet.gases['CH4'].iso_1.lev_03.vibtemp

time0 = time.time()
for i in range(20):
   prof2.calc([72.,34.,451.])
print('Tempo calc 3D: {}'.format((time.time()-time0)/20))

pres = linea.calc_along_LOS(planet.atmosphere, profname = 'pres', set_attr = True)
temp = linea.calc_along_LOS(planet.atmosphere, profname = 'temp', set_attr = True)

vt = linea.calc_along_LOS(planet.gases['CH4'].iso_1.lev_03.vibtemp)




print('Loading lines...')
db_file = inputs['hitran_db']
linee = spcl.read_line_database(db_file, freq_range = wn_range)

# planet = pickle.load(open(inputs['cart_tvibs']+'planet.pic'))

planetmols = [gas.mol for gas in planet.gases.values()]

linee = [lin for lin in linee if lin.Freq >= wn_range[0] and lin.Freq <= wn_range[1]]


linee = smm.check_lines_mols(linee, planet.gases.values())
smm.keep_levels_wlines(planet, linee)

for gas in planet.gases:
    for iso in planet.gases[gas].all_iso:
        print([gas,iso], getattr(planet.gases[gas], iso).levels)

keep_levels = dict()
keep_levels[('CH4', 'iso_1')] = ['lev_00', 'lev_01', 'lev_02', 'lev_09', 'lev_07', 'lev_14', 'lev_08', 'lev_06', 'lev_03', 'lev_05', 'lev_04', 'lev_10']
# keep_levels[('CH4', 'iso_1')] = ['lev_00', 'lev_01', 'lev_09', 'lev_07', 'lev_08', 'lev_06', 'lev_03', 'lev_10']
keep_levels[('CH4', 'iso_2')] = ['lev_00', 'lev_02', 'lev_03']
keep_levels[('HCN', 'iso_1')] = ['lev_00', 'lev_01', 'lev_02', 'lev_04', 'lev_10', 'lev_07']
# keep_levels[('HCN', 'iso_1')] = ['lev_00', 'lev_04']
keep_levels[('C2H2', 'iso_1')] = ['lev_00', 'lev_01', 'lev_02']

smm.keep_levels(planet, keep_levels)

for gas in planet.gases:
    for iso in planet.gases[gas].all_iso:
        print([gas,iso], getattr(planet.gases[gas], iso).levels)

pickle.dump(planet, open(inputs['cart_tvibs']+'planet.pic','w'))
# track_levels = smm.track_all_levels(planet)

### test linea di vista verticale -> assorbimento solar intensity
# set_alts = np.arange(800., 99., -100.)
# set_intens = dict()
#
# Tsun = 5777.0
# spectral_grid = smm.prepare_spe_grid(wn_range).spectral_grid
# solar = spcl.Calc_BB(spectral_grid, Tsun)
# alt_init = 1500.0
# slat = pix_ok[0].sub_solar_lat
# slon = pix_ok[0].sub_solar_lon
# for alt in set_alts:
#     point2 = sbm.Coords([slat, slon, alt], s_ref = 'Spherical')
#     point1 = sbm.Coords([slat, slon, alt_init], s_ref = 'Spherical')
#     linea_vert = sbm.LineOfSight(point1, point2)
#     linea_vert.calc_atm_intersections(planet, delta_x = 5.0, start_from_TOA = False, stop_at_second_point = True, verbose = True, LOS_order = 'photon')
#     radtran = linea_vert.radtran(wn_range, planet, linee, cartLUTs = inputs['cart_LUTS'], cartDROP = inputs['out_dir'], radtran_opt = radtran_opt, solo_absorption = True, initial_intensity = solar, g3D = True, sub_solar_point = pix_ok[0].sub_solar_point())
#     set_intens[alt] = radtran
#     solar = radtran
#     alt_init = alt
#
# damparad = open('./solar_ray_absorb.pic','w')
# pickle.dump(set_intens, damparad)
# damparad.close()
# sys.exit()

LUTopt = dict()
LUTopt['max_pres'] = 2.0 # hPa circa 200 km
LUTopt['temp_step'] = 5.
LUTopt['pres_step_log'] = 1.0

PTcoup_needed = calc_PT_couples_atmosphere(lines, planet.gases.values(), planet.atmosphere, **LUTopt)

LUTS = check_and_build_allluts(inputs, sp_gri, lines, planet.gases.values(), PTcouples = PTcoup_needed, LUTopt = LUTopt)

for pix in pix_ok:
    damparad = open('./radtrans_ch4hcn_tot_{}km.pic'.format(int(pix.limb_tg_alt)),'w')
    linea = pix.LOS()
    radtran = linea.radtran(wn_range, planet, linee, cartLUTs = inputs['cart_LUTS'], cartDROP = inputs['out_dir'], calc_derivatives = False, LUTS = LUTS, useLUTs = True, radtran_opt = radtran_opt, g3D = True, sub_solar_point = pix.sub_solar_point())
    pickle.dump([pix, linea, radtran], damparad)
    damparad.close()


sys.exit()

result = smm.inversion(inputs, planet, linee, baybau, pixels, wn_range = wn_range, radtran_opt = radtran_opt, debugfile = dampa, useLUTs = True)

dampa.close()

tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))

print('Tempo una LOS: {} min'.format(tot_time/180.))
