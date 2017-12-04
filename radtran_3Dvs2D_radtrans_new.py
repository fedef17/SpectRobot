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
import shutil

time0 = time.time()
print(time.ctime())

input_file = 'inputs_spect_robot_3D.in'

cart_atm = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/Tit_atm/climat_0607_manuel/p-T_clima2_07-07/'

cart_input_1D = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/Tests/INPUT_TEST_7418/'

cart_LUTS = '/media/hd_B/Spect_data/LUTs/'

cart_tvibs = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/clima_Maya_22_May_2017/'

cart_tvibs2 = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/clima2_vts_hcn/'

cart_tvibs3 = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/clima2_vts_c2h2/'

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

# @profile
# def rdt3d(inputs):

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
grid = sbm.AtmGrid(['lat', 'alt'], [lat_c, z2])

TT = np.vstack(temps)
PP = np.vstack(press)

Atm = sbm.AtmProfile(grid, TT, 'temp', ['lin','lin'])
Atm.add_profile(PP, 'pres', ['lin','exp'])

planet.add_atmosphere(Atm)

### LOADING MOLECULES
print('Loading molecules...')

n_alt_max = 151

atm_gases_old = sbm.read_input_prof_gbb(inputs['cart_input_1D'] + 'in_vmr_prof.dat', 'vmr', n_alt_max = n_alt_max)

zold = np.linspace(0.,10*(n_alt_max-1),n_alt_max)
gridvmr = sbm.AtmGrid(['lat', 'alt'], [lat_c, zold])
for gas in atm_gases_old:
    coso2d = np.array((len(lat_ext)-1)*[atm_gases_old[gas]])
    atm_gases_old[gas] = sbm.AtmProfile(gridvmr, coso2d, profname='vmr', interp = ['lin','lin'])

nlte_molecs = sbm.add_nLTE_molecs_from_tvibmanuel_3D(planet, inputs['cart_tvibs'], formato = 'Maya', lat_interp = 'lin')

nlte_molecs2 = sbm.add_nLTE_molecs_from_tvibmanuel_3D(planet, inputs['cart_tvibs2'], formato = 'Manuel', correct_levstring = True, lat_interp = 'lin')
for molec in nlte_molecs2.items():
    nlte_molecs[molec[0]] = copy.deepcopy(molec[1])

nlte_molecs3 = sbm.add_nLTE_molecs_from_tvibmanuel_3D(planet, inputs['cart_tvibs3'], formato = 'Manuel2', lat_interp = 'lin')
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
alt_nodes = np.arange(450., 1051., 100.)
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
apriori_prof_errs = apriori_profs+0.015
set_ = smm.LinearProfile_2D('CH4', planet.atmosphere, alt_nodes, lat_limits, apriori_profs, apriori_prof_errs)
baybau.add_set(set_)

alt_nodes = np.arange(550., 1051., 100.)
apriori_profs = []
cososo = atm_gases_old['HCN']
for lat in lat_limits:
    prf = []
    for alt in alt_nodes:
        prf.append(cososo.calc([lat,alt]))
    apriori_profs.append(prf)
apriori_profs = np.array(apriori_profs)
#apriori_profs = atm_gases_old['CH4'].vmr
apriori_prof_errs = apriori_profs+3.e-4
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
apriori_prof_errs = apriori_profs+1.e-4
set_ = smm.LinearProfile_2D('C2H2', planet.atmosphere, alt_nodes, lat_limits, apriori_profs, apriori_prof_errs)
baybau.add_set(set_)


### updating the profile of gases in bayesset
for gas in baybau.sets.keys():
    planet.gases[gas].add_clim(baybau.sets[gas].profile())

planet3D = planet

#planet.gases['CH4'].iso_1.erase_level('lev_12')
############################################################

# keep_levels = dict()
# keep_levels[('CH4', 'iso_1')] = ['lev_00', 'lev_01', 'lev_02', 'lev_09', 'lev_07', 'lev_14', 'lev_08', 'lev_06', 'lev_03', 'lev_05', 'lev_04', 'lev_10']
# # keep_levels[('CH4', 'iso_1')] = ['lev_00', 'lev_01', 'lev_09', 'lev_07', 'lev_08', 'lev_06', 'lev_03', 'lev_10']
# keep_levels[('CH4', 'iso_2')] = ['lev_00', 'lev_02', 'lev_03']
# keep_levels[('HCN', 'iso_1')] = ['lev_00', 'lev_01', 'lev_02', 'lev_04', 'lev_10', 'lev_07']
# # keep_levels[('HCN', 'iso_1')] = ['lev_00', 'lev_04']
# keep_levels[('C2H2', 'iso_1')] = ['lev_00', 'lev_01', 'lev_02']
#
# smm.keep_levels(planet, keep_levels)

# for gas in planet.gases:
#     for iso in planet.gases[gas].all_iso:
#         print([gas,iso], getattr(planet.gases[gas], iso).levels)

######################

planet = sbm.Titan(1500.)
n_alt_max = 151

temp_old = sbm.read_input_prof_gbb(inputs['cart_input_1D'] + 'in_temp.dat', 'temp')
pres_old = sbm.read_input_prof_gbb(inputs['cart_input_1D'] + 'in_pres.dat', 'pres')

zold = np.linspace(0.,10*(n_alt_max-1),n_alt_max)
alt_gri = sbm.AtmGrid('alt', zold)

Atm = sbm.AtmProfile(alt_gri, temp_old, 'temp', 'lin')
Atm.add_profile(pres_old, 'pres', 'exp')

planet.add_atmosphere(Atm)

ciup = copy.deepcopy(planet3D.gases)
mol1 = sbm.read_tvib_gbb(inputs['cart_input_1D']+'in_vibtemp_HCN-C2H2.dat', Atm.get('temp'), ciup)
mol1 = sbm.read_tvib_gbb(inputs['cart_input_1D']+'in_vibtemp_CH4.dat', Atm.get('temp'), mol1)

atm_gases_old = sbm.read_input_prof_gbb(inputs['cart_input_1D'] + 'in_vmr_prof.dat', 'vmr', n_alt_max = n_alt_max)

for gas in atm_gases_old:
    atm_gases_old[gas] = sbm.AtmProfile(alt_gri, atm_gases_old[gas], profname='vmr', interp = 'lin')

for molec in mol1.values():
    molec.link_to_atmos(Atm)
    try:
        molec.add_clim(atm_gases_old[molec.name])
        planet.add_gas(molec)
    except:
        for n in range(30): print(' ')
        print('ATTENZZZZIONEEE: gas {} not found in input vmr profiles'.format(molec.name))
        time.sleep(5)

planet1D = planet

##### SETTING THE BAYESSET:
baybau1D = smm.BayesSet(tag = 'test_CH4_HCN_C2H2_1D')
alt_nodes = np.arange(450., 1051., 100.)

cososo = atm_gases_old['CH4']
prf = []
for alt in alt_nodes:
    prf.append(cososo.calc(alt))
apriori_prof = np.array(prf)
apriori_prof_err = apriori_prof+0.015
set_ = smm.LinearProfile_1D_new('CH4', alt_gri, alt_nodes, apriori_prof, apriori_prof_err)
baybau1D.add_set(set_)

alt_nodes = np.arange(550., 1051., 100.)

cososo = atm_gases_old['HCN']
prf = []
for alt in alt_nodes:
    prf.append(cososo.calc(alt))
apriori_prof = np.array(prf)
apriori_prof_err = apriori_prof+3.e-4
set_ = smm.LinearProfile_1D_new('HCN', alt_gri, alt_nodes, apriori_prof, apriori_prof_err)
baybau1D.add_set(set_)

cososo = atm_gases_old['C2H2']
prf = []
for alt in alt_nodes:
    prf.append(cososo.calc(alt))
apriori_prof = np.array(prf)
apriori_prof_err = apriori_prof+1.e-4
set_ = smm.LinearProfile_1D_new('C2H2', alt_gri, alt_nodes, apriori_prof, apriori_prof_err)
baybau1D.add_set(set_)

### updating the profile of gases in bayesset
for gas in baybau.sets.keys():
    planet1D.gases[gas].add_clim(baybau.sets[gas].profile())

###############################################################

wn_range = [2850.,3450.]
wn_range_obs = [spcl.convertto_nm(wn_range[1], 'cm_1')+10., spcl.convertto_nm(wn_range[0], 'cm_1')-10.]
print(wn_range_obs)

radtran_opt = dict()
radtran_opt['max_T_variation'] = 5.
radtran_opt['max_Plog_variation'] = 1.

print('Loading lines...')
db_file = inputs['hitran_db']
linee = spcl.read_line_database(db_file, freq_range = wn_range)
linee = smm.check_lines_mols(linee, planet3D.gases.values())

nuca = '/work/localuser/fedef/SPECT_ROBOT_RUN/CH4_newband/'
linee += spcl.read_line_database(nuca+'CH4_corrected_sel_0020.dat')
linee += spcl.read_line_database(nuca+'CH4_corrected_sel_0012.dat')
linee += spcl.read_line_database(nuca+'CH4_corrected_sel_0111.dat')

nuca2 = '/work/localuser/fedef/SPECT_ROBOT_RUN/HCN_newband/'
linee += spcl.read_line_database(nuca2+'HCN_new_hitcomplete.dat')

smm.keep_levels_wlines(planet3D, linee)
smm.keep_levels_wlines(planet1D, linee)
planet1D.gases['CH4'].iso_1.erase_level('lev_12')
planet3D.gases['CH4'].iso_1.erase_level('lev_12')


print('Loading lines...')
db_file = inputs['hitran_db']
linee = spcl.read_line_database(db_file, freq_range = wn_range)

# planet = pickle.load(open(inputs['cart_tvibs']+'planet.pic'))

linee = smm.check_lines_mols(linee, planet3D.gases.values())

nuca = '/work/localuser/fedef/SPECT_ROBOT_RUN/CH4_newband/'
linee += spcl.read_line_database(nuca+'CH4_corrected_sel_0020.dat')
linee += spcl.read_line_database(nuca+'CH4_corrected_sel_0012.dat')
linee += spcl.read_line_database(nuca+'CH4_corrected_sel_0111.dat')

nuca2 = '/work/localuser/fedef/SPECT_ROBOT_RUN/HCN_newband/'
linee += spcl.read_line_database(nuca2+'HCN_new_hitcomplete.dat')

smm.keep_levels_wlines(planet3D, linee)
smm.keep_levels_wlines(planet1D, linee)

pickle.dump(planet3D, open(inputs['cart_tvibs']+'planet_3D_latlin.pic','w'))
pickle.dump(planet1D, open(inputs['cart_tvibs']+'planet_1D.pic','w'))


LUTopt = dict()
LUTopt['max_pres'] = 0.1 # hPa circa 200 km
LUTopt['temp_step'] = 5.
LUTopt['pres_step_log'] = 1.0

sp_gri = smm.prepare_spe_grid(wn_range).spectral_grid

PTcoup_needed = smm.calc_PT_couples_atmosphere(linee, planet3D.gases.values(), planet3D.atmosphere, **LUTopt)

LUTS = smm.check_and_build_allluts(inputs, sp_gri, linee, planet.gases.values(), PTcouples = PTcoup_needed, LUTopt = LUTopt)

# sys.exit()

###################################################################
fil = open(inputs['cart_inputs']+'observ__7418_50NW_049_50.1n_66.2.pic','r')
pixels = pickle.load(fil)
fil.close()

i=0
for pix in pixels:
    print('Masking CH4 R branch')
    gri = pix.observation.spectral_grid.grid
    cond = (gri > 3190.) & (gri < 3295.)
    pix.observation.mask[cond] = 0
    pix.observation.noise = copy.deepcopy(pix.observation)
    pix.observation.noise.spectrum = 1.5e-8*np.ones(len(pix.observation.spectrum))
    if len(pix.observation.spectrum) != len(pix.observation.noise.spectrum) or len(pix.observation.spectrum) != len(pix.observation.mask):
        raise ValueError('Inconsistent length of mask or noise')
    pix.pixel_rot = 0.0
    i+=1

pixels.sort(key = lambda x: x.limb_tg_alt)

track_levels_all = dict()
for molnam in ['CH4', 'HCN', 'C2H2']:
    mol = planet3D.gases[molnam]
    for iso in mol.all_iso:
        isomol = getattr(mol, iso)
        track_levels_all[(molnam, iso)] = isomol.levels

track_levels_short = dict()
track_levels_short[('CH4', 'iso_1')] = ['lev_06', 'lev_09', 'lev_10']
track_levels_short[('HCN', 'iso_1')] = ['lev_13', 'lev_28']
track_levels_short[('C2H2', 'iso_1')] = ['lev_01', 'lev_02']

teag = 'sza65_tracklevels_szavar_all'
# radtrans_kwa = {'wn_range': wn_range, 'radtran_opt': radtran_opt, 'LUTopt': LUTopt, 'use_tangent_sza': False, 'nome_inv': teag, 'save_hires': True, 'group_observations': True, 'track_levels': track_levels_all}
# radtrans_arg = [inputs, planet3D, linee, pixels]
dampa = open(inputs['out_dir']+'./radtran_'+teag+'.pic','wb')
result = smm.radtrans(inputs, planet3D, linee, pixels, wn_range = wn_range, radtran_opt = radtran_opt, LUTopt = LUTopt, use_tangent_sza = False, nome_inv = teag, save_hires = True, group_observations = True, track_levels = track_levels_all)
pickle.dump(result, dampa)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))

teag = 'sza65_tracklevels_noszavar_short'
dampa = open(inputs['out_dir']+'./radtran_'+teag+'.pic','wb')
result = smm.radtrans(inputs, planet3D, linee, pixels, wn_range = wn_range, radtran_opt = radtran_opt, LUTopt = LUTopt, use_tangent_sza = True, nome_inv = teag, save_hires = True, group_observations = True, track_levels = track_levels_short)
pickle.dump(result, dampa)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))

teag = 'sza65_3D_tracklevels_inverseLOS_short'
dampa = open(inputs['out_dir']+'./radtran_'+teag+'.pic','wb')
result = smm.radtrans(inputs, planet3D, linee, pixels, wn_range = wn_range, radtran_opt = radtran_opt, LUTopt = LUTopt, use_tangent_sza = False, invert_LOS_direction = True, nome_inv = teag, save_hires = True, group_observations = True, track_levels = track_levels_short)
pickle.dump(result, dampa)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))


print('Faccio le inversions')

bay1 = copy.deepcopy(baybau1D)
time0 = time.time()
teag = '2Dvs3D_sza65_szavar_lin'
dampa = open(inputs['out_dir']+'./out_'+teag+'.pic','wb')
result = smm.inversion_fast_limb(inputs, planet3D, linee, bay1, pixels, wn_range = wn_range, radtran_opt = radtran_opt, debugfile = dampa, LUTopt = LUTopt, nome_inv = teag, group_observations = True)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))

for i in range(20):
    print('\n')

bay2 = copy.deepcopy(baybau1D)
time0 = time.time()
teag = '2Dvs3D_sza65_noszavar_lin'
dampa = open(inputs['out_dir']+'./out_'+teag+'.pic','wb')
result = smm.inversion_fast_limb(inputs, planet3D, linee, bay2, pixels, wn_range = wn_range, radtran_opt = radtran_opt, debugfile = dampa, LUTopt = LUTopt, use_tangent_sza = True, nome_inv = teag, group_observations = True)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))

for i in range(20):
    print('\n')

bay2 = copy.deepcopy(baybau1D)
time0 = time.time()
teag = '2Dvs3D_sza65_inverseLOS_lin'
dampa = open(inputs['out_dir']+'./out_'+teag+'.pic','wb')
result = smm.inversion_fast_limb(inputs, planet3D, linee, bay2, pixels, wn_range = wn_range, radtran_opt = radtran_opt, debugfile = dampa, LUTopt = LUTopt, use_tangent_sza = False, nome_inv = teag, invert_LOS_direction = True, group_observations = True)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))

for i in range(20):
    print('\n')

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

pixels.sort(key = lambda x: x.limb_tg_alt)
pix_rad = pixels[::3]

teag = 'sza80_tracklevels_szavar'
dampa = open(inputs['out_dir']+'./radtran_'+teag+'.pic','wb')
result = smm.radtrans(inputs, planet3D, linee, pix_rad, wn_range = wn_range, radtran_opt = radtran_opt, LUTopt = LUTopt, use_tangent_sza = False, nome_inv = teag, save_hires = True, group_observations = True, track_levels = track_levels_short)
pickle.dump(result, dampa)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))

teag = 'sza80_tracklevels_noszavar'
dampa = open(inputs['out_dir']+'./radtran_'+teag+'.pic','wb')
result = smm.radtrans(inputs, planet3D, linee, pix_rad, wn_range = wn_range, radtran_opt = radtran_opt, LUTopt = LUTopt, use_tangent_sza = True, nome_inv = teag, save_hires = True, group_observations = True, track_levels = track_levels_short)
pickle.dump(result, dampa)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))

teag = 'sza80_3D_tracklevels_inverseLOS'
dampa = open(inputs['out_dir']+'./radtran_'+teag+'.pic','wb')
result = smm.radtrans(inputs, planet3D, linee, pix_rad, wn_range = wn_range, radtran_opt = radtran_opt, LUTopt = LUTopt, use_tangent_sza = False, invert_LOS_direction = True, nome_inv = teag, save_hires = True, group_observations = True, track_levels = track_levels_short)
pickle.dump(result, dampa)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))

for i in range(20):
    print('\n')

bay1 = copy.deepcopy(baybau1D)
time0 = time.time()
teag = '2Dvs3D_sza80_szavar_lin'
dampa = open(inputs['out_dir']+'./out_'+teag+'.pic','wb')
result = smm.inversion_fast_limb(inputs, planet3D, linee, bay1, pixels, wn_range = wn_range, radtran_opt = radtran_opt, debugfile = dampa, LUTopt = LUTopt, nome_inv = teag, group_observations = True)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))

for i in range(20):
    print('\n')

bay2 = copy.deepcopy(baybau1D)
time0 = time.time()
teag = '2Dvs3D_sza80_noszavar_lin'
dampa = open(inputs['out_dir']+'./out_'+teag+'.pic','wb')
result = smm.inversion_fast_limb(inputs, planet3D, linee, bay2, pixels, wn_range = wn_range, radtran_opt = radtran_opt, debugfile = dampa, LUTopt = LUTopt, use_tangent_sza = True, nome_inv = teag, group_observations = True)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))

for i in range(20):
    print('\n')

bay2 = copy.deepcopy(baybau1D)
time0 = time.time()
teag = '2Dvs3D_sza80_inverseLOS_lin'
dampa = open(inputs['out_dir']+'./out_'+teag+'.pic','wb')
result = smm.inversion_fast_limb(inputs, planet3D, linee, bay2, pixels, wn_range = wn_range, radtran_opt = radtran_opt, debugfile = dampa, LUTopt = LUTopt, use_tangent_sza = False, nome_inv = teag, invert_LOS_direction = True, group_observations = True)
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


pixels.sort(key = lambda x: x.limb_tg_alt)
pix_rad = pixels[::3]

teag = 'sza30_tracklevels_szavar'
dampa = open(inputs['out_dir']+'./radtran_'+teag+'.pic','wb')
result = smm.radtrans(inputs, planet3D, linee, pix_rad, wn_range = wn_range, radtran_opt = radtran_opt, LUTopt = LUTopt, use_tangent_sza = False, nome_inv = teag, save_hires = True, group_observations = True, track_levels = track_levels_short)
pickle.dump(result, dampa)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))

teag = 'sza30_tracklevels_noszavar'
dampa = open(inputs['out_dir']+'./radtran_'+teag+'.pic','wb')
result = smm.radtrans(inputs, planet3D, linee, pix_rad, wn_range = wn_range, radtran_opt = radtran_opt, LUTopt = LUTopt, use_tangent_sza = True, nome_inv = teag, save_hires = True, group_observations = True, track_levels = track_levels_short)
pickle.dump(result, dampa)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))

teag = 'sza30_3D_tracklevels_inverseLOS'
dampa = open(inputs['out_dir']+'./radtran_'+teag+'.pic','wb')
result = smm.radtrans(inputs, planet3D, linee, pix_rad, wn_range = wn_range, radtran_opt = radtran_opt, LUTopt = LUTopt, use_tangent_sza = False, invert_LOS_direction = True, nome_inv = teag, save_hires = True, group_observations = True, track_levels = track_levels_short)
pickle.dump(result, dampa)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))


for i in range(20):
    print('\n')

bay1 = copy.deepcopy(baybau1D)
time0 = time.time()
teag = '2Dvs3D_sza30_szavar_lin'
dampa = open(inputs['out_dir']+'./out_'+teag+'.pic','wb')
result = smm.inversion_fast_limb(inputs, planet3D, linee, bay1, pixels, wn_range = wn_range, radtran_opt = radtran_opt, debugfile = dampa, LUTopt = LUTopt, nome_inv = teag, group_observations = True)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))


for i in range(20):
    print('\n')

bay2 = copy.deepcopy(baybau1D)
time0 = time.time()
teag = '2Dvs3D_sza30_noszavar_lin'
dampa = open(inputs['out_dir']+'./out_'+teag+'.pic','wb')
result = smm.inversion_fast_limb(inputs, planet3D, linee, bay2, pixels, wn_range = wn_range, radtran_opt = radtran_opt, debugfile = dampa, LUTopt = LUTopt, use_tangent_sza = True, nome_inv = teag, group_observations = True)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))


for i in range(20):
    print('\n')

bay2 = copy.deepcopy(baybau1D)
time0 = time.time()
teag = '2Dvs3D_sza30_inverseLOS_lin'
dampa = open(inputs['out_dir']+'./out_'+teag+'.pic','wb')
result = smm.inversion_fast_limb(inputs, planet3D, linee, bay2, pixels, wn_range = wn_range, radtran_opt = radtran_opt, debugfile = dampa, LUTopt = LUTopt, use_tangent_sza = False, nome_inv = teag, invert_LOS_direction = True, group_observations = True)
dampa.close()
tot_time = time.time()-time0
print('Tempo totale: {} min'.format(tot_time/60.))

print(time.ctime())
