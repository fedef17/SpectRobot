#!/usr/bin/python
# -*- coding: utf-8 -*-

import decimal
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
#import fparts

Q_hit = spcl.CalcPartitionSum(6, 1)
t = 296.0

#db_cart = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/Spect_data/MW_VIMS_CH4_bianca/'

db_file = '/home/fedefab/Scrivania/Research/Dotto/Spect_data/HITRAN/HITRAN08_2-5mu.par'

#n_mws, mw_tags, mw_ranges = spcl.read_mw_list(db_cart)

linee = []
#for i, tag, mw_rng in zip(range(n_mws), mw_tags, mw_ranges):
#    db_file = db_cart+'sp_'+tag+'.dat'
#    new_linee = spcl.read_line_database(db_file, verbose = True)
#    linee += new_linee
#    if i > 4: break

linee = spcl.read_line_database(db_file, freq_range = [2800.,3500.])

#line_wls = np.array([lin.freq for lin in linee])
#Q_hit = 5.9045e2
#linee_ch4 = [lin for lin in linee if lin.Mol == 6]
linee_ch4 = [lin for lin in linee if lin.Mol == 6]

essesss = [lin.Strength for lin in linee_ch4]

linea = linee_ch4[117]
print(linea.Q_num_lo, linea.Q_num_up)
Bi = spcl.Einstein_A_to_B(linea.A_coeff,linea.Freq)
esse = spcl.Einstein_A_to_LineStrength_nonLTE(linea.A_coeff,linea.Freq,linea.Energy_low,296.0,296.0,linea.g_lo,linea.g_up,Q_hit,  iso_ab = 0.98827)

esse3 = spcl.Einstein_A_to_LineStrength_hitran(linea.A_coeff,linea.Freq,296.0,Q_hit,linea.g_up,linea.Energy_low, iso_ab = 0.98827)


print(linea.Strength,esse,esse3)

sp_step = 5.e-4
min_wl = 2800.0
max_wl = 3500.0
imxsig = 13010

lin_grid = np.arange(-imxsig*sp_step/2,imxsig*sp_step/2,sp_step, dtype = np.float64)

# begin cycle on mw:
#sp_grid = np.arange(min_wl,max_wl+sp_step/2,sp_step)
spoffo = np.arange(min_wl,max_wl+sp_step/2,sp_step,dtype = np.float64)
spect_grid = spcl.SpectralGrid(spoffo, units = 'cm_1')
spoffo = np.zeros(len(spect_grid.grid), dtype = np.float64)
abs_coeff = spcl.SpectralObject(spoffo, spect_grid, units = 'cm2')

Temp = 296.0
Pres = 1.e-2
MM = 16

time0 = time.time()
time_100 = time.time()
for ii,lin in zip(range(len(linee_ch4)),linee_ch4):
    print('linea {} at {}'.format(ii,lin.Freq))
    ind_ok, fr_grid_ok = spcl.closest_grid(spect_grid,lin.Freq)
    lin_grid_ok = spcl.SpectralGrid(lin_grid+fr_grid_ok, units = 'cm_1')

    S = lin.CalcStrength_nonLTE(296.,296.,Q_hit)
    shape = lin.MakeShapeLine(lin_grid_ok, Temp, Pres, MM, Strength = S)

    abs_coeff.add_to_spectrum(shape)
    if ii % 100 == 0:
        print('Made 100 lines in {} s'.format(time.time()-time_100))
        time_100 = time.time()

print('Made {} lines in {} s'.format(len(linee_ch4,time.time()-time0)))

pl.ion()

pl.figure(1)
pl.plot(abs_coeff.spectral_grid.grid,abs_coeff.spectrum)


sys.exit()





spe = np.linspace(0.9,0.3,len(sp_grid))
spectrum = spcl.SpectralIntensity(spe, spect_grid, units = 'nWcm2')

pl.ion()

pl.figure(1)
pl.plot(spectrum.spectral_grid.grid,spectrum.intensity)

spectrum.convertto_Wm2()
spectrum.convertto_nm()

pl.figure(2)
pl.plot(spectrum.spectral_grid.grid,spectrum.intensity)

pl.show()

print(type(linee), len(linee), len(sp_grid))

sys.exit()

cart2 = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/Test_wave2/'

ch4_nom, ch4_wave2 = pickle.load(open(cart2+'ch4_Molec_testMaya.pic','r'))

cart_old = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/Old_vts/INP_TEST_OLDs/'

ch4_old = pickle.load(open(cart_old+'TestOld_ch4_Molec.pic','r'))

print(type(ch4_nom), type(ch4_nom.iso_1), type(ch4_nom.iso_1.lev_05), type(ch4_nom.iso_1.lev_05.vibtemp))

print(ch4_old, ch4_old.iso_1, ch4_old.iso_1.levels)#, type(ch4_old.iso_1.lev_05.vibtemp))

print(type(ch4_wave2), type(ch4_wave2.iso_1), type(ch4_wave2.iso_1.lev_05), type(ch4_wave2.iso_1.lev_05.vibtemp))


for cose in zip(ch4_nom.iso_1.lev_05.vibtemp.grid[0], ch4_nom.iso_1.lev_05.vibtemp.prof, ch4_wave2.iso_1.lev_05.vibtemp.prof, ch4_old.iso_1.lev_05.vibtemp.prof):
    print(cose)


altee = ch4_wave2.atmosphere.grid[0]
temp_old = ch4_old.atmosphere.interp_copy('temp',altee)
temp_nom = ch4_nom.atmosphere.interp_copy('temp',altee)
temp_wave2 = ch4_wave2.atmosphere.interp_copy('temp',altee)

pl.title('TEMP PROFILE')
pl.plot(temp_nom,altee,label='nom')
pl.plot(temp_wave2,altee,label='wave2',linestyle=':')
pl.plot(temp_old,altee,label='old',linestyle='--')
pl.legend()
pl.grid()
pl.show()


for lev in ch4_nom.iso_1.levels:
    print(lev)
    altee = getattr(ch4_old.iso_1,lev).vibtemp.grid[0]
    nom_pr = getattr(ch4_nom.iso_1,lev).vibtemp.interp_copy('prof',altee)
    wave2_pr = getattr(ch4_wave2.iso_1,lev).vibtemp.interp_copy('prof',altee)
    old_pr = getattr(ch4_old.iso_1,lev).vibtemp.interp_copy('prof',altee)
    nom_ratio = sbm.vibtemp_to_ratio(getattr(ch4_nom.iso_1,lev).energy, nom_pr, ch4_nom.atmosphere.interp_copy('temp',altee))
    wave2_ratio = sbm.vibtemp_to_ratio(getattr(ch4_wave2.iso_1,lev).energy, wave2_pr, ch4_wave2.atmosphere.interp_copy('temp',altee))
    old_ratio = sbm.vibtemp_to_ratio(getattr(ch4_old.iso_1,lev).energy, old_pr, ch4_old.atmosphere.interp_copy('temp',altee))

    # pl.plot(nom_pr,altee,label='nom')
    # pl.plot(wave2_pr,altee,label='wave2')
    # pl.plot(old_pr,altee,label='old')
    pl.title(lev+' -- VIBTEMP_diff')
    pl.plot(nom_pr-old_pr,altee,label='nom-old')
    pl.plot(nom_pr-wave2_pr,altee,label='nom-wave2')
    pl.legend()
    pl.grid()
    pl.show()

    pl.title(lev+' -- NLTE RATIO')
    aoo = altee > 200
    pl.plot(nom_ratio[aoo],altee[aoo],label='nom')
    #pl.plot(wave2_ratio[aoo],altee[aoo],label='wave2')
    pl.plot(old_ratio[aoo],altee[aoo],label='old')
    pl.legend()
    pl.grid()
    pl.show()

# sys.exit()
# print(ch4.iso_1.lev_11.vibtemp.calc(647))
#
# z = 500.
# for lin in linee:
#     lin._LinkToMolec_(ch4.iso_1)
#     T = lin.Up_lev.vibtemp.calc(z)
#     print(T,lin._CalcStrength_(T))
