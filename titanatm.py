#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os.path
import matplotlib.pyplot as pl
import math as mt
import spect_base_module as sbm
import pickle
import scipy.io as io

cart = '/home/fede/Scrivania/Dotto/AbstrArt/CH4_HCN_climatology/Tit_atm/'

z = np.linspace(0,1500,151)

temp = sbm.read_input_prof_gbb(cart + 'in_temp.dat', 'temp')
pres = sbm.read_input_prof_gbb(cart + 'in_pres.dat', 'pres')
gases = sbm.read_input_prof_gbb(cart + 'in_vmr_prof.dat', 'vmr')

MM_manu = pickle.load(open(cart+'MM_manu.pic','r'))
# MM considerando solo N2 e CH4:
MM = (1-gases[5,:]*1e-6)*28+gases[5,:]*1e-6*16

# pl.plot(MM,z)
# pl.plot(MM_manu,z[:-1])
# pl.show()

P = sbm.hydro_P(z,temp,MM)

#pl.plot(P,z)
#pl.plot(pres,z)
#pl.xscale('log')
# fig = pl.figure(figsize=(8, 6), dpi=150)
# pl.plot(P/pres,z)
# pl.xlabel('P mia/P manuel')
# pl.ylabel('Altitude (km)')
# pl.grid()
# fig.savefig(cart+'P_diff.eps', format='eps', dpi=150)
# pl.show()
# pl.close()
#
# fig = pl.figure(figsize=(8, 6), dpi=150)
# pl.plot(MM,z)
# pl.xlabel('Mean molecular mass (amu)')
# pl.ylabel('Altitude (km)')
# pl.grid()
# fig.savefig(cart+'MM_Titan.eps', format='eps', dpi=150)
# pl.show()
# pl.close()

fac = 15*np.ones(101)
fac = np.append(np.zeros(50),fac)

fac2 = -20*(1-np.exp(-(z-450)/200))
fac2[0:45]=0

fac3 = -20*(1-np.exp(-(z-400)/200))
fac3[0:50]=0

wl2 = np.linspace(50,200,151)

wave1 = fac*np.sin(2*np.pi*(z-500)/150.0)
wave2 = fac2*np.sin(2*np.pi*(z-450.0)/150.0)
wave4 = fac3*np.sin(2*np.pi*(z-430)/wl2)

wave3 = wave1+wave4

fig = pl.figure(figsize=(8, 6), dpi=150)
pl.title('Wavy temp prof')
pl.plot(temp,z,label='Original')
pl.plot(temp+wave1,z,label='Wave 1')
pl.plot(temp+wave2,z,label='Wave 2')
#pl.plot(temp+wave3,z)
pl.plot(temp+wave3,z,label='Wave 3')
pl.legend(loc=4)
pl.grid()
pl.xlabel('Temp (K)')
pl.ylabel('Altitude (km)')
fig.savefig(cart+'Temp_waves.eps', format='eps', dpi=150)
pl.close()
#pl.show()

z2 = np.linspace(0,1500,301)
T = np.interp(z2,z,temp)
T1 = np.interp(z2,z,temp+wave1)
T2 = np.interp(z2,z,temp+wave2)
T3 = np.interp(z2,z,temp+wave3)
MMM = np.interp(z2,z,MM)
PP = sbm.hydro_P(z2,T,MMM)
P1 = sbm.hydro_P(z2,T1,MMM)
P2 = sbm.hydro_P(z2,T2,MMM)
P3 = sbm.hydro_P(z2,T3,MMM)

fig = pl.figure(figsize=(8, 6), dpi=150)
pl.title('Wavy temp prof')
#pl.plot(P,z,label='Original')
pl.plot(P1/PP,z2,label='Wave 1')
pl.plot(P2/PP,z2,label='Wave 2')
#pl.plot(temp+wave3,z)
pl.plot(P3/PP,z2,label='Wave 3')
pl.legend(loc=4)
pl.grid()
pl.xlabel('Pressure fluctuation (P/P orig)')
pl.ylabel('Altitude (km)')
fig.savefig(cart+'Pres_waves.eps', format='eps', dpi=150)
#pl.show()
pl.close()

sbm.write_input_atm_man(cart+'Atm_orig.dat',z2,T,PP)
sbm.write_input_atm_man(cart+'Atm_wave1.dat',z2,T1,P1)
sbm.write_input_atm_man(cart+'Atm_wave2.dat',z2,T2,P2)
sbm.write_input_atm_man(cart+'Atm_wave3.dat',z2,T3,P3)