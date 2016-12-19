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

z2,T,P = sbm.read_input_atm_man(cart+'pt_co_ref_09_05s.prf')

#MM_manu = pickle.load(open(cart+'MM_manu.pic','r'))
# MM considerando solo N2 e CH4:
MM = (1-gases[5,:]*1e-6)*28+gases[5,:]*1e-6*16
MM_manu = (0.98*28+0.02*16)*np.ones(len(temp))

#MM = MM_manu
MMM = np.interp(z2,z,MM)

PP = sbm.hydro_P(z2,T,MMM)

# fig = pl.figure(figsize=(8, 6), dpi=150)
# pl.plot((PP-P)/P*100,z2)
# pl.title('MM mio, R=2576, g=1.352')
# pl.xlabel('(P mia - P man)/P man %')
# pl.ylabel('Altitude (km)')
# pl.grid()
# fig.savefig(cart+'P_diff_5.eps', format='eps', dpi=150)
# pl.show()
# pl.close()
#
# sys.exit()

# fig = pl.figure(figsize=(8, 6), dpi=150)
# pl.plot(MM,z)
# pl.xlabel('Mean molecular mass (amu)')
# pl.ylabel('Altitude (km)')
# pl.grid()
# fig.savefig(cart+'MM_Titan.eps', format='eps', dpi=150)
# pl.show()
# pl.close()

z0 = 500
l0 = int(z0/5)
n = len(z2)

fac = 15*np.ones(n-l0)
fac = np.append(np.zeros(l0),fac)

fac2 = 20*(1-np.exp(-(z2-z0+50)/200))
fac2[0:l0]=0

fac3 = -20*(1-np.exp(-(z2-z0+100)/200))
fac3[0:l0]=0

wl1 = 150.0
wl2 = np.linspace(50,200,n)

wave1 = fac*np.sin(2*np.pi*(z2-z0)/wl1)
wave2 = fac2*np.sin(2*np.pi*(z2-z0)/wl1)
wave4 = fac3*np.sin(2*np.pi*(z2-z0)/wl2)

wave3 = wave1+wave4


T1 = T+wave1
T2 = T+wave2
T3 = T+wave3
P1 = sbm.hydro_P(z2,T1,MMM)
P2 = sbm.hydro_P(z2,T2,MMM)
P3 = sbm.hydro_P(z2,T3,MMM)

fig = pl.figure(figsize=(8, 6), dpi=150)
pl.title('Wavy temp prof')
pl.plot(T,z2,label='Original')
pl.plot(T1,z2,label='Wave 1')
pl.plot(T2,z2,label='Wave 2')
#pl.plot(temp+wave3,z)
pl.plot(T3,z2,label='Wave 3')
pl.legend(loc=4)
pl.grid()
pl.xlabel('Temp (K)')
pl.ylabel('Altitude (km)')
fig.savefig(cart+'Temp_waves.eps', format='eps', dpi=150)
pl.show()
pl.close()

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
pl.show()
pl.close()

sbm.write_input_atm_man(cart+'Atm_orig.dat',z2,T,PP)
sbm.write_input_atm_man(cart+'Atm_wave1.dat',z2,T1,P1)
sbm.write_input_atm_man(cart+'Atm_wave2.dat',z2,T2,P2)
sbm.write_input_atm_man(cart+'Atm_wave3.dat',z2,T3,P3)

sbm.write_input_prof_gbb(cart+'in_temp_orig.dat',T,'temp')
sbm.write_input_prof_gbb(cart+'in_temp_wave1.dat',T1,'temp')
sbm.write_input_prof_gbb(cart+'in_temp_wave2.dat',T2,'temp')
sbm.write_input_prof_gbb(cart+'in_temp_wave3.dat',T3,'temp')

sbm.write_input_prof_gbb(cart+'in_pres_orig.dat',PP,'pres')
sbm.write_input_prof_gbb(cart+'in_pres_wave1.dat',P1,'pres')
sbm.write_input_prof_gbb(cart+'in_pres_wave2.dat',P2,'pres')
sbm.write_input_prof_gbb(cart+'in_pres_wave3.dat',P3,'pres')


## ADESSO mi leggo le vibtemp di Maya
cmaya = '/home/fede/Scrivania/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/Wave2_maya/'
tvib1 = sbm.read_tvib_manuel(cmaya + 'vt_ch4_2006-07_09_05s_twave2_sza14_vmr3_v01.00_0061', mol = 'CH4')
tvib2 = sbm.read_tvib_manuel(cmaya + 'vt_ch4_2006-07_09_05s_sza14_vmr3_v01.00_0061', mol = 'CH4')
