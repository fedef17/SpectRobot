#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os.path
import matplotlib.pyplot as pl
import math as mt
import spect_base_module as sbm
import pickle
import dill
import scipy.io as io

cart = '/home/fede/Scrivania/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/Old_vts/INP_TEST_OLDs/'
z = np.linspace(0,1500,151)
temp = sbm.read_input_prof_gbb(cart + 'in_temp.dat', 'temp')
pres = sbm.read_input_prof_gbb(cart + 'in_pres.dat', 'pres')

Atm_nom = sbm.AtmProfile(temp,z,gridname=['Alt (km)'],profname='temp')
Atm_nom.add_profile(pres,'pres')
temp = sbm.AtmProfile(temp,z,gridname=['Alt (km)'])
pres = sbm.AtmProfile(pres,z,gridname=['Alt (km)'],profname='pres')


## ADESSO mi leggo le vibtemp

file1 = 'vt_ch4__029_2006_t15_10.2s_29.9_vmrA2_v10_0061'
nome = 'in_vibtemp_old_CHECK.dat'

alts_vib, molecs, levels, energies, vibtemps = sbm.read_tvib_manuel(cart + file1)

for lev in levels:
    print('{:15.15s}{:1s}'.format(lev,'/'))

cmaya = '/home/fede/Scrivania/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/Wave2_maya/'
levstrip = [lev.strip() for lev in levels]
with open(cmaya+'/../CH4_levels2.dat','r') as infi:
    ch4_levs = [line.rstrip() for line in infi]

simmetries = []
for levi in levstrip:
    print(levi)
    simm = []
    for leve in ch4_levs:
        if levi in leve[:-3]:
            print('--->',leve)
            simm.append(leve)
    simmetries.append(simm)

#print(molecs,levels,energies,vibtemps)
vib_ok = []
for tempu in vibtemps:
    #print(np.shape(np.array(tempu)),np.shape(np.array(alts_vib)))
    tempu_ok = sbm.AtmProfile(np.array(tempu),np.array(alts_vib))
    vib_ok.append(tempu_ok)

ch4 = sbm.Molec(6, 'CH4', MM=12)
ch4.add_iso(1, MM = 12, ratio = 0.999)
# Aggiungo lo stato fondamentale
lev_0 = '     0 0 0 0   '
energies.insert(0,0.0)
levels.insert(0, lev_0)
simmetries.insert(0,[])
vib_ok.insert(0, temp)
ch4.iso_1.add_levels(levels,energies,vibtemps=vib_ok,simmetries=simmetries)

pickle.dump(ch4, open(cart+'TestOld_ch4_Molec.pic','w'))


sbm.write_tvib_gbb(cart+nome+'.vibtemp', ch4, Atm_nom, descr='Old vib temp test atmosphere',l_ratio=False,script=__file__)
sbm.write_tvib_gbb(cart+nome, ch4, Atm_nom, descr='Old vib temp test atmosphere',script=__file__)
