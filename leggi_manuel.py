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

cart = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/Old_vts/INP_TEST_OLDs/'
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

cmaya = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/Wave2_maya/'

levstrip = [lev.strip() for lev in levels]
lev_quanta = [sbm.extract_quanta_ch4(lev)[0] for lev in levels]
#with open(cmaya+'/../CH4_levels2.dat','r') as infi:
#    ch4_levs = [line.rstrip() for line in infi]

livelli = sbm.read_mol_levels_HITRAN(molec = 'CH4')
simmetries = []

simmetries = []
for levi in lev_quanta:
    print(levi)
    simm = []
    for leve,levsim in zip(livelli['quanta'],livelli['lev_strings']):
        if leve == levi:
            print('--->',leve)
            simm += levsim
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
quanta = sbm.extract_quanta_ch4(lev_0)[0]
for leve,levsim in zip(livelli['quanta'],livelli['lev_strings']):
    if leve == quanta:
        print('--->',leve)
        simmetries.insert(0,levsim)
vib_ok.insert(0, temp)
ch4.iso_1.add_levels(levels,energies,vibtemps=vib_ok,simmetries=simmetries)

ch4.link_to_atmos(Atm_nom)

pickle.dump(ch4, open(cart+'TestOld_ch4_Molec.pic','w'))


sbm.write_tvib_gbb(cart+nome+'.vibtemp', ch4, Atm_nom, descr='Old vib temp test atmosphere',l_ratio=False,script=__file__)
sbm.write_tvib_gbb(cart+nome, ch4, Atm_nom, descr='Old vib temp test atmosphere',script=__file__)

###################################################################################
###################################################################################
###################################################################################
###################################################################################

# LA PARTE QUI SOTTO é per FARE un CONFRONTO TRA TEMPERATURE VECCHIE ORIGInaLI E VECCHIE RILETTE E RISCRITTE DA leggi_manuel.py. SEMBRA CHE VENGA TUTTO PERFETTO, differenze entro 0.1%

###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################

print('ORA LEGGGOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')

ch4_readss = sbm.read_tvib_gbb(cart+nome, Atm_nom)
ch4_read = ch4_readss[0]

energies = []
levelsss = []
lev_quanta = []
for lev in ch4_read.iso_1.levels:
    energies.append(getattr(ch4_read.iso_1,lev).energy)
    lev_quanta.append(sbm.extract_quanta_ch4(getattr(ch4_read.iso_1,lev).lev_string)[0])
    levelli = []
    levelli.append(getattr(ch4_read.iso_1,lev).lev_string)
    levelli += getattr(ch4_read.iso_1,lev).simmetry
    levelsss.append(levelli)

print(energies)
print(levelsss)
print(lev_quanta)
#sys.exit()

filename = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/Old_vts/INP_TEST_OLDs/in_vibtemp.dat'
ch4_readss = sbm.read_tvib_gbb(filename, Atm_nom)
ch4_read_old = ch4_readss[0]

for Qlev in lev_quanta:
    lev_old = ch4_read_old.iso_1.find_level_from_quanta(Qlev)
    lev_read = ch4_read.iso_1.find_level_from_quanta(Qlev)
    if lev_old is None or lev_read is None:
        continue
    print(Qlev)
    #lev_old = getattr(ch4_read_old.iso_1,lev)
    #lev_read = getattr(ch4_read.iso_1,lev)
    print('{}  vs  {}'.format(lev_old.lev_string,lev_read.lev_string))
    for simm in lev_read.simmetry:
        print('READING: {}'.format(simm))
    for simm in lev_old.simmetry:
        print('Wecj: {}'.format(simm))
    altee = lev_old.vibtemp.grid[0]
    write_pr = lev_old.vibtemp.interp_copy('prof',altee)
    read_pr = lev_read.vibtemp.interp_copy('prof',altee)
    pl.plot(write_pr-read_pr, altee)
    #pl.plot(read_pr, altee)
    pl.title(Qlev)
    pl.grid()
    pl.show()
