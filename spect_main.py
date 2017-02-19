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
import dill


db_cart = '/home/fede/Scrivania/Dotto/AbstrArt/CH4_HCN_climatology/Spect_data/MW_VIMS_CH4_bianca/'

n_mws, mw_tags, mw_ranges = spcl.read_mw_list(db_cart)

linee = []
for i, tag, mw_rng in zip(range(n_mws), mw_tags, mw_ranges):
    db_file = db_cart+'sp_'+tag+'.dat'
    new_linee = spcl.read_line_database(db_file)
    linee += new_linee

    if i > 4: break

print(type(linee), len(linee))

cart2 = '/home/fede/Scrivania/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/Test_wave2/'

ch4_nom, ch4_wave2 = pickle.load(open(cart2+'ch4_Molec_testMaya.pic','r'))
#ch4 = dill.load(open(cart2+'ch4_Molec.dil','r'))

cart_old = '/home/fede/Scrivania/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/Old_vts/INP_TEST_OLDs/'

ch4_old = pickle.load(open(cart_old+'TestOld_ch4_Molec.pic','r'))

print(type(ch4_nom), type(ch4_nom.iso_1), type(ch4_nom.iso_1.lev_05), type(ch4_nom.iso_1.lev_05.vibtemp))

print(ch4_old, ch4_old.iso_1, ch4_old.iso_1.levels)#, type(ch4_old.iso_1.lev_05.vibtemp))

print(type(ch4_wave2), type(ch4_wave2.iso_1), type(ch4_wave2.iso_1.lev_05), type(ch4_wave2.iso_1.lev_05.vibtemp))


for cose in zip(ch4_nom.iso_1.lev_05.vibtemp.grid[0], ch4_nom.iso_1.lev_05.vibtemp.prof, ch4_wave2.iso_1.lev_05.vibtemp.prof, ch4_old.iso_1.lev_05.vibtemp.prof):
    print(cose)

for lev in ch4_nom.iso_1.levels:
    print(lev)
    altee = getattr(ch4_old.iso_1,lev).vibtemp.grid[0]
    nom_pr = getattr(ch4_nom.iso_1,lev).vibtemp.interp_copy('prof',altee)
    wave2_pr = getattr(ch4_wave2.iso_1,lev).vibtemp.interp_copy('prof',altee)
    old_pr = getattr(ch4_old.iso_1,lev).vibtemp.prof

    # pl.plot(nom_pr,altee,label='nom')
    # pl.plot(wave2_pr,altee,label='wave2')
    # pl.plot(old_pr,altee,label='old')
    pl.title(lev)
    pl.plot(nom_pr-old_pr,altee,label='nom-old')
    pl.plot(nom_pr-wave2_pr,altee,label='nom-wave2')
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
