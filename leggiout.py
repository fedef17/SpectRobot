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
import lineshape
import copy
import time
import spect_main_module as smm
import pickletools
import cPickle as pickle

# i0 = 0
# cart = '/home/fedefab/Scrivania/Research/Dotto/Reports/Code_validation/3D_inversion/'
# fil = open(cart+'out_3D_light.pic','r')
i0 = 200
cart = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/Tests/'
fil = open(cart+'out_3D_inversion_test_fast_emilia.pic','r')

sims_it = dict()
retset_it = dict()

for i in range(20):
    print('leggo',i)
    try:
        [num, obs, sims, retset] = pickle.load(fil)
        sims_it[num] = sims
        retset_it[num] = retset
    except:
        break

pl.ion()

for num in sims_it:
    if int(num) > 0:
        break
    sims = sims_it[num]
    pl.figure(i0+num)
    j = 0
    for ob,si in zip(obs[::5],sims[::5]):
        ob.plot(label = 'obs {}'.format(j))
        si.plot(label = 'sim {}'.format(j))
        j+=1
    pl.legend(fontsize = 'small')
    pl.figure(i0+10+num)
    j = 0
    for ob,si in zip(obs[::5],sims[::5]):
        diff = ob-si
        diff.plot(label = 'diff {}'.format(j))
        j+=1
    pl.legend(fontsize = 'small')

    retset = retset_it[num]
    j=0
    for cos in retset.sets:
        j+=1
        logpl = False
        if cos == 'HCN' or cos == 'C2H2':
            logpl = True
        lats = [-90,-75,-60,-30,30,60,75]
        lats = [-60,-30]
        for il,lat in zip(range(len(lats)), lats):
            pl.figure(i0+100+il*10+j)
            pl.title('{} - lat {}'.format(cos, lat))
            pl.grid()
            retset.sets[cos].profile().plot(fix_lat = lat, label = 'It {}'.format(num), logplot = logpl)
            pl.legend()

sys.exit()

fil = open('out_3D_inversion_test1.pic','r')
nuovofil = open('out_3D_light.pic','wb')

for i in range(20):
    print('leggo',i)
    try:
        coso = pickle.load(fil)
    except:
        break
    coso2 = coso[:-1]
    bset = coso[-1]
    for par in bset.params():
        par.erase_hires_deriv()
    coso2.append(bset)
    pickle.dump(coso2, nuovofil)

nuovofil.close()
fil.close()
