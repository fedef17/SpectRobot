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
