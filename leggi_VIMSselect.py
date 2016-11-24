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

cart = '/home/fede/Scrivania/Dotto/AbstrArt/CH4_HCN_climatology/Tit_atm/DATI/'
cub = 'PIXs_HCN-CH4-C2H2_season.sav'

cart2 = '/home/fede/Scrivania/Dotto/AbstrArt/Titan_workshop/Data/All_data/'
cub2 = 'PIXs_VIMS_4-5mu_night2.sav'
cub3 = 'PIXs_VIMS_4-5mu_night_far.sav'


cubo = io.readsav(cart+cub)
pixs = cubo.compPIX



