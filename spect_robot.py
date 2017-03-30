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

# TEST ZONE

planet = sbm.Titan()

spacecraft = sbm.Coords([1e5,0,0])
second = sbm.Coords([3000,0,0])
third = sbm.Coords([1000,2000,2000])

linea1 = sbm.LineOfSight(spacecraft, second)
linea2 = sbm.LineOfSight(spacecraft, third)

linea1.calc_LOS_vector()
#linea1.find_TOA_ingress()
point1 = linea1.calc_atm_intersections(planet)

linea2.calc_LOS_vector()
#linea2.find_TOA_ingress()
point2 = linea2.calc_atm_intersections(planet)

sys.exit()


##### This is the MAIN general program #######
### Please fede don't break the generality here ###

#### READIN INPUTS ####
# Reads inputs: what should I do? simulation? retrieval? calculation of heating/cooling rates? #
# Which are the source files? Observations, geometries, ... other
wn_1 = 2500.
wn_2 = 3500.
wn_step = 0.0005
wn_grid = np.arange(wn_1,wn_2,wn_step)

#### READING SPECTRAL DATABASE ####

db_cart = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/Spect_data/MW_VIMS_CH4_bianca/'

n_mws, mw_tags, mw_ranges = spcl.read_mw_list(db_cart)

linee = []
for i, tag, mw_rng in zip(range(n_mws), mw_tags, mw_ranges):
    db_file = db_cart+'sp_'+tag+'.dat'
    new_linee = spcl.read_line_database(db_file)
    linee += new_linee

print(type(linee), len(linee))

######################################################################

#### Read/Load input atmosphere ####
cart2 = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/T_vibs/Test_wave2/'

ch4_nom, ch4_wave2 = pickle.load(open(cart2+'ch4_Molec_testMaya.pic','r'))
atmosphere = ch4_nom.atmosphere

#### CALCULATING CROSS SECTION AND EMISSIVITY ########
# Here a switch should exist to read them instead of calculating #

opacity, emissivity = spcl.cross_calc(wn_grid,linee,atmosphere,sum_on_levels = True) # the output of this routine is a list of list elements, each one with 3 elements: T,P,opacity(w). If sum_on_levels is False, then each element is made of 4: level,T,P,opacity(w).


#### Load OBSERVING GEOMETRY #######
# Most simple input is defining the observing geometry providing two points: the observer (satellite) and the limb_tangent_point or the surface_interception point. But any point in the ray path will work.
spacecraft_coords = np.array()


geometries = []
geometries.append(sbm.LineOfSight(spacecraft_coords, second_point = tangent_point))
