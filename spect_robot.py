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
planet.add_default_atm()

spacecraft = sbm.Coords([1e5,0,0])
second = sbm.Coords([3000,0,0])
third = sbm.Coords([0,90,500],s_ref='Spherical')
puntosfera = sbm.Coords([30,0,2000],s_ref='Spherical')

linea1 = sbm.LineOfSight(spacecraft, second)
linea2 = sbm.LineOfSight(spacecraft, third)
linea1.details()
linea2.details()

ssp = sbm.Coords(np.array([5,80,0]),s_ref='Spherical')

point1 = linea1.calc_atm_intersections(planet)
point2 = linea2.calc_atm_intersections(planet)
psza1 = linea1.calc_SZA_along_los(planet,ssp)
psza2 = linea2.calc_SZA_along_los(planet,ssp)

pt1 = linea1.calc_along_LOS(planet.atmosphere, 'temp', set_attr = True)
pp1 = linea1.calc_along_LOS(planet.atmosphere, 'pres', set_attr = True)
pt2 = linea2.calc_along_LOS(planet.atmosphere, 'temp', set_attr = True)
pp2 = linea2.calc_along_LOS(planet.atmosphere, 'pres', set_attr = True)

alts1 = [point.Spherical()[2] for point in point1]
alts2 = [point.Spherical()[2] for point in point2]

carta = '/home/fedefab/Scrivania/Research/Dotto/Reports/Rep_040417/imgs/'

fig = pl.figure(figsize=(8, 6), dpi=150)
pl.grid()
pl.ylabel('Vib Temp (K)')
pl.xlabel('LOS point (km)')
pl.title('Vibrational temp. along LOS 2')

enne = len(alts2)-1
for lev in planet.gases['CH4'].iso_1.levels:
    levv = getattr(planet.gases['CH4'].iso_1, lev)
    proff = levv.vibtemp
    pvi1 = linea1.calc_along_LOS(proff, 'temp', set_attr = True, set_attr_name = lev+'_vt')
    pvi2 = linea2.calc_along_LOS(proff, 'temp', set_attr = True, set_attr_name = lev+'_vt')
    #pl.plot(pvi1,alts1[:-1])
    pl.plot(np.linspace(0,5*enne,enne),pvi2,label=lev)

pl.plot(np.linspace(0,5*enne,enne),pt2,label='Kin Temp')
pl.legend(loc=4)
fig.savefig(carta+'Vib_temps_LOS1.pdf', format='pdf', dpi=150)
pl.close()


for i,p1,ps1,t,p in zip(range(len(point1)), point1, psza1, pt1, pp1):
    print('Punto {}, coordinate cart: {}, sferiche: {}. SZA: {}. T,p: {} K, {} hPa\n'.format(i,p1.Cartesian(),p1.Spherical(),ps1,t,p))

for i,p1,ps1,t,p in zip(range(len(point2)), point2, psza2, pt2, pp2):
    print('Punto {}, coordinate cart: {}, sferiche: {}. SZA: {}. T,p: {} K, {} hPa\n'.format(i,p1.Cartesian(),p1.Spherical(),ps1,t,p))


fig = pl.figure(figsize=(8, 6), dpi=150)
pl.grid()
pl.ylabel('SZA (deg)')
pl.xlabel('LOS point (km)')
pl.title('SZA along LOS')
# pl.plot(pt1,alts1[:-1])
# pl.plot(pt2,alts2[:-1])
# pl.show()
pl.plot(np.linspace(0,5*enne,enne+1),psza2)
#pl.legend(loc=4)
fig.savefig(carta+'SZA_LOS2.pdf', format='pdf', dpi=150)
pl.close()


enne = len(alts1)-1

fig = pl.figure(figsize=(8, 6), dpi=150)
pl.grid()
pl.ylabel('Vib Temp (K)')
pl.xlabel('LOS point (km)')
pl.title('Vibrational temp. along LOS 1')

for lev in planet.gases['CH4'].iso_1.levels:
    levv = getattr(planet.gases['CH4'].iso_1, lev)
    proff = levv.vibtemp
    pvi1 = linea1.calc_along_LOS(proff, 'temp', set_attr = True, set_attr_name = lev+'_vt')
    pvi2 = linea2.calc_along_LOS(proff, 'temp', set_attr = True, set_attr_name = lev+'_vt')
    #pl.plot(pvi1,alts1[:-1])
    pl.plot(pvi1,alts1[:-1],label=lev)

pl.plot(pt1,alts1[:-1],label='Kin Temp')
pl.legend(loc=4)
fig.savefig(carta+'Vib_temps_LOS1_alt.pdf', format='pdf', dpi=150)
pl.close()

fig = pl.figure(figsize=(8, 6), dpi=150)
pl.grid()
pl.ylabel('SZA (deg)')
pl.xlabel('LOS point (km)')
pl.title('SZA along LOS')
# pl.plot(pt1,alts1[:-1])
# pl.plot(pt2,alts2[:-1])
# pl.show()
#pl.plot(psza1,alts1)
pl.plot(np.linspace(0,5*enne,enne+1),psza1)
#pl.legend(loc=4)
fig.savefig(carta+'SZA_LOS1.pdf', format='pdf', dpi=150)
pl.close()

#
# pl.plot(pp1,alts1[:-1])
# pl.plot(pp2,alts2[:-1])
# pl.show()
#
# pl.plot(psza1,alts1)
# pl.plot(psza2,alts2)
# pl.show()

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
    new_linee = spcl.read_line_database(db_file, link_to_isomolecs = [ch4_nom.iso_1])
    linee += new_linee

print(type(linee), len(linee))

opacity = dict([])
emissivity = dict([])

LTE_lines = [line for line in linee if line.Up_lev_id is None]
opacityLTE, emissivityLTE = spcl.cross_calc(wn_grid,linee,PT_points,sum_on_levels = True, nlte = False)

opacity['LTE_lines'] = opacityLTE
emissivity['LTE_lines'] = emissivityLTE

for lev in ch4_nom.iso_1.levels:
    opac, emiss = spcl.cross_calc(wn_grid,linee,PT_points,sum_on_levels = True, nlte = True, vibtemp = vt_points)
    opacity[lev] = opac
    emissivity[lev] = emiss

# BOH YIPO OAJFA OAOFJAOJ


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
