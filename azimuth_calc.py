#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import linalg as LA
import numpy as np
import sys
from math import *
import matplotlib.pyplot as pl

# Constants

pi = 2 * acos(0.0)
Rtit = 2575.0


def rad(ang):
    pi = 2 * acos(0.0)
    return ang*pi/180.0


def deg(ang):
    pi = 2 * acos(0.0)
    return ang*180.0/pi


def sphtocart(lat, lon, h, R_p=Rtit):
    """
    Converts from (lat, lon, alt) to (x, y, z).
    Convention: lon goes from 0 to 360, starting from x axis, towards East. lat is the latitude.
    h is the altitude with respect to the spherical surface. R_p is the planet radius.
    :return: 3D numpy array
    """
    r = [cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat)]
    r = np.array(r)
    r *= (R_p + h)
    return r

#################################################
#       INPUTS                                  #
#################################################

lat = rad(float(sys.argv[1]))
lon = rad(float(sys.argv[2]))
h = float(sys.argv[3])
lat_c = rad(float(sys.argv[4]))
lon_c = rad(float(sys.argv[5]))
d_c = float(sys.argv[6])

#################################################
#       MAIN                                    #
#################################################


# Define quantities

r = sphtocart(lat, lon, 0.0)
r = np.array(r)
_r = r / Rtit   # unit vector, direction: from Titan center to P

_N = [-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat)]
_N = np.array(_N)   # unit vector, direction: from P towards the North

_E = [-sin(lon), cos(lon), 0.0]
_E = np.array(_E)   # unit vector, direction: from P towards the East

C = sphtocart(lat_c, lon_c, d_c-Rtit)
C = np.array(C)
_C = C / d_c    # unit vector, direction: from Titan center to Cassini

LOS = C - r     # LOS. Convention: points from the space towards the telescope

LOSh = LOS - np.dot(LOS, _r) * _r  # the projection of the LOS on the horizontal plane
_LOSh = LOSh / LA.norm(LOSh)

azimuth = acos(np.dot(_LOSh, _N))  # The azimuth with respect to the North

if np.dot(_LOSh, _E) < 0.0:
    azimuth = 2*pi - azimuth  # if it is towards W

azimuth = deg(azimuth)
print('N clockwise. Azimuth = {:7.2f}'.format(azimuth))
azimuth1 = (180.0 + azimuth)%360
print('N clockwise, -LOS; or S clockwise, LOS. Azimuth = {:7.2f}'.format(azimuth1))
azimuth1 = 360.0 - azimuth
print('N anticlockwise. Azimuth = {:7.2f}'.format(azimuth1))
azimuth1 = (180.0 - azimuth)%360
print('S anticlockwise. Azimuth = {:7.2f}'.format(azimuth1))