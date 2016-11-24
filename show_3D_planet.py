#!/usr/bin/python
# -*- coding: utf-8 -*-

import decimal
import numpy as np
import sys
import os.path
import matplotlib.pyplot as pl
import math as mt
import spect_base_module as sbm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri


# xx = [1,2,3,4,5]
# yy = [1,2,3,4,5]
# zz = [1,2,3,4,5]
#
# fig = pl.figure(figsize=(8, 6), dpi=150)
# ax = Axes3D(fig)
# ax.scatter(xx,yy,zz, marker='o', s=20, c="goldenrod", alpha=0.6)
# ax.view_init(elev=10., azim=0.)
# pl.show()

def segment(r1,r2,n = 100,fraction = 1):
    """
    Gives vector in direction from r1 to r2, starting at r1 and with length fraction*(|r2-r1|). len(vector)=n
    :param r1: Vector 1. (origin of the final vector)
    :param r2: Vector 2.
    :param n:
    :param fraction:
    :return:
    """




# Create the data. TEST for mayavi
dphi, dtheta = np.pi/250.0, np.pi/250.0
dphi, dtheta = np.pi/18.0, np.pi/20.0
[phi,theta] = np.mgrid[-np.pi/2:np.pi/2+dphi:dphi,0:2*np.pi+dtheta*1.5:dtheta]
#print([phi,theta])
m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4
r = np.sin(m0*phi)**m1 + np.cos(m2*phi)**m3 + np.sin(m4*theta)**m5 + np.cos(m6*theta)**m7

R = 2575.
r=R

phi_ss =  -16.71*np.pi/180.0
theta_ss = 307.13*np.pi/180.0

phi_c = -0.39*np.pi/180.0
theta_c = 7.91*np.pi/180.0
dist_c = 1.2868e5

xc,yc,zc = sbm.sphtocart(phi_c,theta_c,R=dist_c)
lat = -35*np.pi/180.0
lon = 279*np.pi/180.0
x1,y1,z1 = sbm.sphtocart(lat,lon)
xss,yss,zss = sbm.sphtocart(phi_ss,theta_ss)

x = r*np.cos(phi)*np.cos(theta)
y = r*np.cos(phi)*np.sin(theta)
z = r*np.sin(phi)

npsza = np.vectorize(sbm.szafromssp)
color = npsza(phi,theta,phi_ss,theta_ss)

print(x)

# View it.
from mayavi import mlab
s = mlab.mesh(x, y, z, scalars = -color)
p = mlab.points3d(x,y,z, scale_factor=50)
p2 = mlab.points3d(x1,y1,z1, scale_factor=200)
p3 = mlab.points3d(xss,yss,zss, scale_factor=200)
LOS = mlab.plot3d([x1,xc],[y1,yc],[z1,zc], line_width = 100)
#c = mlab.points3d(xc,yc,zc, scale_factor=1000)
#s2 = mlab.mesh(x2, y2, z2, color = (1.,1.,1.), opacity = 0.5)
#s3 = mlab.mesh(x3, y3, z3, color = (1.,1.,1.), opacity = 0.1)
mlab.show()

# (n, m) = (50, 50)
#
# # Meshing a unit sphere according to n, m
# theta = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
# phi = np.linspace(np.pi * (-0.5 + 1./(m+1)), np.pi*0.5, num=m, endpoint=False)
# theta, phi = np.meshgrid(theta, phi)
# theta, phi = theta.ravel(), phi.ravel()
# theta = np.append(theta, [0.]) # Adding the north pole...
# phi = np.append(phi, [np.pi*0.5])
# mesh_x, mesh_y = ((np.pi*0.5 - phi)*np.cos(theta), (np.pi*0.5 - phi)*np.sin(theta))
# triangles = mtri.Triangulation(mesh_x, mesh_y).triangles
# x, y, z = np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), np.sin(phi)
#
# # Defining a custom color scalar field
# vals = np.sin(6*phi) * np.sin(3*theta)
# colors = np.mean(vals[triangles], axis=1)
#
# # Plotting
# fig = pl.figure(figsize=(8, 6), dpi=150)
# ax = fig.gca(projection='3d')
# #ax = fig.add_subplot(1, 1, 1, projection='3d')
# #ax = Axes3D(fig)
# cmap = pl.get_cmap('Blues')
# triang = mtri.Triangulation(x, y, triangles)
# collec = ax.plot_trisurf(triang, z, cmap=cmap, shade=False, linewidth=0.)
# collec.set_array(colors)
# collec.autoscale()
# pl.show()