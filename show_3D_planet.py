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
#dphi, dtheta = np.pi/250.0, np.pi/250.0
dphi, dtheta = np.pi/90.0, np.pi/360.0
[phi,theta] = np.mgrid[-np.pi/2:np.pi/2+dphi:dphi,0:2*np.pi+dtheta*1.5:dtheta]
dphi1, dtheta1 = np.pi/18.0, np.pi/36.0
[phi1,theta1] = np.mgrid[-np.pi/2:np.pi/2+dphi1:dphi1,0:2*np.pi+dtheta1*1.5:dtheta1]
#print([phi,theta])
#m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4
#r = np.sin(m0*phi)**m1 + np.cos(m2*phi)**m3 + np.sin(m4*theta)**m5 + np.cos(m6*theta)**m7

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
xp,yp,zp = sbm.sphtocart(lat,lon)
xss,yss,zss = sbm.sphtocart(phi_ss,theta_ss)

x = r*np.cos(phi)*np.cos(theta)
y = r*np.cos(phi)*np.sin(theta)
z = r*np.sin(phi)

x1 = r*np.cos(phi1)*np.cos(theta1)
y1 = r*np.cos(phi1)*np.sin(theta1)
z1 = r*np.sin(phi1)

npsza = np.vectorize(sbm.szafromssp)
_sza_ = npsza(phi,theta,phi_ss,theta_ss)

print(x)

# View it.
from mayavi import mlab
s = mlab.mesh(x, y, z, scalars = -_sza_)

p2 = mlab.points3d(xp,yp,zp, scale_factor=200)

from matplotlib import cm
cma1 = cm.get_cmap('coolwarm')
stime = np.mod((theta1 - theta_ss)/(2*np.pi)*24. + 12,24)
stime_2 = np.mod((theta1 - theta_ss)/(2*np.pi)*24. + 6,24)
print(type(stime),np.shape(stime))
print(type(x1),np.shape(x1))
x1_l = np.reshape(x1,(1,-1))
y1_l = np.reshape(y1,(1,-1))
z1_l = np.reshape(z1,(1,-1))
stime_l = np.reshape(stime,(1,-1))
print(np.shape(x1_l[0,:]))

#for px1,py1,pz1,pst in zip(x1_l[0,:],y1_l[0,:],z1_l[0,:],stime_l[0,:]):
colo = [(0,0,0),(1,1,1)]
for i in range(2):
    ore = (stime_2 > i*12) & (stime_2 < (i+1)*12)
    mlab.points3d(x1[ore],y1[ore],z1[ore], scale_factor=50, color = colo[i])

terminator = (np.abs(_sza_-np.pi/2) < np.pi/50.)
# per ogni phi (prima coordinata) c'è un theta con sza > np.pi/2 accanto ad un theta con sza < np.pi/2
terminator1 = []
terminator2 = []
for parallelo,szap,latit in zip(theta,_sza_,phi):
    ce90 = szap-np.pi/2
    cont = 0
    for el,elp in zip(ce90,np.append(ce90[1:],ce90[0])):
        if el*elp < 0:
            cont+=1
    print('Lat {}: found {} intersections'.format(latit[0],cont))
    if cont == 0: continue
    zoio=np.argsort(np.abs(ce90))

    thetas = parallelo[zoio[0:2*cont]]
    ord2 = np.argsort(thetas)
    thetas = np.sort(thetas)
    szaps = szap[zoio[0:2*cont]]
    szaps = szaps[ord2]

    for i in range(cont):
        facts = sbm.weight(np.pi/2, szaps[2*i], szaps[2*i+1])
        theta_ok = facts[0]*thetas[2*i]+facts[1]*thetas[2*i+1]
        print(np.pi/2, szaps[2*i], szaps[2*i+1], facts[0], facts[1], thetas[2*i], thetas[2*i+1], theta_ok)
        if len(terminator2) > 0 and len(terminator1) > 0 and np.abs(theta_ok-terminator1[-1][1]) < np.abs(theta_ok-terminator2[-1][1]):
            terminator1.append([latit[0],theta_ok])
        elif len(terminator1) == 0:
            terminator1.append([latit[0],theta_ok])
        elif len(terminator2) == 0:
            terminator2.append([latit[0],theta_ok])
        else:
            terminator2.append([latit[0],theta_ok])

    # facs_0 = np.abs((np.pi/2-szap[zoio[1]])/(szap[zoio[0]]-szap[zoio[1]]))
    # facs_1 = np.abs((np.pi/2-szap[zoio[0]])/(szap[zoio[0]]-szap[zoio[1]]))
    # print('LAT {} : angoli {} {} {} {} {} {}'.format(latit[0],theta_0,theta_1,szap[zoio[0]],szap[zoio[1]],facs_0,facs_1))
    # theta_ok = facs_0*theta_0+facs_1*theta_1

print(len(terminator1),len(terminator2))
tot = terminator1+list(reversed(terminator2))
tot.append(tot[0])
terminator = np.array(tot)
print(np.shape(terminator))
#print(terminator)
#term = np.meshgrid(terminator[:,0],terminator[:,1])
x_T = r*np.cos(terminator[:,0])*np.cos(terminator[:,1])
y_T = r*np.cos(terminator[:,0])*np.sin(terminator[:,1])
z_T = r*np.sin(terminator[:,0])

#p3 = mlab.points3d(xss,yss,zss, scale_factor=200)
mlab.plot3d(x_T,y_T,z_T,tube_radius = 30,color = (0,0,0))
#mlab.points3d(x_T,y_T,z_T,scale_factor= 100,color = (0,0,0))
#LOS = mlab.plot3d([x1,xc],[y1,yc],[z1,zc], line_width = 100)
#c = mlab.points3d(xc,yc,zc, scale_factor=1000)
#s2 = mlab.mesh(x2, y2, z2, color = (1.,1.,1.), opacity = 0.5)
#s3 = mlab.mesh(x3, y3, z3, color = (1.,1.,1.), opacity = 0.1)


# DEVO definire un piano
# o meglio mi serve l'angolo tra due piani
# quindi direi l'angolo ce l'ho già
# è theta meno il theta del punto subsolare: quello è a ore 12.






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
