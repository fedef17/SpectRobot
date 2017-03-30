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
import matplotlib.colors as colors
import scipy.stats as stats

import scipy.stats as stats
import time

########################### MAIN #######################################################

t1 = time.time()
print('STARTED AT {}\n'.format(time.ctime()))
time.sleep(2)

#cart_orig = '/home/federico/VIMS_data/NEW_COLL_HCN-CH4-C2H2_season_sza80/'
cart_orig = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/Titan_workshop/Data/All_data/'
#cub = 'PIXs_HCN-CH4-C2H2_season_sza80.sav'
cub = 'PIXs_VIMS_4-5mu_night_far.sav'#  PIXs_VIMS_4-5mu_night_far.sav
cart = cart_orig

cubo = io.readsav(cart_orig+cub)

pixs = cubo.compPIX
pixs = pixs[1:]
print('Found {} pixels available \n'.format(len(pixs)))
illo=0
for cub in np.unique(np.array([pix.cubo for pix in pixs])):
    print('Cubo {}: {}\n'.format(illo,cub))
    illo+=1


cbarform = '%.1f'
cbarlabel = r'Integrated intensity ({}$W\, m^{{-2}}\, sr^{{-1}}$)'
cbarlabel2 = 'Ratio R_band/P_band'


nome = cart + 'Coverage.pdf'
fig = pl.figure(figsize=(8, 6), dpi=150)
sca = pl.scatter(pixs.year,pixs.lat,c=pixs.sza,cmap='jet',s=4,edgecolor='none')
cb = pl.colorbar()
cb.set_label('SZA')
pl.xlabel('Time of measurement (year)')
pl.ylabel('Latitude')
pl.grid()
pl.title('Coverage of measurements')
fig.savefig(nome, format='pdf', dpi=150)
pl.close(fig)

n_sza = 10
n_lats = 5
n_seas = 10
#lats = np.linspace(-90,90,n_lats+1)
lats = np.array([-90,-60,-30,30,60,90])
alts = np.arange(0.,301.,50.) #[300+50*i for i in range(17)]
#ucos = np.linspace(1.0,1/np.cos(np.max(pixs.sza)*np.pi/180.),n_sza)
#szas = 180*np.arccos(1/ucos)/np.pi
szas = np.arange(90,181,30)
print(szas)
seas = np.linspace(2004,2013,n_seas)
stepyear = seas[1]-seas[0]

n_clim = np.zeros([len(seas),n_lats])
sza_clim = np.zeros([len(seas),n_lats])

for yea,iy in zip(seas,range(n_seas)):
    for lat0,lat1,il in zip(lats[:-1],lats[1:],range(n_lats)):
        cond = (abs(pixs.year-yea)<stepyear) & (pixs.lat>lat0) & (pixs.lat<lat1)
        n_clim[iy,il] = len(pixs[cond])
        sza_clim[iy,il] = np.mean(pixs[cond].sza)
        if len(pixs[cond]) == 0:
            sza_clim[iy,il] = np.nan

seas_lim = np.append([seas[0]-stepyear/2],seas+stepyear/2)

conan = np.isnan(sza_clim)
sza_clim = np.ma.MaskedArray(sza_clim,conan)

nome = cart + 'Coverage_boxes_sza.pdf'
fig = pl.figure(figsize=(8, 6), dpi=150)
ax=pl.subplot()
print(len(lats[:-1]),len(seas),np.shape(sza_clim))
histsza = pl.contourf(seas,(lats[1:]+lats[:-1])/2,sza_clim.T,corner_mask = True)
ax.set_xticks(seas)
ax.set_xticks(seas_lim, minor=True)
ax.grid(which='minor', alpha=0.5)
#pl.xticks(seas)
pl.yticks(lats)
ax.set_yticks(lats, minor=True)
cb = pl.colorbar()
cb.set_label('SZA')
pl.xlabel('Time of measurement (year)')
pl.ylabel('Latitude')
#pl.grid()
pl.title('Mean SZA of measurements')
fig.savefig(nome, format='pdf', dpi=150)
pl.close(fig)

nome = cart + 'Coverage_boxes.pdf'
fig = pl.figure(figsize=(8, 6), dpi=150)
ax = pl.subplot()
latss = [lat[0] for lat in pixs.lat]
yearss = [year[0] for year in pixs.year]
hist2d = pl.hist2d(yearss,latss,bins=[list(seas_lim),list(lats)],norm=colors.LogNorm(),cmin=1e2,cmax=1e4)
ax.set_xticks(seas)
ax.set_xticks(seas_lim, minor=True)
ax.grid(which='minor', alpha=0.5)
#pl.xticks(seas)
pl.yticks(lats)
ax.set_yticks(lats, minor=True)
cb = pl.colorbar()
cb.set_label('Number of measurements')
pl.xlabel('Time of measurement (year)')
pl.ylabel('Latitude')
#pl.grid()
pl.title('Coverage of measurements')
fig.savefig(nome, format='pdf', dpi=150)
pl.close(fig)

#integralini: da 2970 a 3090 per HCN+C2H2, da 3170 a 3290 per CH4-R, da 3290 a 3350 per CH4-Q, da 3350 a 3470 per CH4-P

limits = [[4450,4550],[4900,5100]]
#sol_limits = [[2960,2980],[3120,3160],[3480,3500]]

#sca = pl.scatter(pixs.year,pixs.lat,c=pixs.sza,cmap='jet',s=4,edgecolor='none')
#


pl.xlabel('Wavelength (nm)')
pl.ylabel('Intensity (W m-2 nm-1 sr-1)')
pl.grid()
fig.savefig(nome, format='pdf', dpi=150)
pl.close(fig)

int_45 = []
int_49 = []
scatt_lev = []

for wl,spe,pix in zip(pixs.wl,pixs.spet,pixs):
    media_scat = sbm.integr_sol(wl,spe,wl_range=[4200.,4400.])/200.
    scatt_lev.append(media_scat)
    if media_scat > 0.6e-7:
        print('Scarto pixel del cubo {} con sza {} e pha {}, troppo scattering: {}.'.format(pix.cubo,pix.sza,pix.phang,media_scat))
    int_45.append(sbm.integr_sol(wl,spe,wl_range=limits[0]))
    int_49.append(sbm.integr_sol(wl,spe,wl_range=limits[1]))

#pl.hist(scatt_lev,bins=30,range=[0.,1e-6])
#pl.show()

int_45 = np.array(int_45)
int_49 = np.array(int_49)
scatt_lev = np.array(scatt_lev)
print(int_45,int_49)


n_lat_pix = 6
n_alt_pix = 6
alt_1 = 50
alt_2 = 350

MAX_SCATT = 4e-7

nome = cart + 'Spettri_medi.pdf'
fig = pl.figure(figsize=(8, 6), dpi=150)
ax = pl.subplot()
spemed = []
for alt in alts:
    cond = (pixs.alt > alt-25) & (pixs.alt < alt+25) & (scatt_lev < MAX_SCATT)
    spe = np.mean(pixs[cond].spet)
    spemed.append(spe)
    try:
        pl.plot(pixs[1].wl,spe,label='{:5.0f} km'.format(alt))
    except:
        print('EXCEPTION : not plotted')
        continue

int_cont = np.zeros([2,n_lat_pix,n_alt_pix])-1
lat_g,alt_g = np.mgrid[-90:90:n_lat_pix*1j,alt_1:alt_2:n_alt_pix*1j]
print(alt_g)
alt_step = alt_g[0,1]-alt_g[0,0]
lat_step = lat_g[1,0]-lat_g[0,0]

for lat,il in zip(lat_g[:,0],range(n_lat_pix)):
    for alt,ia in zip(alt_g[0,:],range(n_alt_pix)):
        cond = (abs(pixs.alt - alt) < alt_step/2) & (abs(pixs.lat - lat) < lat_step/2) & (scatt_lev < MAX_SCATT)
        int_cont[0,il,ia] = np.nanmean(int_45[cond])
        int_cont[1,il,ia] = np.nanmean(int_49[cond])

conan = (int_cont == -1) | (int_cont < 0) | (np.isnan(int_cont))
int_cont = np.ma.MaskedArray(int_cont,conan)


print('-----------------------------qqqqqqq-----------------------------------')
print('----------------------------------------------------------------')
print(int_cont)
print('----------------------------------------------------------------')
print('----------------------------qqqqqqqqqqqqqq------------------------------------')

nome = cart + 'Aer45_int_cont.pdf'
quant = int_cont[0,]
sbm.map_contour(nome, lat_g, alt_g, quant, cbarlabel=cbarlabel, xlabel='Latitude', ylabel='Altitude (km)', ylim = [50,350])

nome = cart + 'Aer49_int_cont.pdf'
quant = int_cont[1,]
sbm.map_contour(nome, lat_g, alt_g, quant, cbarlabel=cbarlabel, xlabel='Latitude', ylabel='Altitude (km)', ylim = [50,350])

climat = np.zeros([len(seas),n_lats,n_sza,len(alts)])

# DI NUOVOVOOVOVO

n_lat_pix = 20
n_time_pix = 10
int_cont = np.zeros([2,n_lat_pix,n_time_pix])-1
lat_g,time_g = np.mgrid[-90:90:n_lat_pix*1j,2004:2013:n_time_pix*1j]
time_step = time_g[0,1]-time_g[0,0]
lat_step = lat_g[1,0]-lat_g[0,0]

for lat,il in zip(lat_g[:,0],range(n_lat_pix)):
    for timei,ia in zip(time_g[0,:],range(n_time_pix)):
        alt_ok = 150.
        cond = (abs(pixs.alt - alt_ok) < 50) & (abs(pixs.lat - lat) < lat_step/2) & (abs(pixs.year - timei) < time_step/2) & (scatt_lev < MAX_SCATT)
        int_cont[0,il,ia] = np.nanmean(int_45[cond])
        int_cont[1,il,ia] = np.nanmean(int_49[cond])

conan = (int_cont == -1) | (int_cont < 0) | (np.isnan(int_cont))
int_cont = np.ma.MaskedArray(int_cont,conan)

nome = cart + 'Aer45_int_cont_time.pdf'
quant = int_cont[0,]
sbm.map_contour(nome, lat_g, time_g-2000., quant, cbarlabel=cbarlabel, xlabel='Latitude', ylabel='Time (years from 2000)', ylim = [5,13])

nome = cart + 'Aer49_int_cont_time.pdf'
quant = int_cont[1,]
sbm.map_contour(nome, lat_g, time_g-2000., quant, cbarlabel=cbarlabel, xlabel='Latitude', ylabel='Time (years from 2000)', ylim = [5,13])


figuz = pl.figure()
cond = scatt_lev < MAX_SCATT
pl.scatter(int_45[cond],int_49[cond],color='black')
cond = scatt_lev > MAX_SCATT
pl.scatter(int_45[cond],int_49[cond],color='red')
pl.show()

t2 = time.time()
print('FINISHED AT {}\n'.format(time.ctime()))
print('TOTAL TIME: {} s\n'.format(t2-t1))
