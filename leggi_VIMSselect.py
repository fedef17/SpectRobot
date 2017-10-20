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
from matplotlib.ticker import FormatStrFormatter
import spect_main_module as smm


# def integr(wl,spe,Range,sol_lim):
#     cond = (wl > Range[0]) & (wl < Range[1]) & (~np.isnan(spe))
#     p1_cond = (wl > sol_lim[0][0]) & (wl < sol_lim[0][1])
#     p2_cond = (wl > sol_lim[1][0]) & (wl < sol_lim[1][1])
#     sol = lambda x: np.nanmean(spe[p1_cond])+(x-np.nanmean(wl[p1_cond]))/(np.nanmean(wl[p2_cond])-np.nanmean(wl[p1_cond]))*(np.nanmean(spe[p2_cond])-np.nanmean(spe[p1_cond]))
#
#     fondo = np.array([sol(wlu) for wlu in wl[cond]])
#     fondo[np.isnan(fondo)] = 0.0
#     intt = np.trapz(spe[cond]-fondo,x=wl[cond])
#     return intt
#
#
# def cbar_things(levels):
#     log2 = int(mt.ceil(mt.log10(np.max(levels)))-1)
#     log1 = int(mt.ceil(mt.log10(np.min(levels)))-1)
#
#     expo = log2
#     if(log1 < log2-1): print('from cbar_things -> Maybe better in log scale?\n')
#
#     if expo == 0 or expo == 1 or expo == -1 or expo == 2:
#         lab = ''
#         expo = 0
#     else:
#         lab = r'$\times 10^{{{}}}$ '.format(expo)
#
#     return expo, lab

########################### MAIN #######################################################

# cub = 'PIXs_HCN-CH4-C2H2_season_szaall_far.sav'
cub = 'PIXs_HCN-CH4-C2H2_season_szaall_far_nu.sav'
cart = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/DATA/'
cubo = io.readsav(cart+cub)

pixs = cubo.compPIX
pixs = pixs[1:]

cubes = np.unique([pix.cubo for pix in pixs])
pix_cubes = dict()
for cub in cubes:
    pix_cubes[cub] = [pix for pix in pixs if pix.cubo == cub]
    print(cub, len(pix_cubes[cub]), pix_cubes[cub][0].dist)

fio = open(cart+'data_cubes_VIMS_far.sav', 'w')
pickle.dump(pix_cubes, fio)
fio.close()

nomefilebands = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/DATA/SAMPLE_DATA/band_titano_2006-2008__lat_EQ_sza30_pha63.dat'
nomefilenoise = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/DATA/SAMPLE_DATA/error_2006-2008__lat_EQ_sza30_pha63.dat'
cubo7418 = smm.comppix_to_pixels(pix_cubes['V1536397418'], nomefilebands, nomefilenoise)

print('ciao')


alts = np.array([1045.12,   940.42,   885.04 ,  769.22,   660.85,   605.33,   501.11])
lat = 50.
sza = 65.

oklats = [pix for pix in cubo7418 if abs(pix.limb_tg_lat-lat) < 5. and abs(pix.limb_tg_sza-sza) < 5.]
oks = [pix for pix in oklats if min(abs(alts-pix.limb_tg_alt)) < 1.]


oklats = [pix for pix in cubo7418 if abs(pix.limb_tg_lat-lat) < 5.]
pl.ion()
angsok = np.linspace(0,2*np.pi,50)
pl.plot(np.cos(angsok),np.sin(angsok))
pl.axis('equal')
coordp = []
for pix in cubo7418: coordp.append((pix.limb_tg_lat*np.pi/180., (2575.+pix.limb_tg_alt)/2575.))
dists = [coo[1] for coo in coordp]
angs = [coo[0] for coo in coordp]
signs = []
for pix in cubo7418:
    if pix.limb_tg_lon < 180.:
        signs.append(1)
    else:
        signs.append(-1)
xs = np.array(dists)*np.cos(angs)*np.array(signs)
ys = np.array(dists)*np.sin(angs)
szas = [pix.limb_tg_sza for pix in cubo7418]
pl.scatter(xs, ys, c = szas, s = 1)
pl.colorbar()
pl.grid()

indok = np.array(szas) < 27.
xsok = xs[indok]
ysok = ys[indok]

pixsza30 = [pix for pix in cubo7418 if pix.limb_tg_sza < 27.]
pl.scatter(xsok,ysok,c='red', s=2)

lats = np.array([pi.limb_tg_lat for pi in pixels])
indok = (np.array(szas) > 80.) & (np.array(szas) < 90.) & (lats > 0)
xsok = xs[indok]
ysok = ys[indok]

pixsza80 = [pix for pix in cubo7418 if pix.limb_tg_sza > 80. and pix.limb_tg_sza < 90. and pix.limb_tg_lat > 0.]
pl.scatter(xsok,ysok,c='red', s = 2)

sys.exit()

t1 = time.time()
print('STARTED AT {}\n'.format(time.ctime()))
time.sleep(2)

cart_orig = '/home/federico/VIMS_data/NEW_COLL_HCN-CH4-C2H2_season_sza80/'
cub = 'PIXs_HCN-CH4-C2H2_season_szaall.sav'
cart = '/home/federico/VIMS_data/NEW_COLL_HCN-CH4-C2H2_season_sza80/Pre/'

cart = '/home/fedefab/Scrivania/Research/Dotto/AbstrArt/CH4_HCN_climatology/DATA/'
cart_orig = cart
cubo = io.readsav(cart_orig+cub)

pixs = cubo.compPIX
pixs = pixs[1:]

print(dir(pixs))
pixs[0].phang

filtro = []
print('FILTRO')
for pix in pixs:
    if np.sum(pix.spet < -3.e-8) > 2 or np.sum(np.abs(pix.spet) < 1e-10) > 5:
        filtro.append(False)
    else:
        filtro.append(True)
filtro = np.array(filtro)
print('END_filtro')
print(len(filtro), np.sum(filtro))

cbarform = '%.1f'
cbarlabel = r'Integrated intensity ({}$W\, m^{{-2}}\, sr^{{-1}}$)'
cbarlabel2 = 'Ratio R_band/P_band'

stp = 100.
alts = np.arange(300.,1101.,stp)

nome = cart + 'Coverage_80-120.pdf'
fig = pl.figure(figsize=(8, 6), dpi=150)
sca = pl.scatter(pixs.year,pixs.lat,c=pixs.sza,cmap='jet_r',s=4,edgecolor='none',vmax = 120, vmin = 80)
cb = pl.colorbar()
cb.set_label('SZA')
pl.xlabel('Time of measurement (year)')
pl.ylabel('Latitude')
pl.grid()
pl.title('Coverage of measurements')
fig.savefig(nome, format='pdf', dpi=150)
pl.close(fig)

nome = cart + 'Coverage_80.pdf'
fig = pl.figure(figsize=(8, 6), dpi=150)
sca = pl.scatter(pixs.year,pixs.lat,c=pixs.sza,cmap='jet_r',s=4,edgecolor='none',vmax = 80)
cb = pl.colorbar()
cb.set_label('SZA')
pl.xlabel('Time of measurement (year)')
pl.ylabel('Latitude')
pl.grid()
pl.title('Coverage of measurements')
fig.savefig(nome, format='pdf', dpi=150)
pl.close(fig)

sza_maxs = [0.,40.,60.,80.,120.,180.]

mean_spets = dict()

for sza_min, sza_max in zip(sza_maxs[:-1], sza_maxs[1:]):
    nome = cart + 'Spettri_medi_{:03d}.pdf'.format(int(sza_max))
    fig = pl.figure(figsize=(8, 6), dpi=150)
    ax1 = pl.subplot()
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    pl.xlabel('Wavelength (nm)')
    pl.ylabel('Radiance (W m$^{-2}$ nm$^{-1}$ sr$^{-1}$)')
    pl.title('Mean spectra at {:3d} < SZA < {:3d}'.format(int(sza_min), int(sza_max)))

    spemed = dict()
    for alt in alts:
        cond = (pixs.alt > alt-stp/2.) & (pixs.alt < alt+stp/2.) & (pixs.sza < sza_max) & (pixs.sza > sza_min) & (filtro) & (pixs.phang < 120.)
        print(sza_max,alt,len(pixs[cond]))
        spe = np.nanmean(np.vstack(pixs[cond].spet), axis = 0)
        #print(spe.shape)
        #spe = np.mean(pixs[cond].spet)
        #print(spe.shape)
        spemed[alt] = spe
        try:
            pl.plot(pixs[100].wl,spe,label='{:5.0f} km'.format(alt))
        except:
            print('EXCEPTION : not plotted')
            continue

    pl.legend()
    fig.savefig(nome, format='pdf', dpi=150)
    mean_spets[(sza_min,sza_max)] = spemed


for alt in alts:
    nome = cart + 'Spettri_medi_varsza_{:03d}.pdf'.format(int(alt))
    fig = pl.figure(figsize=(8, 6), dpi=150)
    ax1 = pl.subplot()
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    pl.xlabel('Wavelength (nm)')
    pl.ylabel('Radiance (W m$^{-2}$ nm$^{-1}$ sr$^{-1}$)')
    pl.title('Mean spectra at different SZA. Alt: {:3.0f} km'.format(alt))
    for cos in mean_spets.keys():
        spe = mean_spets[cos][alt]
        pl.plot(pixs[100].wl,spe,label='{:3.0f}-{:3.0f}'.format(*cos))

    pl.legend()
    fig.savefig(nome, format='pdf', dpi=150)

pickle.dump(mean_spets, open(cart+'mean_spectra.pic','w'))

sys.exit()



n_sza = 10
n_lats = 5
n_seas = 10
#lats = np.linspace(-90,90,n_lats+1)
lats = np.array([-90,-60,-30,30,60,90])
alts = np.arange(350.,1051.,50.) #[300+50*i for i in range(17)]
ucos = np.linspace(1.0,1/np.cos(np.max(pixs.sza)*np.pi/180.),n_sza)
szas = 180*np.arccos(1/ucos)/np.pi
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
pl.title('Mean SZA of measurements (SZA < 80)')
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

limits = [[2970,3090],[3170,3290],[3290,3350],[3350,3470]]
sol_limits = [[2960,2980],[3120,3160],[3480,3500]]


#sca = pl.scatter(pixs.year,pixs.lat,c=pixs.sza,cmap='jet',s=4,edgecolor='none')
#


pl.xlabel('Wavelength (nm)')
pl.ylabel('Intensity (W m-2 nm-1 sr-1)')
pl.grid()
fig.savefig(nome, format='pdf', dpi=150)
pl.close(fig)

int_hcn = []
int_R = []
int_Q = []
int_P = []

for wl,spe in zip(pixs.wl,pixs.spet):
    int_hcn.append(sbm.integr_sol(wl,spe,wl_range=limits[0],sol_lim=sol_limits[:-1]))
    int_R.append(sbm.integr_sol(wl,spe,wl_range=limits[1],sol_lim=sol_limits[1:]))
    int_Q.append(sbm.integr_sol(wl,spe,wl_range=limits[2],sol_lim=sol_limits[1:]))
    int_P.append(sbm.integr_sol(wl,spe,wl_range=limits[3],sol_lim=sol_limits[1:]))

int_hcn = np.array(int_hcn)
int_R = np.array(int_R)
int_Q = np.array(int_Q)
int_P = np.array(int_P)

n_lat_pix = 30
n_alt_pix = 30
int_cont = np.zeros([4,n_lat_pix,n_alt_pix])-1
lat_g,alt_g = np.mgrid[-90:90:n_lat_pix*1j,300:1100:n_alt_pix*1j]
alt_step = alt_g[0,1]-alt_g[0,0]
lat_step = lat_g[1,0]-lat_g[0,0]

for lat,il in zip(lat_g[:,0],range(n_lat_pix)):
    for alt,ia in zip(alt_g[0,:],range(n_alt_pix)):
        cond = (abs(pixs.alt - alt) < alt_step/2) & (abs(pixs.lat - lat) < lat_step/2)
        int_cont[0,il,ia] = np.nanmean(int_hcn[cond])
        int_cont[1,il,ia] = np.nanmean(int_R[cond])
        int_cont[2,il,ia] = np.nanmean(int_Q[cond])
        int_cont[3,il,ia] = np.nanmean(int_P[cond])

conan = (int_cont == -1) | (int_cont < 0) | (np.isnan(int_cont))
int_cont = np.ma.MaskedArray(int_cont,conan)

#pl.hist(int_cont[0,:].compressed(),bins=20)
#pl.show()
#pl.hist(int_cont[1,:].compressed(),bins=20)
#pl.show()
#pl.hist(int_cont[2,:].compressed(),bins=20)
#pl.show()
#pl.hist(int_cont[3,:].compressed(),bins=20)
#pl.show()

nome = cart + 'HCN_int_cont.pdf'
quant = int_cont[0,]
sbm.map_contour(nome, lat_g, alt_g, quant, cbarlabel=cbarlabel, xlabel='Latitude', ylabel='Altitude (km)', ylim = [400,1100])

nome = cart + 'R_int_cont.pdf'
quant = int_cont[1,]
sbm.map_contour(nome, lat_g, alt_g, quant, cbarlabel=cbarlabel, xlabel='Latitude', ylabel='Altitude (km)', ylim = [400,1100])

nome = cart + 'Q_int_cont.pdf'
quant = int_cont[2,]
sbm.map_contour(nome, lat_g, alt_g, quant, cbarlabel=cbarlabel, xlabel='Latitude', ylabel='Altitude (km)', ylim = [400,1100])

nome = cart + 'P_int_cont.pdf'
quant = int_cont[3,]
sbm.map_contour(nome, lat_g, alt_g, quant, cbarlabel=cbarlabel, xlabel='Latitude', ylabel='Altitude (km)', ylim = [400,1100])

nome = cart + 'RvsP_int_cont.pdf'
quant = int_cont[1,]/int_cont[3,]
sbm.map_contour(nome, lat_g, alt_g, quant, cbarlabel=cbarlabel2, xlabel='Latitude', ylabel='Altitude (km)', ylim = [400,1100])


# nome = cart + 'HCN_int_cont.pdf'
# fig = pl.figure(figsize=(8, 6), dpi=150)
# pl.grid()
# pl.xlabel('Latitude')
# pl.ylabel('Altitude (km)')
# pl.ylim(400,1100)
# quant = int_cont[0,]
# ncont = 12
# levels = np.linspace(np.percentile(quant.compressed(),5),np.percentile(quant.compressed(),95),ncont)
# print(levels)
# expo, clab = cbar_things(levels)
# quant = quant/10**expo
# levels = levels/10**expo
# pl.contourf(lat_g,alt_g,quant,ncont=ncont,corner_mask = True,levels = levels, extend = 'both')
# cb = pl.colorbar(format=cbarform, pad = 0.1)
# cb.set_label(cbarlabel.format(clab))
# fig.savefig(nome, format='pdf', dpi=150)
# pl.close(fig)
#
# nome = cart + 'R_int_cont.pdf'
# fig = pl.figure(figsize=(8, 6), dpi=150)
# pl.grid()
# pl.xlabel('Latitude')
# pl.ylabel('Altitude (km)')
# pl.ylim(400,1100)
# quant = int_cont[1,]
# ncont = 12
# levels = np.linspace(np.percentile(quant.compressed(),5),np.percentile(quant.compressed(),95),ncont)
# print(levels)
# expo, clab = cbar_things(levels)
# quant = quant/10**expo
# levels = levels/10**expo
# pl.contourf(lat_g,alt_g,quant,ncont=ncont,corner_mask = True,levels = levels, extend = 'both')
# cb = pl.colorbar(format=cbarform, pad = 0.1)
# cb.set_label(cbarlabel.format(clab))
# fig.savefig(nome, format='pdf', dpi=150)
# pl.close(fig)
#
# nome = cart + 'Q_int_cont.pdf'
# fig = pl.figure(figsize=(8, 6), dpi=150)
# pl.grid()
# pl.xlabel('Latitude')
# pl.ylabel('Altitude (km)')
# pl.ylim(400,1100)
# quant = int_cont[2,]
# ncont = 12
# levels = np.linspace(np.percentile(quant.compressed(),5),np.percentile(quant.compressed(),95),ncont)
# print(levels)
# expo, clab = cbar_things(levels)
# quant = quant/10**expo
# levels = levels/10**expo
# pl.contourf(lat_g,alt_g,quant,ncont=ncont,corner_mask = True,levels = levels, extend = 'both')
# cb = pl.colorbar(format=cbarform, pad = 0.1)
# cb.set_label(cbarlabel.format(clab))
# fig.savefig(nome, format='pdf', dpi=150)
# pl.close(fig)
#
# nome = cart + 'P_int_cont.pdf'
# fig = pl.figure(figsize=(8, 6), dpi=150)
# pl.grid()
# pl.xlabel('Latitude')
# pl.ylabel('Altitude (km)')
# pl.ylim(400,1100)
# quant = int_cont[3,]
# ncont = 12
# levels = np.linspace(np.percentile(quant.compressed(),5),np.percentile(quant.compressed(),95),ncont)
# print(levels)
# expo, clab = cbar_things(levels)
# quant = quant/10**expo
# levels = levels/10**expo
# pl.contourf(lat_g,alt_g,quant,ncont=ncont,corner_mask = True,levels = levels, extend = 'both')
# cb = pl.colorbar(format=cbarform, pad = 0.1)
# cb.set_label(cbarlabel.format(clab))
# fig.savefig(nome, format='pdf', dpi=150)
# pl.close(fig)
#
# nome = cart + 'RvsP_int_cont.pdf'
# fig = pl.figure(figsize=(8, 6), dpi=150)
# pl.grid()
# pl.xlabel('Latitude')
# pl.ylabel('Altitude (km)')
# pl.ylim(400,1100)
# quant = int_cont[1,]/int_cont[3,]
# ncont = 12
# levels = np.linspace(np.percentile(quant.compressed(),5),np.percentile(quant.compressed(),95),ncont)
# print(levels)
# expo, clab = cbar_things(levels)
# quant = quant/10**expo
# levels = levels/10**expo
# pl.contourf(lat_g,alt_g,quant,ncont=ncont,corner_mask = True,levels = levels, extend = 'both')
# cb = pl.colorbar(format=cbarform, pad = 0.1)
# cb.set_label(cbarlabel2)
# fig.savefig(nome, format='pdf', dpi=150)
# pl.close(fig)

climat = np.zeros([len(seas),n_lats,n_sza,len(alts)])

# DI NUOVOVOOVOVO

n_lat_pix = 20
n_time_pix = 10
int_cont = np.zeros([4,n_lat_pix,n_time_pix])-1
lat_g,time_g = np.mgrid[-90:90:n_lat_pix*1j,2004:2013:n_time_pix*1j]
time_step = time_g[0,1]-time_g[0,0]
lat_step = lat_g[1,0]-lat_g[0,0]

for lat,il in zip(lat_g[:,0],range(n_lat_pix)):
    for timei,ia in zip(time_g[0,:],range(n_time_pix)):
        alt_ok = 400.
        cond = (abs(pixs.alt - alt_ok) < 50) & (abs(pixs.lat - lat) < lat_step/2) & (abs(pixs.year - timei) < time_step/2)
        int_cont[0,il,ia] = np.nanmean(int_hcn[cond])
        int_cont[2,il,ia] = np.nanmean(int_Q[cond])
        alt_ok = 1000.
        cond = (abs(pixs.alt - alt_ok) < 50) & (abs(pixs.lat - lat) < lat_step/2) & (abs(pixs.year - timei) < time_step/2)
        int_cont[1,il,ia] = np.nanmean(int_R[cond])
        int_cont[3,il,ia] = np.nanmean(int_P[cond])

conan = (int_cont == -1) | (int_cont < 0) | (np.isnan(int_cont))
int_cont = np.ma.MaskedArray(int_cont,conan)


nome = cart + 'HCN_int_cont_time.pdf'
quant = int_cont[0,]
sbm.map_contour(nome, lat_g, time_g-2000., quant, cbarlabel=cbarlabel, xlabel='Latitude', ylabel='Time (years from 2000)', ylim = [5,13])

nome = cart + 'R_int_cont_time.pdf'
quant = int_cont[1,]
sbm.map_contour(nome, lat_g, time_g-2000., quant, cbarlabel=cbarlabel, xlabel='Latitude', ylabel='Time (years from 2000)', ylim = [5,13])

nome = cart + 'Q_int_cont_time.pdf'
quant = int_cont[2,]
sbm.map_contour(nome, lat_g, time_g-2000., quant, cbarlabel=cbarlabel, xlabel='Latitude', ylabel='Time (years from 2000)', ylim = [5,13])

nome = cart + 'P_int_cont_time.pdf'
quant = int_cont[3,]
sbm.map_contour(nome, lat_g, time_g-2000., quant, cbarlabel=cbarlabel, xlabel='Latitude', ylabel='Time (years from 2000)', ylim = [5,13])

nome = cart + 'RvsP_int_cont_time.pdf'
quant = int_cont[1,]/int_cont[3,]
sbm.map_contour(nome, lat_g, time_g-2000., quant, cbarlabel=cbarlabel2, xlabel='Latitude', ylabel='Time (years from 2000)', ylim = [5,13])


# nome = cart + 'HCN_int_cont_time.pdf'
# fig = pl.figure(figsize=(8, 6), dpi=150)
# pl.grid()
# pl.xlabel('Latitude')
# pl.ylabel('Time (years from 2000)')
# pl.ylim(5,13)
# pl.title('Alt = 400+/-50 km')
# quant = int_cont[0,]
# ncont = 12
# levels = np.linspace(np.percentile(quant.compressed(),5),np.percentile(quant.compressed(),95),ncont)
# print(levels)
# expo, clab = cbar_things(levels)
# quant = quant/10**expo
# levels = levels/10**expo
# pl.contourf(lat_g,time_g-2000,quant,ncont=ncont,corner_mask = True,levels = levels, extend = 'both')
# cb = pl.colorbar(format=cbarform, pad = 0.1)
# cb.set_label(cbarlabel.format(clab))
# fig.savefig(nome, format='pdf', dpi=150)
# pl.close(fig)
#
# nome = cart + 'R_int_cont_time.pdf'
# fig = pl.figure(figsize=(8, 6), dpi=150)
# pl.grid()
# pl.xlabel('Latitude')
# pl.ylabel('Time (years from 2000))')
# pl.title('Alt = 1000+/-50 km')
# pl.ylim(5,13)
# quant = int_cont[1,]
# ncont = 12
# levels = np.linspace(np.percentile(quant.compressed(),5),np.percentile(quant.compressed(),95),ncont)
# print(levels)
# expo, clab = cbar_things(levels)
# quant = quant/10**expo
# levels = levels/10**expo
# pl.contourf(lat_g,time_g-2000,quant,ncont=ncont,corner_mask = True,levels = levels, extend = 'both')
# cb = pl.colorbar(format=cbarform, pad = 0.1)
# cb.set_label(cbarlabel.format(clab))
# fig.savefig(nome, format='pdf', dpi=150)
# pl.close(fig)
#
# nome = cart + 'Q_int_cont_time.pdf'
# fig = pl.figure(figsize=(8, 6), dpi=150)
# pl.grid()
# pl.xlabel('Latitude')
# pl.ylabel('Time (years from 2000))')
# pl.title('Alt = 400+/-50 km')
# pl.ylim(5,13)
# quant = int_cont[2,]
# ncont = 12
# levels = np.linspace(np.percentile(quant.compressed(),5),np.percentile(quant.compressed(),95),ncont)
# print(levels)
# expo, clab = cbar_things(levels)
# quant = quant/10**expo
# levels = levels/10**expo
# pl.contourf(lat_g,time_g-2000,quant,ncont=ncont,corner_mask = True,levels = levels, extend = 'both')
# cb = pl.colorbar(format=cbarform, pad = 0.1)
# cb.set_label(cbarlabel.format(clab))
# fig.savefig(nome, format='pdf', dpi=150)
# pl.close(fig)
#
# nome = cart + 'P_int_cont_time.pdf'
# fig = pl.figure(figsize=(8, 6), dpi=150)
# pl.grid()
# pl.xlabel('Latitude')
# pl.ylabel('Time (years from 2000))')
# pl.title('Alt = 1000+/-50 km')
# pl.ylim(5,13)
# quant = int_cont[3,]
# ncont = 12
# levels = np.linspace(np.percentile(quant.compressed(),5),np.percentile(quant.compressed(),95),ncont)
# print(levels)
# expo, clab = cbar_things(levels)
# quant = quant/10**expo
# levels = levels/10**expo
# pl.contourf(lat_g,time_g-2000,quant,ncont=ncont,corner_mask = True,levels = levels, extend = 'both')
# cb = pl.colorbar(format=cbarform, pad = 0.1)
# cb.set_label(cbarlabel.format(clab))
# fig.savefig(nome, format='pdf', dpi=150)
# pl.close(fig)
#
# nome = cart + 'RvsP_int_cont_time.pdf'
# fig = pl.figure(figsize=(8, 6), dpi=150)
# pl.grid()
# pl.xlabel('Latitude')
# pl.ylabel('Time (years from 2000))')
# pl.title('Alt = 1000+/-50 km')
# pl.ylim(5,13)
# quant = int_cont[1,]/int_cont[3,]
# ncont = 12
# levels = np.linspace(np.percentile(quant.compressed(),5),np.percentile(quant.compressed(),95),ncont)
# print(levels)
# expo, clab = cbar_things(levels)
# quant = quant/10**expo
# levels = levels/10**expo
# pl.contourf(lat_g,time_g-2000,quant,ncont=ncont,corner_mask = True,levels = levels, extend = 'both')
# cb = pl.colorbar(format=cbarform, pad = 0.1)
# cb.set_label(cbarlabel2)
# fig.savefig(nome, format='pdf', dpi=150)
# pl.close(fig)


t2 = time.time()
print('FINISHED AT {}\n'.format(time.ctime()))
print('TOTAL TIME: {} s\n'.format(t2-t1))
