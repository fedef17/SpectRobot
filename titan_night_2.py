#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os.path
import matplotlib.pyplot as pl
import math as mt
import spect_base_module as sbm
import matplotlib.gridspec as gridspec
from scipy.interpolate import PchipInterpolator as spline
from scipy.signal import savgol_filter as savgol
import pickle


# SETTO LE COSE DA COSARE

alt_min = 100.0
alt_max = 300.0

tag = '50ppm_nosm'

#names = ['CO','COiso','CH3D_v2', 'CH3D_oth', 'CO2', 'CH4']
names = ['CO','COiso','CH3D_v2', 'CH3D_oth', 'CO2', 'CH4']
#namesfi = ['CO','COiso_LTE_50ppm','CH3D_v2', 'CH3D_oth', 'CO2', 'CH4']
namesfi = names

col1=5 #5
col2=8 #8
alp = 0.2 #0.4







# leggo errori sulla media
lui = open('/home/fede/Scrivania/Dotto/AbstrArt/Titan_workshop/Data/Close/OBS_2006-2008__lat_EQ_sza160/means_2006-2008__lat_EQ_sza160_pha115.dat','r')

for i in range(29):
    lui.readline()

data = np.array([list(map(float, line.split())) for line in lui])
fr_err = data[:,0]
err_tot = np.mean(data[:,17*4:17*4+18],axis=1)

opu = sbm.find_incl(fr_err,4150,5100)
err_tot = err_tot[opu]
fr_err = fr_err[opu]

# pl.plot(fr_err,err_tot)
# pl.scatter(fr_err,err_tot)
# pl.show()
#
# sys.exit()

# BBB = np.vectorize(sbm.BB_nm)
# w = np.linspace(5.,20000.,1000)
# print(len(w))
# bubo = BBB(175.0,w)
# sls = np.trapz(bubo,x=w)
# print(mt.pi*sls/175.0**4)
# pl.plot(w,bubo)
# pl.show()

# devi:
#
# 1 - leggere i file delle sim a bassa / alta risoluzione
# 2 - fare i graficini (data/sim_singole e calcolare i residui)
# 3 - estrarre nuova extinction
# 4 - ricalcolare con nuova extinction
#
# poi:
# 5 - i punti da 2 a 4 magari farli per più dati
# 6 - fare le prove con i dati di giorno
cart = '/home/fede/Scrivania/Dotto/AbstrArt/Titan_workshop/titan_night/'
cart2 = '/home/fede/Scrivania/Dotto/AbstrArt/Titan_workshop/Simulz/NIGHT/'

alt_atm = np.linspace(0,1500,151)
temp = sbm.read_input_prof_gbb(cart + 'in_temp.dat', 'temp')
alt_atm2 = np.linspace(0,1500,301)
temp2 = np.interp(alt_atm2,alt_atm,temp)


[fr_grid,BBtau] = pickle.load(open('BBtau.pic','r'))

alt_los = np.array([50.0 + 25.0*i for i in range(17)])

# BBB1 = np.vectorize(sbm.BB)
#
# for alt,bibbi in zip(alt_los,BBtau):
#     print(alt)
#     ttt = sbm.findT(alt,alt_atm2,temp2)
#     pl.title(str(int(alt)))
#     pl.plot(fr_grid,bibbi,label='B$_{eff}$')
#     pl.plot(fr_grid,BBB1(ttt,fr_grid),label='B')
#     pl.legend()
#     pl.show()

#sys.exit()



# lista_CO = open(cart2 + 'lista_CO','r')
# lista_COiso = open(cart2 + 'lista_COiso','r')
# lista_CH3D_v2 = open(cart2 + 'lista_CH3D_v2','r')
# lista_CH3D_oth = open(cart2 + 'lispickle.dump([ext200_Lavvas,ssa200_Lavvas],open('lavvas.pic','w'))ta_CH3D_oth','r')
# lista_CH4 = open(cart2 + 'lista_CH4','r')
# lista_CO2 = open(cart2 + 'lista_CO2','r')

alts = np.linspace(50.0,450.0,(450-50)/25+1)
alts = alts[::-1]
i=0

# resid_alt = []
# obs_alt = []
# sim_alt = []
si = 1

for alt in alts:
    i+=1
    first = 1
    for name,namefi in zip(names,namesfi):
        fif = cart2 + 'Sim_'+namefi +'/sim_'+'{0:0=2d}'.format(int(i))+'.dat'
        alt_ok,freq,obs,sim,err = sbm.read_sim_gbb(fif, skip_first = 8, skip_last = 2)
        if first:
            sims = sim
            first = 0
        else:
            sims = np.vstack([sims,sim])
    simtot = np.zeros(len(freq))
    for simul in sims:
        simtot = simtot + simul
    resid = obs - simtot
    print(si)
    if si:
        print('entro')
        resid_alt = resid
        obs_alt = obs
        sim_alt = simtot
        si = 0
    else:
        print('entro')
        sim_alt = np.vstack([sim_alt,simtot])
        obs_alt = np.vstack([obs_alt,obs])
        resid_alt = np.vstack([resid_alt,resid])

    #qui aggiungo l'aer

    sbm.plotta_sim_VIMS(cart + 'Sim_'+tag+str(int(alt))+'.eps',freq,obs,simtot,sims,names,
                        err=5e-9,title='Limb at '+ str(int(alt))+' km', auto = False,
                        yscale=[0.0,1.4e-7], yscale_res=[-1e-8,7e-8], xscale = [4100.,5100.])

print(resid_alt)

title = 'Residuals'
fig = pl.figure(figsize=(8, 6), dpi=150)
pl.xlabel('Wavelength (nm)')
pl.ylabel('Radiance (W m$^{-2}$ nm$^{-1}$ sr$^{-1}$)')
pl.title(title)
i=0
for alt,simu in zip(alts[::2],resid_alt[::2]):
    if alt < alt_min or alt > alt_max:
        continue
    colo, li = sbm.findcol(6,i)
    pl.plot(freq,simu,color=colo,linewidth=2.0,label=str(int(alt))+' km')
    i +=1
pl.grid()
pl.legend(loc=2,fontsize='small',fancybox=1,shadow=1)
nomefile = cart + 'Residuals_'+tag+'.eps'
fig.savefig(nomefile, format='eps', dpi=150)
pl.close()

pickle.dump([alts,freq,resid_alt],open('resid.pic','w'))

title = 'Data'
fig = pl.figure(figsize=(8, 6), dpi=150)
pl.xlabel('Wavelength (nm)')
pl.ylabel('Radiance (W m$^{-2}$ nm$^{-1}$ sr$^{-1}$)')
pl.title(title)
pl.xlim(4100.,5100.)
i=0
for alt,simu in zip(alts[::2],obs_alt[::2]):
    if alt < alt_min or alt > alt_max:
        continue
    colo, li = sbm.findcol(6,i)
    pl.plot(freq,simu,color=colo,linewidth=2.0,label=str(int(alt))+' km')
    i +=1
pl.grid()
pl.scatter(5000.,1e-7, color = 'black')
pl.errorbar(5000.,1e-7, yerr=5e-9, linewidth=2.0,color = 'black')
pl.legend(loc=2,fontsize='small',fancybox=1,shadow=1)
nomefile = cart + 'Observed'+tag+'.eps'
fig.savefig(nomefile, format='eps', dpi=150)
pl.close()


title = 'Residuals smooth'
fig = pl.figure(figsize=(8, 6), dpi=150)
pl.xlabel('Wavelength (nm)')
pl.ylabel('Radiance (W m$^{-2}$ nm$^{-1}$ sr$^{-1}$)')
pl.title(title)
pl.xlim(4100.,5100.)
i=0
resid_smooth = freq
for alt,simu in zip(alts,resid_alt):
    #spl = savgol(simu, 5, 2)
    spl = simu
    resid_smooth = np.vstack([resid_smooth,spl])
    colo, li = sbm.findcol(6,i)
    if alt < alt_min or alt > alt_max or alt % 50 > 0:
        continue
    pl.plot(freq,spl,color=colo,linewidth=2.0,label=str(int(alt))+' km')
    i +=1
pl.grid()
pl.legend(loc=2,fontsize='small',fancybox=1,shadow=1)
nomefile = cart + 'Residuals_smooth'+tag+'.eps'
fig.savefig(nomefile, format='eps', dpi=150)
pl.close()

resid_smooth = resid_smooth[1:]

BBB = np.vectorize(sbm.BB_nm)


[alt_los2,tau_alt] = pickle.load(open('tau_alt.pic','r'))
[wl_lav,ext200_Lavvas,ssa200_Lavvas] = pickle.load(open('lavvas.pic','r'))
lav5000 = sbm.find_near(wl_lav,5000.)

tau_alt = tau_alt*(1-ssa200_Lavvas[lav5000])

title = 'Observed absorption optical depth'
fig = pl.figure(figsize=(8, 6), dpi=150)
ax1 = fig.add_subplot(111)
pl.xlabel('Wavelength (nm)')
pl.ylabel('Absorption optical depth')
pl.title(title)
pl.xlim(4100.,5100.)
pl.ylim(-0.1,0.4)

fig2 = pl.figure(figsize=(8, 6), dpi=150)
pl.xlabel('Wavelength (nm)')
pl.ylabel('$S(\lambda)$')
pl.title('$S_{new}(\lambda)$')
pl.xlim(4100.,5100.)
ax2 = fig2.add_subplot(111)


i=0
opt_dep = freq
shape_alt = freq
o5000 = sbm.find_near(freq,5000.0)
print(freq)
#pl.yscale('log')
for alt,resid in zip(alts,resid_smooth):
    ttt = sbm.findT(alt,alt_atm2,temp2)
    print(alt,ttt)
    oki = sbm.find_near(alt_los,alt)
    bibbi = sbm.freqTOwl(fr_grid,BBtau[oki,:],freq,15.0*np.ones(len(freq)))
    #opt = resid/BBB(ttt,freq)
    opt = resid/bibbi
    opt_dep = np.vstack([opt_dep,opt])
    #shape_alt = np.vstack([shape_alt,opt/opt[o5000]])
    shape_alt = np.vstack([shape_alt,opt/tau_alt[oki]])
    print(i)
    colo, li = sbm.findcol(10,i)
    if alt < alt_min or alt > alt_max:# or alt % 50 > 0:
        continue
    print(i)
    ax1.plot(freq,opt,color=colo,linewidth=2.0,label=str(int(alt))+' km')
    #ax2.plot(freq,opt/opt[o5000],color=colo,linewidth=2.0,label=str(int(alt))+' km')
    ax2.plot(freq,opt/tau_alt[oki],color=colo,linewidth=2.0,label=str(int(alt))+' km')
    i +=1
ax1.grid()
ax1.legend(loc=1,bbox_to_anchor=(1.05,1.05), fontsize='x-small',fancybox=1,shadow=1)
ax2.grid()
ax2.legend(loc=1,bbox_to_anchor=(1.05,1.05), fontsize='x-small',fancybox=1,shadow=1)
#pl.show(fig)
#pl.show(fig2)

nomefile = cart + 'Opt_depth_'+tag+'.eps'
fig.savefig(nomefile, format='eps', dpi=150)
nomefile = cart + 'Spect_shape_'+tag+'.eps'
fig2.savefig(nomefile, format='eps', dpi=150)

pl.close(fig)
pl.close(fig2)

opt_dep = opt_dep[1:]
shape_alt = shape_alt[1:]

print(shape_alt)

print(np.shape(shape_alt), np.shape(freq))

ok = sbm.find_incl(alts,alt_min,275.0)
shape_mea = np.mean(shape_alt[ok,:], axis=0)
err_mea = np.std(shape_alt[ok,:], axis=0)




fig = pl.figure(figsize=(8, 6), dpi=150)
ax1 = fig.add_subplot(111)
pl.xlabel('Wavelength (nm)')
pl.ylabel('$S(\lambda)$')
pl.title('New shape')

ok1 = sbm.find_incl(freq,4300.,5100.)
ok2 = sbm.find_incl(wl_lav,4300.,5100.)
lav5000 = sbm.find_near(wl_lav,5000.)

pl.grid()
pl.ylim(-0.2,3.5)
pl.plot(wl_lav[ok2], ext200_Lavvas[ok2]*(1-ssa200_Lavvas[ok2])/(1-ssa200_Lavvas[lav5000]), color = sbm.findcol(6,2)[0],
        linewidth=3.0, label = '$S_{Lav}(\lambda)$')
pl.plot(freq[ok1], shape_mea[ok1], color = sbm.findcol(6,col1)[0], linewidth=3.0, label = '$S_{new}(\lambda)$')
pl.fill_between(freq[ok1], shape_mea[ok1]+err_mea[ok1], shape_mea[ok1]-err_mea[ok1], facecolor=sbm.findcol(12,col2)[0], alpha=alp)

pl.legend(loc=1)
#pl.show()
nomefile = cart + 'NewSHAPE_'+tag+'.eps'
fig.savefig(nomefile, format='eps', dpi=150)
pl.close()


fig = pl.figure(figsize=(8, 6), dpi=150)
ax1 = fig.add_subplot(111)
pl.xlabel('Wavenumber (cm$^{-1}$)')
pl.ylabel('$S(w)$')
pl.title('New shape')
pl.ylim(-0.2,3.5)

ok1 = sbm.find_incl(freq,4300.,5100.)
ok2 = sbm.find_incl(wl_lav,4300.,5100.)
lav5000 = sbm.find_near(wl_lav,5000.)

pl.grid()
pl.plot(1e7/wl_lav[ok2], ext200_Lavvas[ok2]*(1-ssa200_Lavvas[ok2])/(1-ssa200_Lavvas[lav5000]), color = sbm.findcol(6,2)[0], linewidth=3.0, label = '$S_{Lav}(w)$')
pl.plot(1e7/freq[ok1], shape_mea[ok1], color = sbm.findcol(6,col1)[0], linewidth=3.0, label = '$S_{new}(w)$')
pl.fill_between(1e7/freq[ok1], shape_mea[ok1]+err_mea[ok1], shape_mea[ok1]-err_mea[ok1], facecolor=sbm.findcol(12,col2)[0], alpha=alp)

pl.legend(loc=2)
#pl.show()
nomefile = cart + 'NewSHAPE_cm_'+tag+'.eps'
fig.savefig(nomefile, format='eps', dpi=150)
pl.close()


fig = pl.figure(figsize=(8, 6), dpi=150)
ax1 = fig.add_subplot(111)
pl.xlabel('Wavenumber (cm$^{-1}$)')
pl.ylabel('$S(w)$')
pl.title('New shape')

ok1 = sbm.find_incl(freq,4300.,4700.)
ok2 = sbm.find_incl(wl_lav,4300.,4700.)
lav5000 = sbm.find_near(wl_lav,5000.)

pl.grid()
pl.ylim(-0.2,3.5)
pl.plot(1e7/wl_lav[ok2], ext200_Lavvas[ok2]*(1-ssa200_Lavvas[ok2])/(1-ssa200_Lavvas[lav5000]), color = sbm.findcol(6,2)[0], linewidth=3.0, label = '$S_{Lav}(w)$')
pl.plot(1e7/freq[ok1], shape_mea[ok1], color = sbm.findcol(6,col1)[0], linewidth=3.0, label = '$S_{new}(w)$')
pl.fill_between(1e7/freq[ok1], shape_mea[ok1]+err_mea[ok1], shape_mea[ok1]-err_mea[ok1], facecolor=sbm.findcol(12,col2)[0], alpha=alp)

pl.legend(loc=2)
#pl.show()
nomefile = cart + 'NewSHAPE_zoom_'+tag+'.eps'
fig.savefig(nomefile, format='eps', dpi=150)
pl.close()