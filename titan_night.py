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

alt_min = 100.0
alt_max = 300.0

cubo = '/home/fede/Scrivania/Dotto/AbstrArt/Titan_workshop/Data/V1530534304.sav'

cubino = io.readsav(cubo)
print(len(cubino))
print(type(cubino))

# pl.imshow(cubino.data[:,240,:])
# pl.show()
# sys.exit()


# ww = np.linspace(10,5000,10000)
#
# BBB = np.vectorize(sbm.BB)
# tii = 175.0
# bobo = BBB(tii,ww)
# into = np.trapz(bobo,x=ww)
# print(into)
# print(mt.pi*into/tii**4)
# #pl.plot(ww,bobo)
# pl.plot(ww,0.01*bobo/100.0)
# #for tt in tau_alt:
# #    pl.plot(ww,bobo*tt)
# pl.show()
# sys.exit()

cart = '/home/fede/Scrivania/Dotto/AbstrArt/Titan_workshop/titan_night/'

n_alt = 151
alt_step = 10.0

alts = np.linspace(0,(n_alt-1)*alt_step,n_alt)

# Read input for the atmosphere

temp = sbm.read_input_prof_gbb(cart + 'in_temp.dat', 'temp')
pres = sbm.read_input_prof_gbb(cart + 'in_pres.dat', 'pres')
gases = sbm.read_input_prof_gbb(cart + 'in_vmr_prof.dat', 'vmr')


# Leggo Rannou
first = 1
for alt in alts[:70]:
    altnom = str(int(alt))
    freq, extco, ssa, leg_coeff = sbm.read_rannou_aer(cart + '../Rannou_aerosol/aerosol_'+altnom+'Km_mod1.p256')
    if first:
        ext1 = np.array([extco[0]])
        extcotot = extco/extco[0]
        ssatot = ssa
        leg_coefftot = leg_coeff
        first = 0
        continue
    ext1 = np.append(ext1,extco[0])
    extcotot = np.vstack([extcotot,extco/extco[0]])
    ssatot = np.vstack([ssatot,ssa])
    leg_coefftot = np.vstack([leg_coefftot,leg_coeff])


#ext1 = ext1 *1e-8 # 01R
ext1 = ext1 *0.5e-8 # 02R
ext_Rannou = np.append(ext1,np.zeros(len(alts)-len(ext1)))

#tag = '02R'
freq2, ext200, ssa200, leg_coeff = sbm.read_rannou_aer(cart + '../Rannou_aerosol/aerosol_200Km_mod1.p256')
oio = sbm.find_near(freq2, 2000.0)
ext200 = ext200/ext200[oio]

# Flat case
ext_flat = np.zeros(len(ext200))+1.0
ssa_flat = np.zeros(len(ext200))+0.5


#Leggo Lavvas
infile = open(cart + 'Lavvas.dat','r')
lin = infile.readline()
n_wl = int(lin.split()[1])
n_lev = int(lin.split()[0])
lin = infile.readline()
wls = np.array([float(r) for r in lin.split()])
ext_Lavvas = np.zeros(n_wl)
ssa_Lavvas = np.zeros(n_wl)
alt_Lavvas = []
for lev in range(n_lev):
    lin = infile.readline()
    altl = float(lin.split()[1])
    extl = np.array([float(r) for r in lin.split()[3:]])
    lin = infile.readline()
    ssal = np.array([float(r) for r in lin.split()])
    lin = infile.readline()
    alt_Lavvas = np.append(alt_Lavvas,altl)
    ext_Lavvas = np.vstack([ext_Lavvas,extl])
    ssa_Lavvas = np.vstack([ssa_Lavvas,ssal])
infile.close()

ext_Lavvas = ext_Lavvas[1:]
ssa_Lavvas = ssa_Lavvas[1:]



#alt_fin = alt_sint[np.all([max(alt_obs) > alt_sint, alt_sint > min(alt_obs)], axis=0)]
###################################### ext_Lavvas_ok = ext_Lavvas

alts_ok = np.linspace(100,300,5)

fig = pl.figure(figsize=(8, 6), dpi=150)
filename = 'Lavvas_thol.eps'
pl.xlim(4000.,5200.)
pl.grid()
pl.title('Aerosol extinction coeff. x (1-ssa) after Lavvas (2010)')
pl.xlabel('Wavelength (nm)')
pl.ylabel('Absorption coeff. (cm-1)')
#pl.yscale('log')
i=0
for alt in alts_ok:
    il = sbm.find_near(alt_Lavvas,alt)
    print(alt)
    ok = sbm.find_incl(wls,4000.,5200.)
    col, li = sbm.findcol(len(alts_ok),i)
    #pl.plot(wls[ok],ext_Lavvas[il,ok],linewidth=2.0,color=col,label=str(int(alt))+' km')
    pl.plot(wls[ok],ext_Lavvas[il,ok]*(1-ssa_Lavvas[il,ok]),linewidth=2.0,color=col,label=str(int(alt))+' km')
    i+=1
#pl.plot(wls)
pl.legend(loc=1)
#pl.show()
fig.savefig(cart+filename, format='eps', dpi=150)
pl.close(fig)

ok5000 = sbm.find_near(wls,5000.0)
ext_Lavvas_prof = np.interp(alts,alt_Lavvas,ext_Lavvas[:,ok5000])

ok200 = sbm.find_near(alt_Lavvas,200.)
ext200_Lavvas = ext_Lavvas[ok200,ok]/ext_Lavvas[ok200,ok5000]
ssa200_Lavvas = ssa_Lavvas[ok200,ok]
wl_Lav = wls[ok]

pickle.dump([wls[ok],ext200_Lavvas,ssa200_Lavvas],open('lavvas.pic','w'))

# ora trasformo tutto in cm-1

# pl.xlim(2000.0,2500.0)
# pl.ylim(0,2)
# #pl.plot(freq2,1-ssa200)
# pl.plot(freq2,ext200*(1-ssa200))
# pl.plot(1e7/wls[ok],ext200_Lavvas*(1-ssa200_Lavvas))
# pl.show()
# sys.exit()

# pl.yscale('log')
# pl.scatter(alts,ext_Lavvas_prof)
# pl.show()

#sys.exit()
#sys.exit()

# file_P = open(cart + 'shape.dat','r')
# data = [line.split() for line in infile]
# data_arr = np.array(data)
# frfr = [float(r) for r in data_arr[:, 0]]
# ext_P = [float(r) for r in data_arr[:, 2]]

#ext200 = ext_flat # Serie Flat
#ssa200 = ssa_flat

# serie with 4.5 mu peak

# fig = pl.figure(figsize=(8, 6), dpi=150)
# filename = 'Aerprof_'+tag+'.eps'
# pl.xscale('log')
# pl.xlim(1e-12,1e-7)
# pl.ylim(0,600.)
# pl.plot(ext_Rannou,alts)
# pl.plot(0.5*gases[81,:],alts)
# fig.savefig(cart+filename, format='eps', dpi=150)
# pl.close(fig)
#
# fig = pl.figure(figsize=(8, 6), dpi=150)
# filename = 'Tholin_'+tag+'.eps'
# pl.xlim(1950.,2600.)
# pl.ylim(0.,3.)
# pl.plot(freq,ext200)
# pl.plot(freq,ext200*(1-ssa200))
# # for spe in extcotot[:1]:
# #     pl.plot(freq,spe)
# fig.savefig(cart+filename, format='eps', dpi=150)
# pl.close(fig)

# pl.plot(freq,extco)
# for i in range(255):
#     pl.plot(freq,leg_coeff[:,i])
# pl.show()

serie = np.append([1.],leg_coeff[1,:])
print(serie)
serie_10 = np.append([1],leg_coeff[1,:20])
phase = np.polynomial.legendre.Legendre(serie)
phase_10 = np.polynomial.legendre.Legendre(serie_10)

fac=0.25/mt.pi
theta = np.linspace(0,mt.pi,100)
pl.plot(theta,fac*phase(np.cos(theta)))
pl.plot(theta,fac*phase_10(np.cos(theta)))
HG = np.vectorize(sbm.HG_phase_funct)
pl.plot(theta,HG(mt.pi-theta,0.3))
# pl.plot(theta,10*(phase(np.cos(theta))-phase_10(np.cos(theta))))
pl.show()
sys.exit()

alt_los = np.array([50.0 + 25.0*i for i in range(17)])
fr_grid = np.linspace(1950.,2600.,100)
K_thol = np.interp(fr_grid,freq,ext200)
ssa_K_thol = np.interp(fr_grid,freq,ssa200)
oeoie = sbm.find_near(fr_grid,2000.0)
ssa2000 = ssa_K_thol[oeoie]
aer_emiss = np.zeros((len(alt_los),len(fr_grid)))
BBtau = np.zeros((len(alt_los),len(fr_grid)))


#fr_grid = fr_grid+60.0

#pl.plot(fr_grid,K_thol)#*(1-ssa_K_thol))
#pl.plot(1e7/wls[ok],ext200_Lavvas)#*(1-ssa200_Lavvas))
#pl.show()
#sys.exit()

###################################################################################################
###
###
###                         QUI SCEGLI LE COSE!!

ext_prof = ext_Lavvas_prof
tag = '01Lav'

ext_prof = ext_Lavvas_prof/10
tag = '01Lav_div10'


###
###
###
###
###
########################################################################################################



gas_ok = np.array([5,6,82])-1
i=0
tauu = open(cart + 'Tau_' + tag + '.dat', 'w')
tauu.write('Optical depth at different heights:\n')
tauu.write('{:1s}\n'.format('#'))
tau_alt = []

for alt_tg in alt_los:
    #z_los,steps,T,P,VMRs,Rcols = sbm.LOS_2D(alt_tg,alts,temp,pres,gases[gas_ok,:],Rpl=2575.0)
    z_los,steps,T,P,VMRs,Rcols,Tau_aer = sbm.LOS_2D(alt_tg,alts,temp,pres,gases[gas_ok,:],ext_prof)
    tauu.write('{:8.1f}  {:12.3e}\n'.format(alt_tg,sum(Tau_aer)))
    tautot=sum(Tau_aer)
    tau_alt = np.append(tau_alt,tautot)
    j=0
    for fr,kth,ssah in zip(fr_grid,K_thol,ssa_K_thol):
        k = 0
        Tran = 1
        Tran2 = 1
        for dl,ti in zip(Tau_aer,T):
            Tran *= (1-kth*Tau_aer[k])
            Tran2 *= (1-Tau_aer[k])
            bibo = sbm.BB(ti,fr)
            aer_emiss[i,j] += dl*bibo*kth*(1-ssah)*Tran #sum(Tau_aer*BBB(T,fr)*kth*ssah)
            BBtau[i,j] += dl*bibo*Tran2/tautot
            k+=1
        j+=1
    i+=1


# fig = pl.figure(figsize=(8, 6), dpi=150)
# filename = 'Opt_depth_tholin_'+tag+'.eps'
# pl.xlabel('Wavelength (nm)')
# pl.ylabel('Absorption optical depth')
# pl.title('Absorption optical depth at different limbs')
# pl.grid()
# pl.xlim(3800.,5200.)
#
# i=0
# for alt, tau in zip(alt_los, tau_alt):
#     color, li = sbm.findcol(13,i)
#     print(alt)
#     if(alt < 50 or alt > 300):
#         continue
#     i += 1
#     pl.plot(wls[ok],tau*ext200_Lavvas*(1-ssa200_Lavvas), color = color, label = str(int(alt))+' km')
#
# pl.legend(loc=2)
# fig.savefig(cart+filename, format='eps', dpi=150)
# pl.close(fig)


ioio = sbm.find_near(wl_Lav,5000.0)
fac = (1-ssa200_Lavvas[ioio])

gigi = sbm.find_incl(alt_los,50,350)

color, li = sbm.findcol(13,5)

fig = pl.figure(figsize=(8, 6), dpi=150)
filename = 'Lavvas_shape.eps'
pl.ylabel('$S_{Lav}(\lambda)$')
pl.xlabel('Wavelength (nm)')
pl.title('$S_{Lav}(\lambda)$')
pl.grid()
pl.xlim(4000.0,5200.0)
pl.plot(wl_Lav, ext200_Lavvas*(1-ssa200_Lavvas)/(1-ssa200_Lavvas[ioio]), color = color, linewidth = 3.0)
#pl.legend(loc=2)
fig.savefig(cart+filename, format='eps', dpi=150)
pl.close(fig)


color, li = sbm.findcol(13,10)
fig = pl.figure(figsize=(8, 6), dpi=150)
filename = 'Tau_ref_'+tag+'.eps'
pl.ylabel('Altitude (km)')
pl.xlabel('Absorption optical depth at 5$\mu m$')
pl.title(r'$\tau_{Lav}(5\mu m, z)$')
pl.grid()
pl.xscale('log')
pl.plot(fac*tau_alt[gigi],alt_los[gigi], color = color, linewidth = 3.0)
#pl.legend(loc=2)
fig.savefig(cart+filename, format='eps', dpi=150)
pl.close(fig)

fig = pl.figure(figsize=(8, 6), dpi=150)
filename = 'Tau_div10_'+tag+'.eps'
pl.ylabel('Altitude (km)')
pl.xlabel('Absorption optical depth at 5$\mu m$')
pl.title(r'$\tau_{Lav}(5\mu m, z)$')
pl.grid()
pl.xscale('log')

color, li = sbm.findcol(13,10)
pl.plot(fac*tau_alt[gigi],alt_los[gigi], color = color, linewidth = 3.0, label = '01Lav')
color, li = sbm.findcol(13,2)
pl.plot(fac*tau_alt[gigi]/10.0,alt_los[gigi], color = color, linewidth = 3.0, label = '01Lav_div10')

pl.legend(loc=1)
fig.savefig(cart+filename, format='eps', dpi=150)
pl.close(fig)


fig = pl.figure(figsize=(8, 6), dpi=150)
filename = 'Opt_depth_tholin_'+tag+'.eps'
pl.xlabel('Wavelength (nm)')
pl.ylabel('Absorption optical depth')
pl.title('Absorption optical depth at different limbs')
pl.grid()
pl.xlim(4100.,5100.)
pl.ylim(-0.1,0.4)

i=0
for alt, tau in zip(alt_los[::-1], tau_alt[::-1]):
    color, li = sbm.findcol(10,i)
    print(alt)
    if(alt < alt_min or alt > alt_max):
        continue
    i += 1
    pl.plot(wls[ok],tau*ext200_Lavvas*(1-ssa200_Lavvas), color = color, linewidth=2.0,label = str(int(alt))+' km')

pl.legend(loc=1,bbox_to_anchor=(1.05,1.05), fontsize='small',fancybox=1,shadow=1)
fig.savefig(cart+filename, format='eps', dpi=150)
pl.close(fig)

#sys.exit()

print(alt_los)
lu=4
ae = aer_emiss[lu,:]
bb = BBtau[lu,:]
#pl.plot(fr_grid,ae/bb/tau_alt[5])
#pl.plot(fr_grid,K_thol*(1-ssa_K_thol))

#pl.plot(1e7/wls[ok],ext200_Lavvas*(1-ssa200_Lavvas))
#pl.show()

#for alt,tau in zip(alt_los,tau_alt):
#   oki = sbm.find_near(alt_los,alt)
oki = lu
bibbi = sbm.freqTOwl(fr_grid,BBtau[oki,:],wls[ok],15.0*np.ones(len(wls[ok])))
    #opt = resid/BBB(ttt,freq)
    #opt = resid/bibbi
pl.plot(wls[ok],bibbi*ext200_Lavvas*(1-ssa200_Lavvas)*tau_alt[lu]/sbm.freqTOwl(fr_grid,ae,wls[ok],15.0*np.ones(len(wls[ok]))))
#pl.plot(wls[ok],sbm.freqTOwl(fr_grid,ae,wls[ok],15.0*np.ones(len(wls[ok]))))
#pl.plot(wls[ok],bibbi*ext200_Lavvas*(1-ssa200_Lavvas)*tau_alt[lu],label=str(alt_los[lu]))
pl.legend(loc=2)
#pl.show()
pl.close()

#sys.exit()

#pickle.dump([fr_grid,BBtau],open('BBtau.pic','w')) ### Commento perchè ha funzionato troppo bene!!

tauu.close()

pickle.dump([alt_los,tau_alt],open('tau_alt.pic','w'))

out = open(cart + 'Sim' + tag + '.dat', 'w')
out.write('Simulazione con profilo '+tag+' \n')
out.write('Tangent altitudes (km): \n')
out.write('{:1s}\n'.format('#'))
sbm.writevec(out,alt_los,len(alt_los),'{:8.1f}')
out.write(' \n')
out.write('Wn (cm-1) spe_alt (1:n_alts) ( nW / (cm2*cm-1) ) \n')
out.write('{:1s}\n'.format('#'))
for fr, sim in zip(fr_grid, aer_emiss.T):
    out.write('{:10.3f}'.format(fr))
    sbm.writevec(out,sim,len(sim),'{:12.4e}')
out.close()

wl, fwhm = sbm.read_bands(cart + 'band_titano.dat')

print(wl)
print(fwhm)


[alts,wal,resid_alt] = pickle.load(open('resid.pic','r'))
oioi = sbm.find_near(alts,200.)
uiui = sbm.find_near(alts,150.)

fig = pl.figure(figsize=(8, 6), dpi=150)
filename = 'Spesim_'+tag+'.eps'
pl.xlabel('Wavelength (nm)')
pl.ylabel('Radiance (W m$^{-2}$ nm$^{-1}$ sr$^{-1}$)')
#pl.title('Simulazione con profilo '+tag)
pl.grid()
pl.ylim(-1e-8,1e-7)
pl.xlim(4100.0,5100.0)

outspe = wl
i = 0
for alt, spe_sim in zip(alt_los, aer_emiss):
    spe = sbm.freqTOwl(fr_grid,spe_sim,wl,fwhm)
    outspe = np.vstack([outspe,spe])
    color, li = sbm.findcol(len(alt_los),i)
    i += 1
    if(alt < 100 or alt > 350):
        continue
    if(int(alt) % 50 == 0):
        pl.plot(wl[:-1],0.9*spe[:-1], color = color, label = str(int(alt))+' km', linewidth=2.0)
    #nome = 'Spesim_'+str(int(alt))+'_'+tag+'.eps'
    #sbm.plot(wl,spe,xtitle='Wl (nm)',ytitle='Rad ( W/(m2*nm*sr) )',cart+nome)
#pl.show()

pl.plot(wal,resid_alt[oioi,:],color = 'red', label = 'Res. 200 km', linewidth=2.0)
pl.plot(wal,resid_alt[uiui,:],color = 'brown', label = 'Res. 150 km', linewidth=2.0)

#pl.legend(loc=2)
pl.legend(loc=2, fontsize='small',fancybox=1,shadow=1)
fig.savefig(cart+filename, format='eps', dpi=150)
pl.close(fig)


out = open(cart + 'Sim_' + tag + '_VIMS.dat', 'w')
out.write('Simulazione con profilo '+tag+' \n')
out.write('Tangent altitudes (km): \n')
out.write('{:1s}\n'.format('#'))
sbm.writevec(out,alt_los,len(alt_los),'{:8.1f}')
out.write(' \n')
out.write('Wl (nm) spe_alt (1:n_alts) ( W / (m2*nm) ) \n')
out.write('{:1s}\n'.format('#'))
for spe in outspe.T:
    sbm.writevec(out,spe,len(spe),'{:12.4e}')
out.close()


# print(z_los)
# print(steps)
# print(T)
# print(P)
# print(VMRs)

#sbm.write_input_prof_gbb(cart+'Prof'+tag+'.dat', ext_Rannou, 'vmr')

print(Rcols[-1])