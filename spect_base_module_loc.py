#!/usr/bin/python
# -*- coding: utf-8 -*-

import decimal
import numpy as np
import sys
import os.path
import matplotlib.pyplot as pl
import math as m
from numpy import linalg as LA

#parameters
Rtit = 2575.0 # km
Mtit = 1.3452e23 # kg
kb = 1.38065e-19 # boltzmann constant to use with P in hPa, n in cm-3, T in K (10^-4 * Kb standard)
c_R = 8.31446 # J K-1 mol-1
c_G = 6.67408e-11 # m3 kg-1 s-2


#################### DEFINE classes! ###############

class Pixel(object):
    """
    Each instance is a pixel, with all useful things of a pixel. Simplified version with few attributes.
    """

    def __init__(self, cube, year, dist, lat, alt, sza, phang, wl, spe, bbl):
        self.cube = cube
        self.year = year
        self.dist = dist
        self.lat = lat
        self.alt = alt
        self.sza = sza
        self.phang = phang
        self.wl = np.array(wl)
        self.spe = np.array(spe)
        self.bbl = np.array(bbl)


    def plot(self, range=None, mask=True, show=True, nomefile=None):
        """
        Plots spectrum in the required range. If mask=True does not show masked values.
        :param range: Wl range in which to plot
        :return:
        """
        if range is not None:
            ok = (self.wl >= range[0]) & (self.wl <= range[1]) & (self.bbl == 1)
        else:
            ok = (self.bbl == 1)

        fig = pl.figure(figsize=(8, 6), dpi=150)
        pl.plot(self.wl[ok],self.spe[ok])
        pl.grid()
        pl.xlabel('Wavelength (nm)')
        pl.ylabel('Radiance (W/m^2/nm/sr)')
        if show: pl.show()
        if nomefile is not None:
            fig.savefig(nomefile, format='eps', dpi=150)
            pl.close()

        return

    def integr(self, range=None):
        """
        Integrates the spectrum in the selected range. If no range the full spectrum is integrated.
        """

        if range is None:
            range = [np.min(self.wl),np.max(self.wl)]

        cond = (self.wl >= range[0]) & (self.wl <= range[1])
        intt = np.trapz(self.spe[cond],x=self.wl[cond])

        return intt



class PixelSet(np.ndarray):
    """
    A set of pixels. Takes as input an existing array of pixels and adds as attributes the vectorized attributes of Pixel.
    No new methods for now.
    """

    def __new__(cls, input_array, descr=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.descr = descr
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.descr = getattr(obj, 'descr', None)
        for name in obj[0].__dict__.keys():
            setattr(self,name,np.array([getattr(pix,name) for pix in obj]))

    def read_res(self, cart=''):
        """
        Leggi i risultati da cart e appiccicali ai pixel. Crea nuovi attributi anche per PixelSet.
        :param cart:
        :return:
        """
        fit_results = ['h3p_col','h3p_temp','err_col','err_temp','ch4_col','err_ch4','chisq','offset','wl_shift']
        for attr in fit_results:
            print(attr)
            #leggi
            for pix in self.pixels:
                pix.add_res(attr,res)
            #appiccica ai pixel
            #vettorizza
        print('da scrivere')
        return

    def ciao(self):
        print('ciao')
        return

    def ortomap(self, attr):
        print('da scrivere')
        attr_to_plot = getattr(self,attr)
        jirfu.ortomap(self.lat,self.lon,attr_to_plot)
        return

    # def __init__(self, descr='', pixels=None):
    #     self.descr = descr
    #     if pixels is not None:
    #         self.pixels = pixels
    #         for name in pixels[0].__dict__.keys():
    #             setattr(self,name,np.array([getattr(pix,name) for pix in pixels]))
            # self.cube = np.array([pix.cube for pix in pixels])
            # self.year = year
            # self.dist = dist
            # self.lat = lat
            # self.alt = alt
            # self.sza = sza
            # self.phang = phang
            # self.wl = np.array(wl)
            # self.spe = np.array(spe)
            # self.bbl = np.array(bbl)

    # def _wrapped_method(nome):
    #     return
    #
    # def _wrapped_attr(self, attr):
    #     return
    #
    # wrapped_attr = ('cube', 'dist', 'sza', 'spe', 'bbl', 'wl', 'year', 'lat', 'alt', 'phang')
    # for attr in wrapped_attr:
    #     setattr(Set,attr,_wrapped_attr(attr))



#### Funzioni

def trova_spip(file, hasha = '#'):
    """
    Trova il '#' nei file .dat
    """
    gigi = 'a'
    while gigi != hasha :
        linea = file.readline()
        gigi = linea[0]
    else:
        return

def read_obs(filename):
    """
    Reads files of VIMS observations. (gbb_2015 format)
    :param filename:
    :return:
    """
    infile = open(filename, 'r')
    trova_spip(infile)
    line = infile.readline()
    cosi = line.split()
    n_freq = int(cosi[0])
    n_limb = int(cosi[1])
    trova_spip(infile)
    dists = []
    while len(dists) < n_limb:
        line = infile.readline()
        dists += list(map(float, line.split()))
    dists = np.array(dists)
    trova_spip(infile)
    alts = []
    while len(alts) < n_limb:
        line = infile.readline()
        alts += list(map(float, line.split()))
    alts = np.array(alts)
    trova_spip(infile)
    data = [line.split() for line in infile]
    data_arr = np.array(data)
    freq = [float(r) for r in data_arr[:, 0]]
    obs = data_arr[:, 1:2*n_limb+2:2]
    obs = obs.astype(np.float)
    flags = data_arr[:, 2:2*n_limb+2:2]
    flags = flags.astype(np.int)
    infile.close()
    return n_freq, n_limb, dists, alts, freq, obs, flags


def writevec(file,vec,n_per_line,format_str):
    """
    Writes a vector in formatted output to a file.
    :param file: File to write to.
    :param vec: Vector to be written
    :param n_per_line: Number of elements of vector per line
    :param format_str: String format of each number written
    :return: nada
    """
    n = len(vec)
    com = n/n_per_line
    for i in range(com):
        i1 = i*n_per_line
        i2 = i1 + n_per_line
        strin = n_per_line*format_str+'\n'
        file.write(strin.format(*vec[i1:i2]))
    nres = n - com * n_per_line
    i1 = com * n_per_line
    if(nres > 0):
        strin = nres*format_str+'\n'
        file.write(strin.format(*vec[i1:n]))

    return


def write_obs(n_freq, n_limb, dists, alts, freq, obs, flags, filename, old_file = 'None'):
    """
    Writes files of VIMS observations. (gbb_2015 format)
    :param filename:
    :return:
    """
    from datetime import datetime

    infile = open(filename, 'w')
    data = datetime.now()
    infile.write('Modified on: {}\n'.format(data))
    infile.write('Original file: {}\n'.format(old_file))
    infile.write('\n')
    infile.write('Number of spectral points, number of tangent altitudes:\n')
    infile.write('{:1s}\n'.format('#'))
    infile.write('{:12d}{:12d}\n'.format(n_freq,n_limb))
    infile.write('\n')
    infile.write('Altitudes of satellite (km):\n')
    infile.write('{:1s}\n'.format('#'))
    writevec(infile,dists,8,'{:15.4e}')
    #infile.write((8*'{:15.4e}'+'\n').format(dists))
    infile.write('\n')
    infile.write('Tangent altitudes (km): \n')
    infile.write('{:1s}\n'.format('#'))
    writevec(infile,alts,8,'{:10.2f}')
    #infile.write((8*'{:10.2f}'+'\n').format(alts))
    infile.write('\n')
    infile.write('Wavelength (nm), spectral data (W m^-2 nm^-1 sr^-1):\n')
    infile.write('{:1s}\n'.format('#'))

    for fr, ob, fl in zip(freq, obs, flags):
        str = '{:10.4f}'.format(fr)
        for oo, ff in zip(ob,fl):
            str = str + '{:15.4e}{:3d}'.format(oo,ff)
        str = str + '\n'
        infile.write(str)

    infile.close()
    return


def read_input_prof_gbb(filename, type, n_alt = 151, alt_step = 10.0, n_gas = 86, n_lat = 4):
    """
    Reads input profiles from gbb standard formatted files (in_temp.dat, in_pres.dat, in_vmr_prof.dat).
    Profile order is from surface to TOA.
    type = 'vmr', 'temp', 'pres'
    :return: profiles
    """
    alts = np.linspace(0,(n_alt-1)*alt_step,n_alt)

    infile = open(filename, 'r')

    if(type == 'vmr'):
        print(type)
        trova_spip(infile)
        trova_spip(infile)
        first = 1
        for i in range(n_gas):
            lin = infile.readline()
            #print(lin)
            num = lin.split()[0]
            nome = lin.split()[1]
            prof = []
            while len(prof) < n_alt:
                line = infile.readline()
                prof += list(map(float, line.split()))
            prof = np.array(prof[::-1])
            if(first):
                proftot = prof
                first = 0
            else:
                proftot = np.vstack([proftot,prof])
            for j in range(n_lat-1): # to skip other latitudes
                prof = []
                while len(prof) < n_alt:
                    line = infile.readline()
                    prof += list(map(float, line.split()))
        proftot = np.array(proftot)


    if(type == 'temp' or type == 'pres'):
        print(type)
        trova_spip(infile)
        trova_spip(infile)
        prof = []
        while len(prof) < n_alt:
            line = infile.readline()
            prof += list(map(float, line.split()))
        proftot = np.array(prof[::-1])

    return proftot


def write_input_prof_gbb(filename, prof, type, n_alt = 151, alt_step = 10.0):
    """
    Writes input profiles in gbb standard formatted files (in_temp.dat, in_pres.dat, in_vmr_prof.dat)
    :return:
    """
    alts = np.linspace(0,(n_alt-1)*alt_step,n_alt)

    infile = open(filename, 'w')
    n_per_line = 8

    if(type == 'vmr'):
        strin = '{:10.3e}'

    if(type == 'temp'):
        strin = '{:10.5f}'

    if(type == 'pres'):
        strin = '{:11.4e}'

    infile.write('{:1s}\n'.format('#'))
    writevec(infile,prof[::-1],n_per_line,strin)

    return


def read_input_atm_man(filename):
    """
    Reads input atmosphere in manuel standard.
    :param filename:
    :return:
    """
    infile = open(filename,'r')
    trova_spip(infile,hasha='$')
    n_alt = int(infile.readline())
    trova_spip(infile,hasha='$')
    prof = []
    while len(prof) < n_alt:
        line = infile.readline()
        prof += list(map(float, line.split()))
    alts = np.array(prof)

    trova_spip(infile,hasha='$')
    prof = []
    while len(prof) < n_alt:
        line = infile.readline()
        prof += list(map(float, line.split()))
    pres = np.array(prof)

    trova_spip(infile,hasha='$')
    prof = []
    while len(prof) < n_alt:
        line = infile.readline()
        prof += list(map(float, line.split()))
    temp = np.array(prof)

    return alts, temp, pres


def write_input_atm_man(filename, z, T, P, n_alt = 301, alt_step = 5.0):
    """
    Writes input profiles in manuel standard formatted files
    :return:
    """
    alts = np.linspace(0,(n_alt-1)*alt_step,n_alt)

    infile = open(filename, 'w')
    n_per_line = 5


    str1 = '{:8.1f}'
    str2 = '{:11.4e}'
    str3 = '{:9.3f}'

    infile.write('# Atmosphere with wavy prof, reference Atm 05 S, 2006/07\n')
    infile.write('\n')
    infile.write('Number of levels\n')
    infile.write('{:1s}\n'.format('$'))
    infile.write('{}\n'.format(n_alt))
    infile.write('\n')
    infile.write('Altitudes [km]\n')
    infile.write('{:1s}\n'.format('$'))
    writevec(infile,z,n_per_line,str1)
    infile.write('\n')
    infile.write('Pressure [hPa]\n')
    infile.write('{:1s}\n'.format('$'))
    writevec(infile,P,n_per_line,str2)
    infile.write('\n')
    infile.write('Temperature [K]\n')
    infile.write('{:1s}\n'.format('$'))
    writevec(infile,T,n_per_line,str3)
    infile.close()

    return


def read_input_prof_lin(filename, n_col, n_alt = 151, alt_step = 10.0):
    """
    Reads input profiles from
    :return:
    """


def write_input_prof_lin():
    """
    Reads input profiles from gbb
    :return:
    """


def read_sim_gbb(filename,skip_first = 0, skip_last = 0):
    """
    Read sim_*.dat or spet_*.dat files in gbb format.
    :return:
    """
    infile = open(filename, 'r')
    line = infile.readline()
    line = infile.readline()
    alt = line.split()[0]
    trova_spip(infile)

    for i in range(skip_first):
        line = infile.readline()

    data = [line.split() for line in infile]
    data_arr = np.array(data)
    freq = np.array([float(r) for r in data_arr[:, 1]])
    obs = np.array([float(r) for r in data_arr[:, 2]])
    sim = np.array([float(r) for r in data_arr[:, 3]])
    err = np.array([float(r) for r in data_arr[:, 4]])
    #flags = data_arr[:, 2:2*n_limb+2:2]
    #flags = flags.astype(np.int)
    infile.close()

    if skip_last == 0:
        return alt,freq,obs,sim,err
    else:
        return alt,freq[:-skip_last],obs[:-skip_last],sim[:-skip_last],err[:-skip_last]


def HG_phase_funct(deg,g):
    """
    Henyey-Greenstein phase function.
    :param deg: angle in radiants
    :param g: asymmetry factor (from 0 to 1)
    :return:
    """
    phunc=(1.0/(4.0*m.pi))*(1.0-g**2)/(1.0+g**2+2.0*g*m.cos(deg))**1.5

    return phunc



def read_rannou_aer(filename):
    """
    Reads the aerosol properties (ext. coeff., ssa and the 256 coefficients of Legendre polinomials for the phase function)
    :return: freq (nm), extcoeff (cm-1), ssa, matrix(n_freq,n_poli)
    """
    infile = open(filename, 'r')
    data = [line.split() for line in infile]
    data_arr = np.array(data)
    freq = np.array([float(r) for r in data_arr[:, 0]])
    extco = np.array([float(r) for r in data_arr[:, 1]])
    ssa = np.array([float(r) for r in data_arr[:, 2]])
    leg_coeff = data_arr[:, 3:]
    leg_coeff = leg_coeff.astype(np.float)
    infile.close()

    return freq,extco,ssa,leg_coeff


def gaussian(x, mu, fwhm):
    pi = m.acos(-1)
    sig = fwhm / (2*m.sqrt(2*m.log(2.0)))
    fac = m.sqrt(2*pi)*sig
    return np.exp(-(x - mu)**2 / (2 * sig**2)) / fac


def freqTOwl(freq,spe_freq,wl,fwhm):
    """
    Converts a HIRES spectrum in nW/(cm2*cm-1) into a LOWRES spectrum in W/(m2*nm)
    :param freq: HIRES freq (cm-1)
    :param spe_freq: HIRES spectrum ( nW/(cm2*cm-1) )
    :param wl: LOWRES wl grid (nm)
    :param fwhm: FWMH of the ILS (nm)
    :return: spe : LOWRES spectrum in W/(m2*nm)
    """

    spe_freq = 10**(-5)*freq**2/10**7 * spe_freq
    freq = 10**7/freq

    freq = freq[::-1] # reordering freq
    spe_freq = spe_freq[::-1]

    spe = []
    for w, fw in zip(wl, fwhm):
        gauss = gaussian(freq, w, fw)
        convol = np.trapz(gauss*spe_freq, x=freq)
        spe.append(float(convol))

    spe = np.array(spe)

    return spe


def read_bands(filename):
    """
    Reads bands and ILS fwhm of VIMS observations. (gbb_2015 format)
    :param filename:
    :return:
    """
    infile = open(filename, 'r')
    trova_spip(infile)
    data = [line.split() for line in infile]
    data_arr = np.array(data)
    wl = [float(r) for r in data_arr[:, 0]]
    fwhm = [float(r) for r in data_arr[:, 2]]
    infile.close()

    return wl, fwhm


def findcol(n,i, color_map = 5):
    """
    Gives the best COLOR choice for line i in a n-lines plot.
    :param n: total number of lines
    :param i: line considered
    :param color_map: OPTIONAL, name of the color map.
    :return: RGBA tuple like (0.1,0.4,0.3,1.0)
    """
    import matplotlib.cm as cm

    cmaps = ['spectral','jet','gist_ncar','gist_rainbow','hsv','nipy_spectral']

    if n < 5:
        oss = 0.2
        fa = 0.6
    elif n > 5 and n < 11:
        oss = 0.1
        fa = 0.8
    else:
        oss = 0.0
        fa = 1.0

    cmap = cm.get_cmap(cmaps[color_map])
    colo = oss+fa*i/(n-1)

    # setting linestyle
    lis = ['--','-','-.','-']#,':']
    oi = i % 4

    return cmap(colo), lis[oi]


def plotta_sim_VIMS(nomefile,freq,obs,sim,sims,names,err=1.5e-8,title='Plot', auto = True,
                    xscale=[-1,-1],yscale=[-1,-1],yscale_res=[-1,-1]):
    """
    Plots observed/simulated with residuals and single contributions.
    :param obs: Observed
    :param sim: Simulated total
    :param n_sims: Number of simulated mol. contributions
    :param sims: matrix with one sim per row
    :param names: names of each sim
    :return:
    """
    from matplotlib.font_manager import FontProperties
    import matplotlib.gridspec as gridspec

    fontP = FontProperties()
    fontP.set_size('small')
#    legend([plot1], "title", )
    n_sims = sims.shape[0]
    fig = pl.figure(figsize=(8, 6), dpi=150)

    gs = gridspec.GridSpec(2, 1,height_ratios=[2,1])
    ax1 = pl.subplot(gs[0])
    pl.ylabel('Radiance (W m$^{-2}$ nm$^{-1}$ sr$^{-1}$)')
    pl.title(title)
    ax2 = pl.subplot(gs[1])
    if not auto:
        ax1.set_xlim(xscale)
        ax1.set_ylim(yscale)
        ax2.set_xlim(xscale)
        ax2.set_ylim(yscale_res)

    pl.xlabel('Wavelength (nm)')

#    pl.subplot(211)
    colo, li = findcol(n_sims+2,-1)
    ax1.plot(freq,obs,color=colo,label='Data',linewidth=1.0)
    ax1.scatter(freq,obs,color=colo,linewidth=1.0)
    ax1.errorbar(freq,obs,color=colo,yerr=err, linewidth=1.0)
    colo, li = findcol(n_sims+2,0)
    ax1.plot(freq,sim,color=colo,linewidth=3.0,label='Sim')
    i=1
    for name,simu in zip(names,sims):
        colo, li = findcol(n_sims+2,i)
        ax1.plot(freq,simu,color=colo,linestyle=li,label=name,linewidth=2.0)
        i +=1
    ax1.grid()
    ax1.legend(loc=1,bbox_to_anchor=(1.05,1.1),fontsize='small',fancybox=1,shadow=1)

#    pl.subplot(212)
    ax2.grid()
    ax2.plot(freq,obs-sim,color='red',linewidth=3.0)
#    ax2.fill_between(freq,err*np.ones(len(freq)),-err*np.ones(len(freq)), facecolor=findcol(12,8)[0], alpha=0.1)
    ax2.plot(freq,err*np.ones(len(freq)),color='black',linestyle='--',linewidth=2.0)
    ax2.plot(freq,-err*np.ones(len(freq)),color='black',linestyle='--',linewidth=2.0)

    fig.savefig(nomefile, format='eps', dpi=150)
    pl.close(fig)

    return

#################################################################################
##                                                                            ###
##                                                                            ###
##       GEOMETRICAL FUNCTIONS relating with orbit, atmospheric grid, ...    ####
##                                                                            ###
##                                                                            ###
#################################################################################



def szafromsspDEG(lat, lon, lat_ss, lon_ss):
    """
    Returns sza at certain (lat, lon) given (lat_ss, lon_ss) of the sub_solar_point.
    All angles in DEG.
    """
    lat = lat * np.pi / 180.0
    lon = lon * np.pi / 180.0
    lat_ss = lat_ss *np.pi/180.0
    lon_ss = lon_ss *np.pi/180.0
    sc_prod = m.cos(lat) * m.cos(lat_ss) * m.cos(lon) * m.cos(lon_ss) \
              + m.cos(lat) * m.cos(lat_ss) * m.sin(lon) * m.sin(lon_ss) \
              + m.sin(lat) * m.sin(lat_ss)
    sza = m.acos(sc_prod) * 180.0 / np.pi
    return sza


def szafromssp(lat, lon, lat_ss, lon_ss):
    """
    Returns sza at certain (lat, lon) given (lat_ss, lon_ss) of the sub_solar_point.
    All angles in radians.
    """
    sc_prod = m.cos(lat) * m.cos(lat_ss) * m.cos(lon) * m.cos(lon_ss) \
              + m.cos(lat) * m.cos(lat_ss) * m.sin(lon) * m.sin(lon_ss) \
              + m.sin(lat) * m.sin(lat_ss)
    sza = m.acos(sc_prod)
    return sza


def rad(ang):
    pi = 2 * acos(0.0)
    return ang*pi/180.0


def deg(ang):
    pi = 2 * acos(0.0)
    return ang*180.0/pi


def sphtocart(lat, lon, h=0., R=Rtit):
    """
    Converts from (lat, lon, alt) to (x, y, z).
    Convention: lon goes from 0 to 360, starting from x axis, towards East. lat is the latitude.
    h is the altitude with respect to the spherical surface. R_p is the planet radius.
    :return: 3D numpy array
    """
    r = [m.cos(lat)*m.cos(lon), m.cos(lat)*m.sin(lon), m.sin(lat)]
    r = np.array(r)
    r *= (R + h)
    return r


def LOS_2D(alt_tg,alts,T,P,gas_ok,ext_coef,Rpl=2575.0):
    """
    Calculates LOS path in the atm for given tangent altitude. Homogeneous atmosphere. No refraction.
    :param alt_tg: tangent altitude of the LOS (km)
    :param alts: altitude grid (km)
    :param T: temp. profile (K)
    :param P: pres. profile (hPa)
    :param ext_coef: Aerosol ext. coeff. in cm-1
    :param gas_ok: VMR profiles of selected gases (matrix n_gases x n_alts)
    :param Rpl: planetary radius
    :return:
    """

    n = P/(kb*T) # num. density in cm-3

    Rtoa = np.max(alts)
    step = 10.0 # step in km
    _LOS = np.array([1,0])
    R_0 = np.array([-(m.sqrt((Rtoa+Rpl)**2-(Rpl+alt_tg)**2)),Rpl+alt_tg]) # first LOS point

    R = R_0 + step * _LOS #first step
    #print(R_0)
    #print(R)
    #print(LA.norm(R)-Rpl)
    z_los = np.array([Rtoa,LA.norm(R)-Rpl])
    R_los = np.vstack([R_0,R])
    steps = np.array([step])

    while LA.norm(R) < Rtoa+Rpl:
        R = R + step * _LOS
        R_los = np.vstack([R_los,R])
        z_los = np.append(z_los,LA.norm(R)-Rpl)
        #print(R)
        #print(LA.norm(R)-Rpl)
        steps = np.append(steps,step)

    Tau_aer = np.interp(z_los[:-1], alts, ext_coef)
    Tau_aer = Tau_aer * steps * 1e5
    temps = np.interp(z_los[:-1], alts, T)
    nlos = np.interp(z_los[:-1], alts, n)
    press = np.interp(z_los[:-1], alts, np.log(P))
    press = np.exp(press)
    gases = np.zeros(len(z_los[:-1]))
    Rcols = np.zeros(len(z_los[:-1]))
    for gas in gas_ok:
        VMRs = np.interp(z_los[:-1], alts, gas)
        Rcol = VMRs * nlos * steps * 1e5
        gases = np.vstack([gases,VMRs])
        Rcols = np.vstack([Rcols,Rcol])
    gases = gases[1:,:]
    Rcols = Rcols[1:,:]

    return z_los[:-1],steps,temps,press,gases,Rcols,Tau_aer


def hydro_P(z,T,MM,P_0=None,R=Rtit,M=Mtit):
    """
    Calculates hydrostatic pressure in hPa, given temp. profile.
    :param z: altitude (km)
    :param T: temperature (K)
    :param MM: mean molecular mass (amu)
    :param P_0: Surface pressure (hPa)
    :param R: Planetary radius (km)
    :param M: Planetary mass (kg)
    :return: P, pressure profile on z grid
    """
    P_huy = 1.4612e3 #hPa

    if P_0 is None:  # define P_0 as the huygens pressure
        P_0 = P_huy
    reverse = False
    if z[1] < z[0]: # reverse order if z decreasing
        reverse = True
        z = z[::-1]
        T = T[::-1]
        MM = MM[::-1]
    if np.size(MM) == 1: # if MM is a scalar, then multiply for a np.ones vector
        mu = MM
        MM = mu*np.ones(len(z))

    R = R*1e3 # from km to m
    z = z*1e3
    MM = MM*1e-3 # from amu to kg/mol

    g = c_G*M/(R+z)**2
    print(g[0])
    #g=g*1.352/g[0]

    HH = MM*g/(c_R*T)

    P = np.zeros(len(z))
    P2 = P
    P[0] = P_0
    for i in range(1,len(z)):
        # dz = z[i]-z[i-1]
        # int = HH[i-1]*dz+0.5*dz*(HH[i]-HH[i-1])  # integrazione stupida
        # P[i] = P[i-1]*np.exp(-int)
        int = np.trapz(HH[0:i+1],x=z[0:i+1])
        P2[i] = P[0]*np.exp(-int)

    if reverse:
        P2 = P2[::-1]

    return P2


def findT(alt,alt_atm,temp,diff=1.0):
    """
    Finds T at altitude alt.
    :return:
    """
    T = 0
    for tt,altu in zip(temp,alt_atm):
        if(abs(alt-altu)<diff):
            T = tt

    return T


def find_near(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


def find_incl(array,value1,value2):
    ok = np.all([value2 >= array, array >= value1], axis=0)
    return ok

#################################################################################
##                                                                            ###
##                                                                            ###
##       OTHER FUNCTIONS                                                     ####
##                                                                            ###
##                                                                            ###
#################################################################################


def BB(T,w):
    """
    Black body at temp. T in units of nW/(cm2*cm-1).
    :param T: Temperature
    :param w: Wavenumber (cm-1)
    :return:
    """
    rc1 = 1.1904e-3
    rhck = 1.4388

    BB = rc1 * w**3 / (m.exp(w*rhck/T)-1)

    return BB


def BB_nm(T,w):
    """
    Black body at temp. T in units of W/(m2*nm).
    :param T: Temperature
    :param w: Wavelength (nm)
    :return:
    """
    rc1 = 1.1904
    rhck = 1.4388

    if T*w > 5e5 :
        BB = rc1 * m.pow((1.e4/w),5) / (m.exp(1.e7/w*rhck/T)-1)
    else:
        BB = rc1 * m.pow((1.e4/w),5) * m.exp(-1.e7/w*rhck/T)

    return BB