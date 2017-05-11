*  Created by  Marco Ridolfi    on   
*  Added some parameters by Piera Raspollini on 17.1.96
*                  "        Michael Hoepfner on 17.1.95 
*                  "        Massimo Carlotti on 29.1.96
*                  "        Michael Hoepfner on  5.2.96
*  Modified              by Piera Raspollini on 5.1.96
*
*  imxapo = maximum number of points in the apodisation function (path-
*           difference domain)
*  imxi   = maximum number of sampling points in the synthetic spectra      
*           computed at the observed frequencies
*  imxils = maximum number of sampling points in the instrument line-shape
*           function (fine-grid!)
*  imxilc = max number of sampling point in the instrument
*           line-shape function (coarse grid!)
*  imxj   = maximum dimension of J matrix (VCMob=J*Jt)
*           imxj=imxi+imxilc
*  imxsnc = max number of sampling point for the sinc function used to
*           interpolate the instrument line-shape function
*  imxite = maximum number of macro-iterations in retrieval procedure
*  imxpro = max number of elements in p, t profiles
*  imxlins = max number of line of sight used for the FOV convolution
*  imxlev = max number of levels used for modelling the atmosphere
*  imxsl  = max number of sub-levels between the pointings of the simulations
*  imxgas = max number of gas in the retrieval   
*  imxlay = max number of layers for modelling the atmosphere
*           (=imxlev-1)
*  imxmw  = max number of microwindows
*  imxlin = max number of lines per microwindow
*  imxsig = max number of wavenumber grid-points for a microwindow
*  imxiso = max number of total isotopes in the HITRAN database (=85)
*  imxcof = max number of coefficients for the calculation of the
*           quotient of the partition sum (=4) 
*  imxept = max number of extra paths
*  imxhit = max number of gases in the HITRAN data base (=36)
*  imxism = max number of isotopes in HITRAN data base per molecule (=8)
*  imxgmw = max number of gases per MW
*  imxlmb = max number of sweeps
*  imxobs = max number of observational point (for Jacobian matrix)

*  imxsi2 = max number of 1s in the irregular grids of the considered
*           microwindows
*  WARNING: if irregular grids for all the considered microwindows
*              are not available, put:
*  imxsi2 = imxsig
*  imxsig_c = max. n. of points in the fine grid for continuum
*  imxtit = max. n. of points in the fine grid spectrum for titan atmosphere
*  imxvims = max. n. of points in the coarse grid spectrum for titan atmosphere
*  imxparz = derivatives for a single LOS including multiplicities


      integer*4 imxnlte,imxband,imxvib
      integer*4 imxpro,imxlev,imxsl,imxlay
      integer*4 imxgas,imxgmw,imxmw,imxsig,imxsi2,imxtit,
     &          imxlin,imxiso,imxcof,imxvims,imxggg
      integer*4 imxept,imxhit,imxi,imxils,imxism,imxlmb
      integer*4 imxobs,imxtop,imxite,imxbv,imxnx 
      integer*4 imxilc,imxsnc,imxapo,imxj
      integer*4 imxlins,imxsig_c,imxpar,imxtgt,imxgfit
      integer*4 imxtrgts,imxder
      integer*4 imxparz
      integer*4 imxlines, imxsig_long

      parameter (imxlines = 200000, imxsig_long = 2000000)
      parameter (imxmw=1,imxsig=13010,imxi=210,imxils=2900,
     &          imxgas=25,imxlin=82000,imxept=1,imxhit=42,imxism=10,
     &          imxgmw=imxgas,imxlmb=40,imxilc=imxils)
      parameter (imxpro=321,imxlev=300,imxsl=90)
      parameter (imxiso=85,imxcof=4)
      parameter (imxnlte=8,imxband=16,imxvib=imxband)
      parameter (imxlins=3)
      parameter (imxtgt=10)
      parameter (imxggg=82)
      parameter (imxgfit=3)
      parameter (imxpar=55)
      parameter (imxlay=imxlev-1)
      parameter (imxobs=1)
      parameter (imxite=15,imxbv=1,imxnx=1)
      parameter (imxj=imxilc+imxi)
      parameter (imxsnc=4800,imxapo=513)
      parameter (imxsi2=imxsig)
      parameter (imxsig_c=20)
      parameter (imxtit=200000)
      parameter (imxvims=400)
      parameter (imxtrgts=2)
      parameter (imxder=40)
      parameter (imxparz=300)
*
*  rg0      = acceleration of gravity (m/s**2)
*  rairmass = average molec. weigth of the air (kg/kmol) (US STD)
*  rmovr    = 1000 * rairmass / R(=8314.32[N.m/(kmol.K)])
*  rt0n     = temperature on level sea for refraction index calculation
*  rp0n     = pressure on level sea for refraction index calculation
*  rp0h     = reference pressure for pressure broadening
*  rt0h     = reference temperature for pressure broadening
*  rt0int   = reference temperature for the line intensity
*  rbc      = Boltzmann constant (for density in mol/cm-3)
*  rhck     = h*c/k [K/cm-1]
*  rc1      = constant for the Planck-function (2 h c^2)
*  rcn      = constant in refraction index expression (n=1.+(rcn*rt0n/rp0n)*p/T) 
*  refind   = multiplicative constant in the expression of refraction index n
*             refind= rcn*rt0n/rp0n 
*  rk       = multiplicative constant to be used for the calculation of columns
*             rk=1.d+5/rbc
*  dsqpi    = sqrt(pi)
*  dsqln2   = sqrt(ln2)
*  dinvpi   = 1/pi
*  dcdop    = used in Doppler broadening: sqrt(2 ln2 k avog / c^2)
*  dext     = extension of the already (with iadd*delta) extended Mw
*             where ioutin is set to 1
*
      real*8 rg0,rairmass,rmovr,rt0n,rp0n,rcn,rc1,dinvpi,dext
      real*8 rt0h,rp0h,rt0int,rbc,rhck,dsqpi,dsqln2,dcdop,refind,rk 
      real*8 pai
      parameter (rg0=9.80665D0, rairmass=28.9644D0,rmovr=3.483676D0,
     &       rt0n=288.16D0,rp0n=1013.25D0,rbc=1.380658D-19,
     &       rhck=1.4387687D0,rcn=.000272632D0,refind=rt0n*rcn/rp0n,
     &       rk=1.d+5/rbc,dext=0.4D0)
      parameter(rt0h=296.D0,rp0h=1013.25D0,rt0int=296.D0)
      parameter(rc1=1.19043934d-3,dsqpi=1.772453851D0,
     &          dsqln2=0.832554611D0,dinvpi=0.318309886D0,
     &          dcdop=3.5811737d-7,pai=3.141592654d0)
*
*                       used in cross:
* rdmult is the number of Doppler half-widths from the line-centre
*  from which the Lorentz function instead of the Voigt-function is used
*  Error: rdmult=10 -> 1.5% ; rdmult=20 -> 0.4% ; rdmult=30 -> 0.18%
* rvmult is the number of (Doppler+Lorentz=~Voigt) half-widths from the
*  line-centre where the transition between local coarse and local fine grid
*  occurs (rvmult >= rdmult !!!!!!!!!!)
* iqlclf is the quotient between coarse and fine grid intervalls.
* rvlf multiplier for (Doppler+Lorentz=~Voigt) half-width to determine the
*  local fine grid
*  optimized parameters (4.6.95): rdmult=30, rvmult=50,rvlf=0.1,iqlclf=5
*
      real*8 rdmult,rvmult,rvlf
      integer*4 iqlclf 
      parameter(rdmult=30.D0,rvmult=50.D0,rvlf=0.1D0,iqlclf=5)
*
*
*
* added parameters for vmr - retrieval : 
*
*
*  imxpre = maximum number of points for the precalculated line shape
*  nrepcode = Hitran code of gas for which line shape is precalculated
*             (=12, HNO3)
*  nrepiso = isotope no. of gas for which line shape is prec.
*            (=1)
*  rephw0 =  reference half width of the line to be prec.
*  repexph = reference exponent of half width of the line to be prec.      
*
      integer*4 imxpre,nrepcode,nrepiso
      real*8 rephw0,repexph     
      parameter(imxpre=imxsig)
      parameter(nrepcode=12,nrepiso=1)
      parameter(rephw0=0.11D0,repexph=0.75D0)
* Continuum handling parameters:
* imxcop = Max. number of continuum parameters
      integer*4 imxsav,imxcop,imxpun,imxhol
      parameter(imxsav=2000,imxcop=130,imxpun=100,imxhol=100)
* dtineig = min. allowed value for the eigenvalues of A for 
*           the singularity not being eliminated.
      real*8 dtineig
      parameter(dtineig=1.d-40)
*
*******************************************************************************
*
* NUOVI parameters
*
* imxray = max No di raggi per la suddivisione in settori
* imxseq = max No di sequenze di limb scanning nell' orbita
* imxint = max No di intersezioni che le linee di vista fanno con livelli+raggi
* imxtru = max No di T nella matrice di occupazione
* imxtop = max No di parametri da fittare. C' era ma e' ridefinito
* imxspic = max No di spicchi nella suddivisione dell' atmosfera      
* imxver = max No di vertici nella suddivisione in spicchi      
* imxlat = maximum number of latitudes in the initial profile
* imxsup = max. n. of points of superpoition of adjacent MWs

      integer*4 imxray,imxseq,imxint,imxtru,imxspic,imxver,imxlat
      integer*4 imxsup
      parameter(imxsup=30)
      parameter(imxray=1500,imxseq=8,imxint=600,imxtru=107)
      parameter(imxtop=imxlmb*imxseq+imxtru*imxseq,
     &          imxspic=(imxlev-1)*imxray,imxver=imxlev*imxray,
     &          imxlat=100)
 
      integer*4 imxgxs   ! Max. N. of X-SECT files available
      integer*4 imxbnd   ! Max. N. of spectral bands in each X-SECT file
      integer*4 imxptp   ! Max. N. of PT pairs in the X-SECT files
      integer*4 imxfxs   ! Max. N. of freq. pts for X-SECT
      integer*4 imxags   ! N. of profiles in file in_vmr.dat (all RFM gases)

      parameter(imxgxs=25,imxbnd=5,imxptp=70,imxfxs=77000,imxags=81)
