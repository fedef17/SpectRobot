************************************************************************
      subroutine sum_all_lines(spe_ini, matrix, init, fin,
     &                        n_lines, n_spe, spe_fin)
Cf2py intent(in) spe_ini, matrix, init, fin, n_lines, n_spe
Cf2py intent(out) spe_fin

      implicit none
      include 'parameters.inc'

      real*8 spe_ini(imxsig_long), matrix(imxlines,imxsig)
      real*8 spe_fin(imxsig_long)
      integer*4 init(imxlines), fin(imxlines), n_lines
      integer*4 n_spe,ilin, i, j

      spe_fin = spe_ini

      do ilin=1,n_lines
        i=1
        do j=init(ilin),fin(ilin)
          spe_fin(j) = spe_fin(j) + matrix(ilin,i)
          i = i+1
        enddo
      enddo

      end

************************************************************************

************************************************************************
      subroutine sum_shapes(y1,y2,summ)
Cf2py intent(in) y1, y2
Cf2py intent(out) summ

      implicit none
      include 'parameters.inc'

      real*8 y1(imxsig),y2(imxsig),summ(imxsig)
      integer*4 j

      do j=1,imxsig
        summ(j) = y1(j) + y2(j)
      enddo

      end
************************************************************************



************************************************************************
***************************************************************************
*  SUBROUTINE                 : humli
*  MODIFIED BY                : michael hoepfner
*  DATE OF MODIFICATION       : 12.1.96
*  MODIFIED BY                :
*  LAST MODIFICATION BY       :
*
*
*  DESCRIPTION    : Calculation of the complex probability function.
*                   The real part of it corresponds to the Voigt-lineshape.
*                   Sources:
*                  -J. Humlicek, 'Optimized computation of the voigt
*                   and complex probability functions', J.Quant.Radiat.
*                   Transfer, 27, 437-444,1982.
*
*  INPUTS:
*   rx              x-coefficient=sqrt(ln2)*(sigma-sigma0)/dopplerhalfwidth
*   ry              y-coefficient=sqrt(ln2)*lorentzhalfwidth/dopplerhalfwi.
*  OUTPUTS:
*   rre             Voigt-lineshape*dopplerhalfwidth*sqrt(pi/ln2)
*  CALLED BY:  cross
*              shapecalc
***************************************************************************
      subroutine humli_bb(rx,ry,rre)
Cf2py intent(in) rx, ry
Cf2py intent(out) rre

      implicit none
      include 'parameters.inc'
*      integer*4 imxsig, imxpre
*      parameter (imxsig=13010,imxpre=imxsig)

      complex*16 c1,c2,c3
      real*8 rx,ry,rre,r1,r2

      r1 = abs(rx) + ry
      r2 = (0.195D0*abs(rx)) - 0.176D0
      c2  = cmplx(ry,-rx)
*
*  region 1
*
      if (r1 .ge. 15.0) then
        c3 = c2*0.5641896D0/(0.5D0+(c2*c2))
        rre=dble(c3)
*
*  region 2
*
      elseif (r1 .ge. 5.5) then
        c1 = c2*c2
        c3 = c2*(1.410474D0 + c1*.5641896D0)/(.75D0 + c1*(3.D0+c1))
        rre=dble(c3)
*
*  region 3
*
      elseif (ry .ge. r2) then
           c3 =
     +     (16.4955D0+c2*(20.20933D0+c2*(11.96482D0+c2*(3.778987D0+
     +     c2*.5642236D0))))/
     +     (16.4955D0+c2*(38.82363D0+c2*(39.27121D0+
     +     c2*(21.69274D0+c2*(6.699398D0+c2)))))
           rre=dble(c3)
*
*  region 4
*
         else
           c1 = c2*c2
           c3=c2*(36183.31D0-c1*(3321.9905D0-
     +     c1*(1540.787D0-c1*(219.0313D0-c1*
     +     (35.76683D0-c1*(1.320522D0-c1*.56419D0))))))/
     +     (32066.6D0-c1*(24322.84D0-c1*
     +     (9022.228D0-c1*(2186.181D0-c1*(364.2191D0-
     +     c1*(61.57037D0-c1*(1.841439D0-c1)))))))
           rre=exp(dble(c1))*cos(dimag(c1))-dble(c3)
         endif

*
*  end of subroutine humli
*
        end
************************************************************************

***************************************************************************
*  SUBROUTINE                 : humliv
*
*  DESCRIPTION    : see humli. This one computes a whole vector for
*                   efficiency. The x(j) are assumed ordered.
*                   i.e. (i>j) ==> (x(i)>x(j)) and equally spaced
*
*  INPUTS:
*     x(imxsig)    input vector
*     i1,i2        start and stop index
*     x0           central frequency
*     lw,dw        Lorentz and Doppler FWHM
*
*  OUTPUTS:
*   y()            Voigt-lineshape*dopplerhalfwidth*sqrt(pi/ln2)
*  CALLED BY:  cross
*
***************************************************************************
      subroutine humliv_bb(x,i1,i2,x0,lw,dw,y)
Cf2py intent(in) x, i1, i2, x0, lw, dw
Cf2py intent(out) y

      implicit none
      include 'parameters.inc'
*      integer*4 imxsig, imxpre
*      parameter (imxsig=13010,imxpre=imxsig)

* Input vars:
      real*8 x(imxsig),x0,lw,dw
      integer*4 i1,i2

* Output vars:
      real*8 y(imxsig)

*Internal vars:
      integer*4 j,k,l,ir,ir2,il,il2
      real*8 rx,ry,drun,dstep,tst,r2

      real*8 xrun,xstep,x2,a,b,c,d,e,f,g,h,ry2
      complex*16 c1,c2,c3

*
* i2 > i1 ?
*

      if(i1.gt.i2) then
         write(*,*)'i1,i2',i1,i2,x0
         stop'Error in humliv: called with i1 > i2'
      endif
*
* Init
*
      if(dw.gt.0.d0) then
         ry=lw/dw
      else
         stop'Error in humliv: called with dw <=0'
      end if
      dstep=(x(i1+1)-x(i1))/dw
      xstep=dstep
      ry2=ry*ry

*
*  forward loop: x0 < x(i1)
*
      if(x0.le.x(i1)) then
         tst=5.5d0
         j=i1
         rx=(x(j)-x0)/dw
         do while((rx+ry.lt.tst).and.(j.le.i2))
            r2=(0.195D0*rx)-0.176D0
            c2=cmplx(ry,-rx)
            if(ry.lt.r2) then   ! region 4
               c1=c2*c2
               c3= c2 *( 36183.31- c1*(3321.9905 -
     &              c1*(1540.787-c1*(219.0313-c1*
     &              (35.76683-c1*(1.320522-
     &              c1*.56419))))))/
     &              (32066.6-c1*(24322.84-c1*
     &              (9022.228-c1*(2186.181-c1*(364.2191-
     &              c1*(61.57037 -c1*(1.841439 -c1)))))))
!               c3= c2 *( 36183.31D0 - c1*(3321.9905D0 -
!     &              c1*(1540.787D0-c1*(219.0313D0-c1*
!     &              (35.76683D0-c1*(1.320522D0-
!     &              c1*.56419D0))))))/
!     &              (32066.6D0-c1*(24322.84D0-c1*
!     &              (9022.228D0-c1*(2186.181D0-c1*(364.2191D0-
!     &              c1*(61.57037D0-c1*(1.841439D0-c1)))))))

                    y(j)=exp(dble(c1))*cos(dimag(c1))-dble(c3)

            else                ! region 3
               c3 = (16.4955+c2*(20.20933 +c2*
     &              (11.96482 +c2*(3.778987 +
     &              c2*.5642236 ))))/
     &              (16.4955 +c2*(38.82363 +c2*(39.27121 +
     &              c2*(21.69274 +c2*(6.699398 +c2)))))
!               c3 = (16.4955D0+c2*(20.20933D0+c2*
!     &              (11.96482D0+c2*(3.778987D0+
!     &              c2*.5642236D0))))/
!     &              (16.4955D0+c2*(38.82363D0+c2*(39.27121D0+
!     &              c2*(21.69274D0+c2*(6.699398D0+c2)))))

               y(j)=dble(c3)

            end if
            j=j+1
            rx=rx+xstep
         end do
         if(j.le.i2) then
            tst=15.0d0
            l=max(nint((tst-ry-rx)/xstep),0)+j
            l=min(l,i2)
            if(l.gt.j) then
               a=ry*(1.0578555D0+ry2*(4.6545642D0+ry2*
     &              (3.1030428D0+0.5641896D0*ry2)))
               b=ry*(2.9619954D0+ry2*(0.5641896D0+
     &              1.6925688D0*ry2))
               c=ry*(-2.5388532D0+ry2*1.6925688D0)
               d=ry*0.5641896D0
               e=0.5625D0+ry2*(4.5D0+ry2*(10.5D0+
     &              ry2*(6.D0+ry2)))
               f=-4.5D0+ry2*(9.D0+ry2*(6.D0+4.D0*ry2))
               g=10.5D0+ry2*(-6.D0+6.D0*ry2)
               h=4.D0*ry2 - 6.D0
               drun=(x(j)-x0)/dw
               xrun=drun
               do k=j,l
                  x2=xrun*xrun
                  y(k)=(a+x2*(b+x2*(c+d*x2)))/
     &                 (e+x2*(f+x2*(g+x2*(h+x2))))
                  xrun=xrun+xstep
               end do
               l=l+1
            end if
            if(l.lt.j) l=j
            if(l.lt.i2) then
               a=ry*(1.1283792D0+2.2567584D0*ry2)
               b=2.2567584D0*ry
               c=(1.D0+2.D0*ry2)*(1.D0+2.D0*ry2)
               d=-4.D0 + 8.D0*ry2
               e=4.D0
               drun=(x(l)-x0)/dw
               xrun=drun
               do k=l,i2
                  x2=xrun*xrun
                  y(k)=(a+x2*b)/(c+x2*(d+e*x2))
                  xrun=xrun+xstep
               end do
            end if              ! l<= i2
         end if                 ! j<=i2
      elseif(x0.ge.x(i2)) then  ! x0 > x(i2)
         tst=5.5d0
         j=i2
         rx=(x0-x(j))/dw
         do while((rx+ry.lt.tst).and.(j.ge.i1))
            r2=(0.195D0*rx)-0.176D0
            c2=cmplx(ry,-rx)
            if(ry.lt.r2) then   ! region 4
               c1=c2*c2
               c3=c2*(36183.31 -c1*(3321.9905 -
     &              c1*(1540.787 -c1*(219.0313 -c1*
     &              (35.76683 -c1*(1.320522 -
     &              c1*.56419 ))))))/
     &              (32066.6 -c1*(24322.84 -c1*
     &              (9022.228 -c1*(2186.181 -c1*(364.2191 -
     &              c1*(61.57037 -c1*(1.841439 -c1)))))))

!               c3=c2*(36183.31D0-c1*(3321.9905D0-
!     &              c1*(1540.787D0-c1*(219.0313D0-c1*
!     &              (35.76683D0-c1*(1.320522D0-
!     &              c1*.56419D0))))))/
!     &              (32066.6D0-c1*(24322.84D0-c1*
!     &              (9022.228D0-c1*(2186.181D0-c1*(364.2191D0-
!     &              c1*(61.57037D0-c1*(1.841439D0-c1)))))))

                    y(j)=exp(dble(c1))*cos(dimag(c1))-dble(c3)

            else                ! region 3
!               c3=(16.4955D0+c2*(20.20933D0+c2*
!     &              (11.96482D0+c2*(3.778987D0+
!     &              c2*.5642236D0))))/
!     &              (16.4955D0+c2*(38.82363D0+c2*(39.27121D0+
!     &              c2*(21.69274D0+c2*(6.699398D0+c2)))))
               c3=(16.4955 +c2*(20.20933 +c2*
     &              (11.96482 +c2*(3.778987 +
     &              c2*.5642236 ))))/
     &              (16.4955 +c2*(38.82363 +c2*(39.27121 +
     &              c2*(21.69274 +c2*(6.699398 +c2)))))

               y(j)=dble(c3)
            end if
            j=j-1
            rx=rx+xstep
         end do
         if(j.ge.i1) then
            tst=15.0d0
            l=j-max(nint((tst-ry-rx)/dw/xstep),0)
            l=max(l,i1)
            if(l.eq.i2) l=i2+1
            if(l.lt.j) then
               a=ry*(1.0578555D0+ry2*(4.6545642D0+ry2*
     &              (3.1030428D0+0.5641896D0*ry2)))
               b=ry*(2.9619954D0+ry2*(0.5641896D0+
     &              1.6925688D0*ry2))
               c=ry*(-2.5388532D0+ry2*1.6925688D0)
               d=ry*0.5641896D0
               e=0.5625D0+ry2*(4.5D0+ry2*(10.5D0+
     &              ry2*(6.D0+ry2)))
               f=-4.5D0+ry2*(9.D0+ry2*(6.D0+4.D0*ry2))
               g=10.5D0+ry2*(-6.D0+6.D0*ry2)
               h=4.D0*ry2 - 6.D0
               drun=(x0-x(l))/dw
               xrun=drun
               do k=l,j
                  x2=xrun*xrun
                  y(k)=(a+x2*(b+x2*(c+d*x2)))/
     &                 (e+x2*(f+x2*(g+x2*(h+x2))))
                  xrun=xrun-xstep
               end do
            end if
            if(l.ge.i1) then
               a=ry*(1.1283792D0+2.2567584D0*ry2)
               b=2.2567584D0*ry
               c=(1.D0+2.D0*ry2)*(1.D0+2.D0*ry2)
               d=-4.D0 + 8.D0*ry2
               e=4.D0
               drun=(x0-x(i1))/dw
               xrun=drun
               do k=i1,l-1
                  x2=xrun*xrun
                  y(k)=(a+x2*b)/(c+x2*(d+e*x2))
                  xrun=xrun-xstep
               end do
            end if
         end if
      else                      ! x(i1)<x0<x(i2)
         rx=(x0-x(i1))/dw
         tst=15.d0
         il=i1
         if(rx+ry.ge.tst) then ! region 4, left
            il=max(nint((rx-ry-tst)/xstep),0)+i1
         end if
         rx=(x(i2)-x0)/dw
         ir=i2
         if(rx+ry.ge.tst) then ! region 4, right
            ir=i2-max(nint((rx-ry-tst)/xstep),0)
         end if
         if(il.gt.i1 .or. ir.lt.i2) then
            a=ry*(1.1283792D0+2.2567584D0*ry2)
            b=2.2567584D0*ry
            c=(1.D0+2.D0*ry2)*(1.D0+2.D0*ry2)
            d=-4.D0 + 8.D0*ry2
            e=4.D0
            if(il.gt.i1) then
               drun=(x0-x(i1))/dw
               xrun=drun
               do k=i1,il
                  x2=xrun*xrun
                  y(k)=(a+x2*b)/(c+x2*(d+e*x2))
                  xrun=xrun-xstep
               end do
            end if
            if(ir.lt.i2) then
               drun=(x(ir)-x0)/dw
               xrun=drun
               do k=ir,i2
                  x2=xrun*xrun
                  y(k)=(a+x2*b)/(c+x2*(d+e*x2))
                  xrun=xrun+xstep
               end do
            end if
         end if
         rx=(x0-x(il))/dw
         tst=5.5d0
         il2=il
         if(rx+ry.ge.tst) then ! region 3, left
            il2=il+max(nint((rx-ry-tst)/xstep),0)
         end if
         ir2=ir
         rx=(x(ir)-x0)/dw
         if(rx+ry.ge.tst) then ! region 3, right
            ir2=ir-max(nint((rx-ry-tst)/xstep),0)
         end if
         if(il2.gt.il .or. ir2.lt.ir) then
            a=ry*(1.0578555D0+ry2*(4.6545642D0+ry2*
     &           (3.1030428D0+0.5641896D0*ry2)))
            b=ry*(2.9619954D0+ry2*(0.5641896D0+
     &           1.6925688D0*ry2))
            c=ry*(-2.5388532D0+ry2*1.6925688D0)
            d=ry*0.5641896D0
            e=0.5625D0+ry2*(4.5D0+ry2*(10.5D0+
     &           ry2*(6.D0+ry2)))
            f=-4.5D0+ry2*(9.D0+ry2*(6.D0+4.D0*ry2))
            g=10.5D0+ry2*(-6.D0+6.D0*ry2)
            h=4.D0*ry2 - 6.D0
            if(il.lt.il2) then
               drun=(x0-x(il))/dw
               xrun=drun
               do j=il,il2
                  x2=xrun*xrun
                  y(j)=(a+x2*(b+x2*(c+d*x2)))/
     &                 (e+x2*(f+x2*(g+x2*(h+x2))))
                  xrun=xrun-xstep
               end do
            end if
            if(ir2.lt.ir) then
               drun=(x(ir2)-x0)/dw
               xrun=drun
               do j=ir2,ir
                  x2=xrun*xrun
                  y(j)=(a+x2*(b+x2*(c+d*x2)))/
     &                 (e+x2*(f+x2*(g+x2*(h+x2))))
                  xrun=xrun+xstep
               end do
            end if
         end if
         if(il2.eq.il) il2=il-1
         if(ir2.eq.ir) ir2=ir+1
         do j=il2+1,ir2-1
            rx=abs(x(j)-x0)/dw
            r2=(0.195D0*rx)-0.176D0
            c2=cmplx(ry,-rx)
            if(ry.lt.r2) then   ! region 4
               c1=c2*c2
!               c3=c2*(36183.31D0-c1*(3321.9905D0-
!     &              c1*(1540.787D0-c1*(219.0313D0-c1*
!     &              (35.76683D0-c1*(1.320522D0-
!     &              c1*.56419D0))))))/
!     &              (32066.6D0-c1*(24322.84D0-c1*
!     &              (9022.228D0-c1*(2186.181D0-c1*(364.2191D0-
!     &              c1*(61.57037D0-c1*(1.841439D0-c1)))))))
               c3=c2*(36183.31 -c1*(3321.9905 -
     &              c1*(1540.787 -c1*(219.0313 -c1*
     &              (35.76683 -c1*(1.320522 -
     &              c1*.56419 ))))))/
     &              (32066.6 -c1*(24322.84 -c1*
     &              (9022.228 -c1*(2186.181 -c1*(364.2191 -
     &              c1*(61.57037 -c1*(1.841439 -c1)))))))
                    y(j)=exp(dble(c1))*cos(dimag(c1))-dble(c3)

            else                ! region 3
!               c3=(16.4955D0+c2*(20.20933D0+c2*
!     &              (11.96482D0+c2*(3.778987D0+
!     &              c2*.5642236D0))))/
!     &              (16.4955D0+c2*(38.82363D0+c2*(39.27121D0+
!     &              c2*(21.69274D0+c2*(6.699398D0+c2)))))
               c3=(16.4955 +c2*(20.20933 +c2*
     &              (11.96482 +c2*(3.778987 +
     &              c2*.5642236 ))))/
     &              (16.4955 +c2*(38.82363 +c2*(39.27121 +
     &              c2*(21.69274 +c2*(6.699398 +c2)))))

               y(j)=dble(c3)
            end if
         end do

      end if                    ! x0 < x(i1)

*
*  end of subroutine humliv
*
      end
************************************************************************

***************************************************************************
*  SUBROUTINE                 : shapecalc
*  CREATED BY                 : michael hoepfner
*  DATE OF CREATION           : 31.1.96
*  DATE OF LAST MODIFICATION  :
*  MODIFIED BY                :
*  LAST MODIFICATION BY       :
*
*  DESCRIPTION  :   Calculation of the lineshape, which is later used
*                   for the calculation of many lines with equal
*                   halfwidths just by shifting and interpolation.
*
*  INPUTS:
*    rp            Equivalent pressure
*    rt            Equivalent temperature
*    dsi1          First wavenumber of the microwindow
*    isi           number of grid-points in the actual mw
*    rw            Molecular weight of the gas
*    delta         Difference between the wavenumber grid-points
*
*  OUTPUTS:
*    rshape(2*imxsig) the precalculated line shape
*
*  SUBROUTINES:
*    humli
*
*  CALLED BY:  cross
***************************************************************************
      subroutine shapecalc_bb(rp,rt,dsi1,isi,delta,rw,rshape)
Cf2py intent(in) rp, rt, dsi1, isi, delta, rw
Cf2py intent(out) rshape

      implicit none
      include 'parameters.inc'
*      integer*4 imxsig, imxpre
*      parameter (imxsig=13010,imxpre=imxsig)

      real*8 rp,rt,rshape(0:imxpre),rw,rx,ry,rdsqln,
     &       rlhalf,rdhalf
      real*8 delta,dsi1,dsil,dsidif
      integer*4 isi,jsig,iprec


*
*  as the central line wavenumber the mean wavenumber of the mw is used
*
      dsil=dsi1+0.5D0*(isi*delta)
*
*  calculation of doppler half-width
*
      rdhalf=dsil * dcdop * sqrt(rt/rw)
*
*  calculation of lorentz half-width
*
      rlhalf=rephw0 * (rp/rp0h) * (rt0h/rt)**repexph
*
*  rdsqln=rdhalf / sqrt(ln 2)
*
      rdsqln=rdhalf/dsqln2
*
*  calculation of the y-coefficient
*
      ry=rlhalf/rdsqln
*
*  calculation of the number of points to be precalculated
*
      iprec=int(rdhalf*rdmult/delta)+10
*
*  begin loop over wavenumber grid
*
      do 10 jsig=0,iprec
        dsidif=jsig*delta
*
*  calculation of x-coefficient
*
        rx=dsidif/rdsqln
*
*  calculation of Voigt-lineshape
*
        call humli_bb(rx,ry,rshape(jsig))
10    continue
      end
***************************************************************************

*************************************************************************
*  FUNCTION                   : fconn2
*  CREATED BY                 : michael hoepfner
*  DATE OF CREATION           : 1.7.96
*  MODIFIED BY                :
*  LAST MODIFICATION BY       :
*
*  DESCRIPTION    : Calculation of the n2- continuum (2100-2640cm-1).
*                  V. Menoux, R. Le Doucen, C. Boulet, A. Roblin, and
*                  A.M. Bouchardy, 'Collision-induced absorption in the
*                  fundamental band of N2: temperature dependence of the
*                  absorption for n2-n2 and n2-o2 pairs', Appl. Opt.,
*                  32, 263-268, 1993.
*
*  INPUTS:
*    dsi       wavenumber where the n2-continuum should be calculated
*    rt        equivalent temperature of n2 of the actual path
*    rp        equivalent pressure of n2 of the actual path
*
*  OUTPUTS:
*    fconn2    absorption cross section of the n2 continuum
*
*************************************************************************
      real*8 function fconn2(dsi,rt,rp)
      implicit none
      include 'parameters.inc'
*      integer*4 imxsig, imxpre
*      parameter (imxsig=13010,imxpre=imxsig)

      integer*4 inpt,itmp
      parameter(inpt=64,itmp=6)
      real*8 dsi,rp,rt,rn,dsi1(inpt),rcoef(inpt,itmp),rt1(itmp),
     &       ratio(itmp),r1,r2,r3,r4,rbiabco_nn,reff,rden
      integer*4 j,i1,i2,it1,it2
      data (rt1(j),j=1,itmp)/193.,213.,233.,253.,273.,297./
      data (ratio(j),j=1,itmp)/1.,0.94,0.94,0.93,0.90,0.83/
      data (dsi1(j),j=1,inpt)/
     &2090.,2100.,2110.,2120.,2130.,2140.,2150.,2160.,2170.,2180.,
     &2190.,2200.,2210.,2220.,2230.,2240.,2250.,2260.,2270.,2280.,
     &2290.,2300.,2305.,2310.,2315.,2320.,2325.,2330.,2335.,2340.,
     &2345.,2350.,2355.,2360.,2365.,2370.,2380.,2390.,2400.,2410.,
     &2420.,2430.,2440.,2450.,2460.,2470.,2480.,2490.,2500.,2510.,
     &2520.,2530.,2540.,2550.,2560.,2570.,2580.,2590.,2600.,2610.,
     &2620.,2630.,2640.,2650./
      data (rcoef(j,6),j=1,inpt)/
     &0.000,0.015,0.020,0.030,0.050,0.080,0.110,0.160,0.205,0.260,
     &0.320,0.385,0.455,0.535,0.625,0.700,0.775,0.845,0.900,0.940,
     &1.000,1.100,1.280,1.450,1.600,1.800,1.930,2.000,1.940,1.820,
     &1.650,1.500,1.430,1.370,1.350,1.345,1.350,1.350,1.340,1.300,
     &1.245,1.150,1.055,0.945,0.800,0.685,0.590,0.505,0.415,0.345,
     &0.295,0.235,0.185,0.150,0.125,0.100,0.085,0.070,0.050,0.035,
     &0.030,0.020,0.015,0.000/
      data (rcoef(j,5),j=1,inpt)/
     &0.000,0.010,0.015,0.020,0.030,0.060,0.100,0.140,0.180,0.230,
     &0.290,0.350,0.430,0.505,0.580,0.680,0.785,0.860,0.925,0.975,
     &1.025,1.185,1.330,1.490,1.675,1.880,2.040,2.100,2.055,1.915,
     &1.755,1.600,1.510,1.450,1.420,1.405,1.400,1.400,1.390,1.365,
     &1.290,1.180,1.060,0.920,0.775,0.630,0.550,0.465,0.385,0.325,
     &0.275,0.230,0.170,0.140,0.110,0.090,0.070,0.060,0.045,0.030,
     &0.030,0.015,0.010,0.000/
      data (rcoef(j,4),j=1,inpt)/
     &0.000,0.010,0.010,0.015,0.020,0.050,0.090,0.125,0.170,0.220,
     &0.275,0.330,0.400,0.460,0.550,0.650,0.780,0.860,0.940,0.990,
     &1.060,1.220,1.370,1.520,1.750,1.975,2.130,2.200,2.165,2.060,
     &1.860,1.665,1.580,1.515,1.490,1.475,1.470,1.475,1.470,1.425,
     &1.330,1.200,1.080,0.890,0.730,0.600,0.515,0.425,0.350,0.300,
     &0.240,0.190,0.150,0.130,0.100,0.080,0.060,0.050,0.035,0.030,
     &0.020,0.020,0.010,0.000/
      data (rcoef(j,3),j=1,inpt)/
     &0.000,0.010,0.010,0.015,0.020,0.040,0.075,0.110,0.155,0.200,
     &0.260,0.310,0.355,0.430,0.525,0.630,0.800,0.850,0.950,1.025,
     &1.100,1.250,1.410,1.560,1.850,2.080,2.285,2.375,2.310,2.165,
     &2.000,1.810,1.700,1.615,1.580,1.570,1.575,1.570,1.550,1.480,
     &1.370,1.220,1.100,0.850,0.690,0.560,0.465,0.385,0.320,0.285,
     &0.210,0.180,0.140,0.110,0.090,0.070,0.050,0.040,0.025,0.020,
     &0.020,0.010,0.010,0.000/
      data (rcoef(j,2),j=1,inpt)/
     &0.000,0.010,0.010,0.010,0.020,0.030,0.060,0.080,0.130,0.160,
     &0.205,0.265,0.325,0.405,0.500,0.610,0.810,0.900,0.990,1.085,
     &1.200,1.355,1.470,1.650,1.960,2.165,2.410,2.540,2.440,2.325,
     &2.140,1.900,1.800,1.750,1.725,1.720,1.720,1.710,1.670,1.570,
     &1.440,1.250,1.080,0.830,0.650,0.530,0.450,0.350,0.300,0.260,
     &0.200,0.160,0.120,0.100,0.080,0.050,0.040,0.025,0.015,0.010,
     &0.010,0.010,0.005,0.000/
      data (rcoef(j,1),j=1,inpt)/
     &0.000,0.005,0.010,0.010,0.015,0.020,0.030,0.050,0.080,0.110,
     &0.150,0.205,0.275,0.350,0.455,0.580,0.860,1.010,1.130,1.230,
     &1.350,1.500,1.650,1.850,2.150,2.470,2.640,2.750,2.700,2.580,
     &2.400,2.150,2.025,1.970,1.955,1.960,1.950,1.940,1.850,1.750,
     &1.610,1.350,1.100,0.800,0.630,0.500,0.400,0.335,0.270,0.220,
     &0.180,0.140,0.110,0.075,0.050,0.035,0.025,0.010,0.010,0.010,
     &0.005,0.005,0.002,0.000/
*
*  if the wavenumber is in the range 2090-2650 cm-1: calculate continuum
*
      if (dsi.gt.dsi1(1).and.dsi.lt.dsi1(inpt)) then
*
* search index for wavenumber:
*
        do 10 j=2,inpt
          if (dsi.lt.dsi1(j)) then
            i1=j-1
            i2=j
            goto 20
          end if
10      continue
20      continue
*
* search index for temperature:
*
        do 30 j=2,itmp
         if (rt.lt.rt1(j)) then
           it1=j-1
           it2=j
           goto 40
         end if
30      continue
        it1=itmp-1
        it2=itmp
40      continue
*
*  linear interpolation to the wavenumber
*
        r3=(dsi-dsi1(i1))/(dsi1(i2)-dsi1(i1))
        r1=rcoef(i1,it1)+(rcoef(i2,it1)-rcoef(i1,it1))*r3
        r2=rcoef(i1,it2)+(rcoef(i2,it2)-rcoef(i1,it2))*r3
*
*  linear interpolation(extrapolation) to the temperature
*  result: binary absorption coefficient for n-n collision
*
        r4=(rt-rt1(it1))/(rt1(it2)-rt1(it1))
        rbiabco_nn=r1+(r2-r1)*r4
*
* linear interpolation of the collision efficiency with temperature
*
        reff=ratio(it1)+(ratio(it2)-ratio(it1))*r4
*
*  calculation of the air density
*
        rden=7.24292d18*rp/rt
*
* calculation of the absorption cross section
*  (1.38..e-45=1e-6/(2*Loschmidt-constant))
*
        fconn2=(0.789+reff*0.211)*rbiabco_nn*rden*1.3852919d-45
      else
       fconn2=0.
      end if
      end
*************************************************************************
*************************************************************************
*  FUNCTION                   : fcono2
*  CREATED BY                 : michael hoepfner
*  DATE OF CREATION           : 9.7.96
*  MODIFIED BY                :
*  LAST MODIFICATION BY       :
*
*  DESCRIPTION    : Calculation of the o2-continuum:
*                   Ref:
*                   J.J. Orlando,G.S.Tyndall,K.E.Nickerson,J.G.Calvert,
*                   'The temperature dependence of collision induced
*                    absorption by Oxygen near 6 um',
*                    JGR,96,D11,20755-20760
*
*
*  INPUTS:
*    dsi       wavenumber where the o2-continuum should be calculated
*    rt        equivalent temperature of o2 of the actual path
*    rp        equivalent pressure of o2 of the actual path
*
*  OUTPUTS:
*    fcono2    absorption cross section of the o2 continuum
*
*************************************************************************
      real*8 function fcono2(dsi,rt,rp)
      implicit none
      include 'parameters.inc'
*      integer*4 imxsig, imxpre
*      parameter (imxsig=13010,imxpre=imxsig)

      integer inpt
      parameter(inpt=201)
      real*8 ran2(inpt),rbn2(inpt),rcn2(inpt)
      real*8 rao2(inpt),rbo2(inpt),rco2(inpt)
      real*8 dsi,rp,rt
      real*8 dsi1,dsi2,ddsi,dsii1
      real*8 ran2i,rbn2i,rcn2i,rao2i,rbo2i,rco2i,r1,rkn2,rko2,rden
      integer*4 i1,i2
      common /paro2/ran2,rbn2,rcn2,rao2,rbo2,rco2
      data dsi1,dsi2,ddsi/1400.,1800.,2./
*
*  if the wavenumber is in the range 1400-1800 cm-1: calculate continuum
*
      if (dsi.gt.dsi1.and.dsi.lt.dsi2) then
*
* calculate index for wavenumber:
*
        i1=int((dsi-dsi1)/ddsi) + 1
        i2=i1+1
*
*  linear wavenumber interpolation
*
        dsii1=dsi1+(i1-1)*ddsi
        r1=(dsi-dsii1) / ddsi
        ran2i=ran2(i1)+(ran2(i2)-ran2(i1)) * r1
        rbn2i=rbn2(i1)+(rbn2(i2)-rbn2(i1)) * r1
        rcn2i=rcn2(i1)+(rcn2(i2)-rcn2(i1)) * r1
        rao2i=rao2(i1)+(rao2(i2)-rao2(i1)) * r1
        rbo2i=rbo2(i1)+(rbo2(i2)-rbo2(i1)) * r1
        rco2i=rco2(i1)+(rco2(i2)-rco2(i1)) * r1
        if(rt.eq.220..and.dsi.eq.1601.)
     &   print*,ran2i,rbn2i,rcn2i,rao2i,rbo2i,rco2i
*
* interpolation to the temperature -> binary absorption coefficients
*  (results in 10^-45 cm^5 molecule^-2)
*
        r1=rt/100.
        rkn2=ran2i+rbn2i*r1+rcn2i*r1*r1
        rko2=rao2i+rbo2i*r1+rco2i*r1*r1
*
*  calculation of the air density
*
        rden=7.24292d18*rp/rt
*
* calculation of the absorption cross section
*  (result in cm^2/molecule)
*
        fcono2=(0.789*rkn2+0.211*rko2)*rden*1.d-45
      else
       fcono2=0.
      end if
      end
*************************************************************************
***************************************************************************
*  BLOCK DATA                 : cocono2
*  CREATED BY                 : michael hoepfner
*  DATE OF CREATION           : 9.7.96
*  DATE OF LAST MODIFICATION  :
*  MODIFIED BY                :
*  LAST MODIFICATION BY       :
*
*
*  DESCRIPTION     : Contains the parameters for the o2-contiuum,
*                    used in fcono2.
*  k = A + B(T/100) + C(T/100)^2 IN UNITS OF 10^-45 CM^5 MOLEC^-2
*  WITH T IN KELVIN
*
*     The original data with resolution of ca.0.46cm-1 have been smoothed
*     to the difference of 2cm-1 using a boxcar of 2cm-1 width.
*
*                       first wavenumber: 1400cm-1
*                       last wavenumber:  1800cm-1
*                       difference:       2cm-1
*
*  OUTPUTS:
*         ran2,rbn2,rcn2,rao2,rbo2,rco2
***************************************************************************
      block data cocono2
      implicit none
      integer inpt
      parameter(inpt=201)
      real*8 ran2(inpt),rbn2(inpt),rcn2(inpt)
      real*8 rao2(inpt),rbo2(inpt),rco2(inpt)
      common /paro2/ran2,rbn2,rcn2,rao2,rbo2,rco2
*
*     n2-coefficients
*
       data ran2/
     & 0.8338, 0.3252, 0.5941,-1.9909,-0.4284, 1.2427, 1.8601,
     & 2.0459, 2.7521, 1.0432,-0.6456,-0.3063,-1.4817,-1.1882,
     & 0.5519,-2.0230, 0.5589,-0.4755, 1.3585, 0.8203, 0.1872,
     & 1.6585,-2.0038, 0.8011,-1.0008, 0.1287,-0.2070,-1.3659,
     & 2.1838,-1.5326, 3.9681, 1.4065, 0.6703, 3.9185, 1.1161,
     & 2.8951, 3.2494, 2.2142, 3.8101, 2.2459, 2.5915, 3.0403,
     & 2.2514, 2.0334, 1.8486, 1.1049, 3.8817, 3.0357, 5.5624,
     & 5.0952, 6.2089, 6.8301, 5.2948, 5.8911, 6.9207, 9.5082,
     & 8.9812, 9.2534,11.4455,11.1232,10.6398,10.2904,10.8923,
     &10.3374,12.6393,14.2395,13.6580,11.1595,12.1398,13.2150,
     &13.5813,16.0297,18.6975,16.4025,17.6430,18.2316,19.4232,
     &15.7450,17.5787,15.5633,16.6350,18.2538,18.5372,16.7940,
     &14.6615,14.8990,14.9848,14.3228,14.5115,12.3334,15.7348,
     &13.9588,12.4117,16.0892,16.1193,14.5068,13.9017,12.8940,
     &15.3238,15.1795,15.9795,14.1545,13.9350,15.1092,14.8780,
     &14.8897,15.8913,11.0800,14.0040, 9.2903,10.7538, 8.4847,
     & 7.4341,11.8377,11.0880, 9.4219,10.5377, 6.7509, 5.1124,
     & 7.3145, 8.6192, 8.4325, 7.0094, 6.5484, 4.6524, 3.7349,
     & 2.1837, 0.8540, 2.6612, 3.6900, 5.3597, 4.3818, 4.2934,
     & 1.4499, 0.0713, 2.4226, 0.5787, 1.6480, 3.0431, 2.6583,
     & 0.2409, 1.8491, 1.1420,-0.6066,-0.5242, 1.1550, 0.6951,
     & 0.4917, 0.6307,-0.9017,-1.8957, 0.0912, 0.1381, 1.2475,
     & 1.1953, 2.9540,-0.3206, 1.4140, 0.5343,-0.2674, 0.8747,
     & 1.5793, 0.3035, 4.4709, 4.1612, 4.0605, 4.2652, 0.7769,
     & 0.6478, 7.1128, 4.1568, 3.2938, 2.7150, 2.2877, 0.2945,
     & 1.9548, 2.1216, 3.3283, 1.2027, 0.5987, 2.1435, 2.6405,
     & 3.1827, 0.2785, 1.2328, 1.0883, 0.2276, 1.9297, 1.2603,
     &-2.1295, 1.4715, 1.9788, 1.3886, 1.1339, 2.8055, 1.5492,
     &-0.0851, 0.5162, 0.0857,-0.2469, 1.0799 /
*
      data rbn2/
     &-0.3511, 0.0888,-0.1802, 1.5995, 0.4620,-0.6881,-1.0933,
     &-1.1380,-1.8783,-0.4629, 0.6943, 0.5449, 1.3472, 1.1605,
     &-0.0614, 1.8496, 0.0737, 0.8229,-0.4544,-0.0101, 0.4684,
     &-0.5614, 2.0825, 0.1127, 1.4219, 0.7062, 1.0897, 2.0307,
     &-0.3642, 2.1360,-1.5803, 0.3690, 0.8629,-1.3736, 0.6293,
     &-0.6340,-0.8947,-0.0273,-1.1825,-0.0629,-0.1494,-0.4891,
     & 0.3294, 0.5604, 0.7640, 1.4695,-0.4380, 0.2152,-1.5600,
     &-1.0263,-1.7001,-2.1593,-1.0211,-1.2746,-2.0504,-3.7466,
     &-3.4150,-3.5358,-4.9711,-4.7652,-4.4991,-4.0196,-4.2080,
     &-3.8511,-5.3794,-6.1847,-5.6577,-3.8779,-4.2692,-4.7232,
     &-4.6701,-6.1394,-7.7430,-5.8906,-6.6191,-6.5072,-7.2341,
     &-4.6262,-5.8451,-4.5560,-5.4047,-6.7243,-6.9979,-6.0970,
     &-4.6646,-5.0282,-5.3977,-5.0254,-5.1772,-3.8007,-6.2118,
     &-5.1456,-4.1843,-6.5773,-6.5703,-5.4798,-5.1153,-4.4652,
     &-6.1072,-5.8857,-6.4064,-5.3657,-5.2736,-6.1179,-5.9846,
     &-5.9412,-6.7041,-3.3640,-5.5479,-2.3415,-3.5085,-2.0418,
     &-1.4350,-4.5266,-4.1620,-3.0629,-3.8193,-1.4484,-0.4244,
     &-2.0592,-3.0370,-3.0514,-2.4250,-2.0785,-0.9861,-0.4035,
     & 0.7381, 1.4798, 0.1881,-0.7881,-1.9582,-1.4636,-1.4469,
     & 0.3659, 1.2364,-0.5501, 0.6112,-0.1427,-1.1905,-0.9518,
     & 0.7561,-0.3470, 0.0693, 1.2428, 1.1419,-0.0383, 0.1846,
     & 0.2035, 0.1275, 1.1622, 1.7889, 0.3750, 0.3276,-0.4858,
     &-0.5398,-1.6790, 0.6368,-0.5813,-0.1142, 0.5299,-0.3339,
     &-0.8078, 0.0118,-2.9292,-2.6259,-2.5596,-2.6777,-0.2602,
     &-0.1997,-4.6324,-2.6974,-2.0966,-1.7837,-1.5029,-0.0542,
     &-1.2229,-1.3110,-2.2363,-0.8114,-0.3008,-1.3830,-1.7062,
     &-2.1442,-0.0793,-0.7408,-0.5837, 0.0005,-1.1904,-0.8416,
     & 1.4078,-0.8567,-1.3341,-0.9772,-0.8155,-1.9269,-0.9753,
     & 0.1467,-0.2974,-0.0479, 0.1779,-0.7767 /
*
      data rcn2/
     & 0.0772,-0.0216, 0.0433,-0.2463,-0.0391, 0.1485, 0.2198,
     & 0.2087, 0.3892, 0.1162,-0.0726,-0.0630,-0.1968,-0.1687,
     & 0.0500,-0.2899, 0.0216,-0.1071, 0.1202, 0.0336,-0.0467,
     & 0.1385,-0.3286, 0.0197,-0.2093,-0.0896,-0.1669,-0.3443,
     & 0.0668,-0.3398, 0.2788,-0.0759,-0.1468, 0.2350,-0.1037,
     & 0.1282, 0.1828, 0.0188, 0.2288, 0.0387, 0.0335, 0.1075,
     &-0.0719,-0.1130,-0.1575,-0.3010, 0.0326,-0.0851, 0.2348,
     & 0.1129, 0.2151, 0.3083, 0.1117, 0.1372, 0.2867, 0.5655,
     & 0.5224, 0.5349, 0.7747, 0.7561, 0.7412, 0.6274, 0.6393,
     & 0.5925, 0.8533, 0.9497, 0.8658, 0.5801, 0.6231, 0.6884,
     & 0.6606, 0.8993, 1.1518, 0.8099, 0.9354, 0.8523, 0.9747,
     & 0.5259, 0.7241, 0.5384, 0.6956, 0.9225, 0.9555, 0.8387,
     & 0.5806, 0.6472, 0.7409, 0.6709, 0.6865, 0.4571, 0.8629,
     & 0.7001, 0.5512, 0.9273, 0.9172, 0.7336, 0.6721, 0.5713,
     & 0.8442, 0.7808, 0.8644, 0.7231, 0.7083, 0.8693, 0.8489,
     & 0.8345, 0.9644, 0.3845, 0.7935, 0.2498, 0.4704, 0.2297,
     & 0.1304, 0.6566, 0.6263, 0.4457, 0.5550, 0.1786, 0.0083,
     & 0.2976, 0.4756, 0.4927, 0.4321, 0.3450, 0.1867, 0.0920,
     &-0.1174,-0.2321,-0.0236, 0.1751, 0.3647, 0.3000, 0.2885,
     &-0.0081,-0.1513, 0.1740,-0.0223, 0.0948, 0.2802, 0.2301,
     &-0.0767, 0.0991, 0.0372,-0.1662,-0.1571, 0.0421, 0.0145,
     & 0.0241, 0.0305,-0.1490,-0.2479,-0.0146,-0.0108, 0.1271,
     & 0.1478, 0.3237,-0.0822, 0.1242, 0.0702,-0.0600, 0.0889,
     & 0.1636, 0.0327, 0.5380, 0.4674, 0.4573, 0.4741, 0.0580,
     & 0.0504, 0.7966, 0.4802, 0.3728, 0.3333, 0.2867, 0.0294,
     & 0.2325, 0.2420, 0.4097, 0.1703, 0.0653, 0.2546, 0.3056,
     & 0.3880, 0.0224, 0.1384, 0.1019, 0.0037, 0.2072, 0.1627,
     &-0.2086, 0.1391, 0.2424, 0.1936, 0.1583, 0.3419, 0.1669,
     &-0.0202, 0.0569, 0.0213,-0.0135, 0.1544 /
*
*     o2-coefficients
*
      data rao2/
     &-0.9740,-0.4652,-1.9048,-4.1859,-3.6752,-2.3123,-2.1715,
     &-3.5489,-3.5006,-2.0589,-2.1889,-1.6985,-2.3701,-2.0742,
     &-2.2795,-3.1066,-3.4849,-2.0139,-2.0013,-1.1607,-1.8444,
     &-2.1751,-1.7883,-2.2177,-2.4519,-1.4836,-1.3604,-1.0434,
     &-0.1096, 1.3337, 0.2890, 0.3661, 1.2833, 1.4055,-0.0493,
     &-0.6347, 0.2023, 1.8390, 2.4094, 1.5622, 0.7282, 0.9504,
     & 1.1096, 1.8438, 3.2691, 4.6977, 4.6473, 4.3898, 4.8161,
     & 6.0214, 5.2925, 5.6625, 6.0638, 7.0042, 9.2439, 8.9995,
     & 7.5303, 8.0106, 9.0700, 8.8291, 8.6347,10.6380,11.2066,
     & 9.4439, 9.2751, 8.3983,10.3005,11.6520,12.7026,12.5238,
     &14.6052,16.9723,15.7072,14.8935,16.2675,17.4788,18.8102,
     &19.5232,21.6785,21.4455,21.7575,20.3625,17.5294,15.5140,
     &14.8840,15.2078,14.2535,13.5392,13.1758,14.5556,14.1517,
     &12.7932,11.8688,11.7772,13.1173,12.1812,11.3013,11.7208,
     &12.3498,13.4743,13.9010,13.1697,12.2632,12.0835,11.1798,
     &11.8908,11.8598,11.4002, 9.2102, 9.4736,10.5932, 8.9645,
     & 7.3809, 7.7449, 8.1389, 8.1693, 7.8748, 4.6177, 3.5832,
     & 4.6924, 3.6910, 2.9627, 3.1526, 3.3931, 3.3958, 2.6854,
     & 2.2745, 3.1182, 2.7254,-0.7269,-0.7594,-1.1030,-0.0248,
     & 0.1495,-1.7404, 0.2396, 1.0820,-0.9409, 0.3545,-0.9249,
     &-2.4652,-0.8291,-1.1067,-0.0049,-0.5605,-2.1405,-1.7370,
     &-1.2455, 0.3080, 1.1392,-0.3645, 1.3180,-0.4377,-1.7553,
     &-0.5579,-0.3328,-0.8520,-0.3244, 0.0390, 0.6534,-0.2118,
     &-2.8482,-2.7259,-1.7205,-0.9888, 0.0322, 1.7415, 1.6393,
     & 2.5997, 1.1016, 1.2896, 0.2486,-1.4121,-1.3167,-0.8622,
     &-0.4823, 0.6862,-0.0073,-1.4674,-0.2388, 0.4956, 0.1664,
     &-0.3236,-1.1881,-0.3572, 1.3373, 0.5741, 0.5937, 0.6778,
     &-0.7513,-0.0130, 0.7895, 1.1132, 0.1063,-0.5657,-0.4340,
     & 0.1771,-0.2024,-0.3285, 0.3689, 0.6318 /
*
      data rbo2/
     & 0.6956, 0.4121, 1.4193, 3.0067, 2.7149, 1.7803, 1.6634,
     & 2.6063, 2.4693, 1.4195, 1.5911, 1.3663, 1.8486, 1.6568,
     & 1.8878, 2.5069, 2.7556, 1.7118, 1.6745, 1.1132, 1.6728,
     & 1.9254, 1.6997, 2.0557, 2.2493, 1.6351, 1.6554, 1.4818,
     & 0.7248,-0.2550, 0.6585, 0.6886, 0.0830, 0.0411, 1.1691,
     & 1.5846, 0.9917,-0.0414,-0.2617, 0.4180, 1.0925, 0.9809,
     & 0.9732, 0.5556,-0.4125,-1.3693,-1.2231,-0.9876,-1.2337,
     &-1.9849,-1.3548,-1.4881,-1.7287,-2.4298,-3.9591,-3.6579,
     &-2.5439,-2.7878,-3.5224,-3.3340,-3.1690,-4.4951,-4.7605,
     &-3.4041,-3.1804,-2.4582,-3.6587,-4.4792,-5.0234,-4.7244,
     &-6.0161,-7.3841,-6.2101,-5.3112,-5.9903,-6.5967,-7.3353,
     &-7.7059,-9.2488,-9.2152,-9.5274,-8.5950,-6.6693,-5.4331,
     &-5.2154,-5.6399,-5.1455,-4.8332,-4.7435,-5.7572,-5.4302,
     &-4.5342,-3.9465,-3.8486,-4.7411,-4.1187,-3.6215,-3.9558,
     &-4.3156,-5.0375,-5.3458,-4.8668,-4.3105,-4.1804,-3.5364,
     &-4.0943,-4.2043,-4.0063,-2.5603,-2.7500,-3.4938,-2.5237,
     &-1.5947,-1.8869,-2.1864,-2.2736,-2.2352,-0.2345, 0.2837,
     &-0.5595, 0.1210, 0.5497, 0.2188,-0.1249,-0.2525, 0.1242,
     & 0.1842,-0.6053,-0.2801, 2.1389, 2.1047, 2.1785, 1.3123,
     & 1.1165, 2.2754, 0.8112, 0.0996, 1.4182, 0.5105, 1.3578,
     & 2.3736, 1.1597, 1.2240, 0.3735, 0.8748, 1.9672, 1.6767,
     & 1.2809, 0.0762,-0.6124, 0.3824,-0.7688, 0.4685, 1.3763,
     & 0.5799, 0.4645, 0.7786, 0.3618, 0.0294,-0.4454, 0.2507,
     & 2.2107, 2.0941, 1.4043, 0.8385, 0.0317,-1.2364,-1.2063,
     &-1.8318,-0.6531,-0.8158,-0.1122, 1.0415, 1.0101, 0.7088,
     & 0.3949,-0.5302, 0.0139, 1.1200, 0.2292,-0.3129,-0.0871,
     & 0.2951, 0.8927, 0.2732,-0.9000,-0.4144,-0.4703,-0.4672,
     & 0.6248, 0.1043,-0.5183,-0.6288, 0.0950, 0.4281, 0.2682,
     &-0.1221, 0.1957, 0.3024,-0.2122,-0.3792 /
*
      data rco2/
     &-0.0993,-0.0618,-0.2279,-0.4989,-0.4581,-0.2966,-0.2719,
     &-0.4267,-0.3772,-0.1801,-0.2259,-0.2072,-0.2838,-0.2485,
     &-0.2965,-0.4057,-0.4418,-0.2514,-0.2291,-0.1328,-0.2386,
     &-0.2749,-0.2371,-0.3049,-0.3402,-0.2372,-0.2465,-0.2159,
     &-0.0576, 0.1107,-0.0704,-0.0812, 0.0253, 0.0322,-0.1702,
     &-0.2325,-0.1198, 0.0459, 0.0605,-0.0633,-0.1885,-0.1699,
     &-0.1755,-0.1093, 0.0592, 0.2246, 0.1839, 0.1439, 0.1891,
     & 0.3090, 0.1910, 0.2000, 0.2393, 0.3736, 0.6370, 0.5722,
     & 0.3777, 0.4127, 0.5454, 0.5161, 0.4969, 0.7231, 0.7628,
     & 0.5240, 0.4808, 0.3527, 0.5597, 0.7019, 0.7923, 0.7411,
     & 0.9599, 1.1680, 0.9435, 0.7547, 0.8480, 0.9391, 1.0458,
     & 1.0886, 1.3592, 1.3759, 1.4349, 1.2629, 0.9216, 0.7185,
     & 0.7043, 0.7904, 0.7110, 0.6740, 0.6724, 0.8321, 0.7559,
     & 0.6016, 0.5060, 0.4833, 0.6242, 0.5168, 0.4489, 0.5111,
     & 0.5609, 0.6776, 0.7297, 0.6489, 0.5617, 0.5389, 0.4302,
     & 0.5347, 0.5704, 0.5500, 0.3182, 0.3475, 0.4665, 0.3200,
     & 0.1765, 0.2237, 0.2772, 0.3022, 0.3131, 0.0002,-0.0665,
     & 0.0819,-0.0351,-0.1093,-0.0350, 0.0377, 0.0698, 0.0204,
     & 0.0421, 0.1859, 0.1016,-0.3222,-0.3128,-0.3080,-0.1570,
     &-0.1233,-0.3014,-0.0454, 0.0857,-0.1351, 0.0041,-0.1470,
     &-0.3153,-0.1011,-0.0994, 0.0424,-0.0733,-0.2642,-0.2230,
     &-0.1501, 0.0703, 0.1995, 0.0351, 0.2157,-0.0027,-0.1648,
     &-0.0420,-0.0333,-0.0819,-0.0023, 0.0658, 0.1469, 0.0047,
     &-0.3538,-0.3331,-0.2202,-0.1156, 0.0366, 0.2696, 0.2655,
     & 0.3563, 0.1337, 0.1656, 0.0464,-0.1514,-0.1542,-0.1056,
     &-0.0431, 0.1276, 0.0237,-0.1802,-0.0263, 0.0733, 0.0346,
     &-0.0421,-0.1450,-0.0305, 0.1643, 0.0899, 0.1050, 0.0920,
     &-0.1101,-0.0204, 0.0951, 0.0926,-0.0366,-0.0704,-0.0292,
     & 0.0324,-0.0363,-0.0568, 0.0364, 0.0620 /
      end
***************************************************************************
************************************************************************

*  SUBROUTINE                 : fco2chi
*  CREATED BY                 : michael hoepfner
*  DATE OF CREATION           : 11.1.96
*  DATE OF LAST MODIFICATION  :  4.3.97
*  MODIFIED BY                : georg echle

*  LAST MODIFICATION BY       :
*
*  DESCRIPTION    : Calculation of the chi-factor for the correction
*                   of the co2-lineshape. The chi-factor is calculated
*                   for the n2- and the o2-broadening of co2-lines using
*                   the parametrizations from:
*                  -C. Cousin, R. Le Doucen, C. Boulet, and A. Henry,
*                   'Temperature dependence of the absorption in the
*                    region beyond the 4.3-um band head of co2. 2: n2
*                    and o2 broadening', Appl. Opt.,24,3899-3907,1985.
*                  -V. Menoux, R. Le Doucen, J. Boissoles, and C. Boulet,
*                   'Line shape in the low frequency wing of self- and
*                    n2 broadened v3 co2 lines: temperature dependence
*                    of the asymmetrie', Appl. Opt.,30,281-286,1991.
*
*                    The chi-factors are linearily interpolated in the
*                    ranges 193-238K and 238-296K and linearily extrapolated
*                    to lower(higher) temperatures from these ranges..
*                    This subroutine is only valid for calculations up
*                    to 130cm-1 from the lines center, since beyond
*                    this wavenumber the asymmetrie of the chi-factor
*                    is only known for 296K of n2 and o2 and
*                    for 193K for o2. (the only asymmetrie included here
*                    is for 193K for n2 in the range 50-130cm-1).
*                    The chi-factor is then calculated by weighting of
*                    the n2 and o2 factors according to their atmospheric
*                    abundance
*
*  INPUTS:
*   rt               equivalent temperature
*   dsidif           distance to the line center (sigma-sigma0)
*   nswco2           switch for the calculation of the chi-factor
*                    in the case of co2-lines (=0: no chi-factor,
*                    =1: due to n2/o2 broadening,
*                    =3: only due to n2 -broadening)
*  OUTPUTS:
*   fco2chi          chi-factor
*  CALLED BY: and cross_bb
**************************************************************************
*
      real*8 function fco2chi(rt,dsidif,nswco2)
      implicit none
      integer*4 nswco2
      real*8 rt,rchin2,r1,r2,r3,r4,rchio2,rtq
      real*8 dsidif,dsi

      dsi=abs(dsidif)
*
*  if Temperature < 238K
*
      if (rt.lt.238.) then
        if (dsi.le.5.) then
          rchin2=1.
          rchio2=1.
        else if (dsi.le.9.) then
          r1=1.
          r2=1.968*exp(-0.1354*dsi)
          rtq=(rt-193.)/45.
          rchin2=(r2-r1)*rtq + r1
          rchio2=rchin2
        else if (dsi.le.11.) then
          r1=3.908*exp(-0.1514*dsi)
          r2=1.968*exp(-0.1354*dsi)
          r3=1.
          r4=r2
          rtq=(rt-193.)/45.
          rchin2=(r2-r1)*rtq + r1
          rchio2=(r4-r3)*rtq + r3
        else if (dsi.le.22.) then
          r1=3.908*exp(-0.1514*dsi)
          r2=1.968*exp(-0.1354*dsi)
          r3=7.908*exp(-0.1880*dsi)
          r4=r2
          rtq=(rt-193.)/45.
          rchin2=(r2-r1)*rtq + r1
          rchio2=(r4-r3)*rtq + r3
        else if (dsi.le.23.) then
          r1=3.908*exp(-0.1514*dsi)
          r2=0.160*exp(-0.0214*dsi)
          r3=7.908*exp(-0.1880*dsi)
          r4=r2
          rtq=(rt-193.)/45.
          rchin2=(r2-r1)*rtq + r1
          rchio2=(r4-r3)*rtq + r3
        else if (dsi.le.28.) then
          r1=0.207 - 3.778e-3 * dsi
          r2=0.160*exp(-0.0214*dsi)
          r3=0.122 - 7.539e-4 * dsi
          r4=r2
          rtq=(rt-193.)/45.
          rchin2=(r2-r1)*rtq + r1
          rchio2=(r4-r3)*rtq + r3
        else if (dsi.le.35.) then
          r1=0.219*exp(-0.0276*dsi)
          r2=0.160*exp(-0.0214*dsi)
          r3=0.122 - 7.539e-4 * dsi
          r4=r2
          rtq=(rt-193.)/45.
          rchin2=(r2-r1)*rtq + r1
          rchio2=(r4-r3)*rtq + r3
        else if (dsi.le.50.) then
          r1=0.219*exp(-0.0276*dsi)
          r2=0.160*exp(-0.0214*dsi)
          r3=0.349*exp(-0.0369*dsi)
          r4=r2
          rtq=(rt-193.)/45.
          rchin2=(r2-r1)*rtq + r1
          rchio2=(r4-r3)*rtq + r3
        else if (dsi.le.130.) then
          if (dsidif.lt.0) then
            r1=0.20894*exp(-0.026694*dsi)
          else
            r1=0.146*exp(-0.0196*dsi)
          end if
          r2=0.162*exp(-0.0216*dsi)
          r3=0.129*exp(-0.0170*dsi)
          r4=r2
          rtq=(rt-193.)/45.
          rchin2=(r2-r1)*rtq + r1
          rchio2=(r4-r3)*rtq + r3
        else if(dsi.le.135.) then
          if(dsidif.lt.0) then
            r1=2.824997*exp(-0.0467266*dsi)
          else
            r1=0.146*exp(-0.0196*dsi)
          endif
          r2=0.162*exp(-0.0216*dsi)
          r3=0.129*exp(-0.0170*dsi)
          r4=r2
          rtq=(rt-193.)/45.
          rchin2=(r2-r1)*rtq + r1
          rchio2=(r4-r3)*rtq + r3
        else if(dsi.le.160.) then
          if(dsidif.lt.0) then
            r1=2.824997*exp(-0.0467266*dsi)
          else
            r1=1.164*exp(-0.035*dsi)
          endif
          r2=0.162*exp(-0.0216*dsi)
          r3=0.1455*exp(-0.0350*dsi)
          r4=r2
          rtq=(rt-193.)/45.
          rchin2=(r2-r1)*rtq + r1
          rchio2=(r4-r3)*rtq + r3
        else
          if(dsidif.lt.0) then
            r1=1.192053*exp(-0.0413334*dsi)
          else
           r1=1.164*exp(-0.035*dsi)
          endif
          r2=0.162*exp(-0.0216*dsi)
          r3=0.1455*exp(-0.0350*dsi)
          r4=r2
          rtq=(rt-193.)/45.
          rchin2=(r2-r1)*rtq + r1
          rchio2=(r4-r3)*rtq + r3
        end if
*
*  if Temperature >= 238K
*
      else
        if (dsi.le.0.5) then
          rchin2=1.
          rchio2=1.
        else if (dsi.le.3.) then
          r1=1.
          r2=1.064*exp(-0.1235*dsi)
          rtq=(rt-238.)/58.
          rchin2=(r2-r1)*rtq + r1
          rchio2=1.
        else if (dsi.le.5.) then
          r1=1.
          r2=1.064*exp(-0.1235*dsi)
          r3=1.
          r4=3.341*exp(-0.4021*dsi)
          rtq=(rt-238.)/58.
          rchin2=(r2-r1)*rtq + r1
          rchio2=(r4-r3)*rtq + r3
        else if (dsi.le.8.) then
          r1=1.968*exp(-0.1354*dsi)
          r2=1.064*exp(-0.1235*dsi)
          r3=r1
          r4=3.341*exp(-0.4021*dsi)
          rtq=(rt-238.)/58.
          rchin2=(r2-r1)*rtq + r1
          rchio2=(r4-r3)*rtq + r3
        else if (dsi.le.20.) then
          r1=1.968*exp(-0.1354*dsi)
          r2=1.064*exp(-0.1235*dsi)
          r3=r1
          r4=0.155*exp(-0.0179*dsi)
          rtq=(rt-238.)/58.
          rchin2=(r2-r1)*rtq + r1
          rchio2=(r4-r3)*rtq + r3
        else if (dsi.le.22.) then
          r1=1.968*exp(-0.1354*dsi)
          r2=0.125*exp(-0.0164*dsi)
          r3=r1
          r4=0.155*exp(-0.0179*dsi)
          rtq=(rt-238.)/58.
          rchin2=(r2-r1)*rtq + r1
          rchio2=(r4-r3)*rtq + r3
        else if (dsi.le.50.) then
          r1=0.160*exp(-0.0214*dsi)
          r2=0.125*exp(-0.0164*dsi)
          r3=r1
          r4=0.155*exp(-0.0179*dsi)
          rtq=(rt-238.)/58.
          rchin2=(r2-r1)*rtq + r1
          rchio2=(r4-r3)*rtq + r3
        else if (dsi.le.70.) then
          r1=0.162*exp(-0.0216*dsi)
          r2=0.146*exp(-0.0196*dsi)
          r3=r1
          r4=0.238*exp(-0.0266*dsi)
          rtq=(rt-238.)/58.
          rchin2=(r2-r1)*rtq + r1
          rchio2=(r4-r3)*rtq + r3
        else if (dsi.le.140.) then
          r1=0.162*exp(-0.0216*dsi)
          r2=0.146*exp(-0.0196*dsi)
          r3=r1
          r4=0.146*exp(-0.0196*dsi)
          rtq=(rt-238.)/58.
          rchin2=(r2-r1)*rtq + r1
          rchio2=(r4-r3)*rtq + r3
        else
          r1=0.162*exp(-0.0216*dsi)
          r3=r1
          if(dsidif.lt.0) then
            r2=1.8593*exp(-0.03776*dsi)
            r4=r2
          else
            r2=0.146*exp(-0.0196*dsi)
            r4=r2
          endif
          rtq=(rt-238.)/58.
          rchin2=(r2-r1)*rtq + r1
          rchio2=(r4-r3)*rtq + r3
        end if
      end if
*
*  calculation of the chi-factor by weighting of the chi-factors
*  for n2 and o2 according to the relative abundance in the atmosphere
*  (if this is greater than 1 the chi-factor is set to 1.)
*
      if (nswco2.eq.1) then
        fco2chi=0.789*rchin2+0.211*rchio2
      else if (nswco2.eq.2) then
        fco2chi=rchin2
      else
        write(*,*) 'In fco2chi: wrong input of switch -nswco2-:'
        write(*,*) 'STOP program'
        stop
      end if
      if (fco2chi.gt.1.) fco2chi=1.
      end

******************************************************************************
