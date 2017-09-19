************************************************************************
      subroutine curgod_fort_1(nd, x, n_p, res)
Cf2py intent(in) nd, x, n_p
Cf2py intent(out) res

      implicit none
      include 'parameters.inc'

      real*8 nd(imxstp), x(imxstp)
      real*8 res, dx, fu, D
      integer*4 n_p, i

      res = 0.0
      do i = 1, n_p-1
        dx = x(i+1)-x(i)
        fu = nd(i+1)/nd(i)
        D = log(fu)/dx
        res = res + (nd(i+1)-nd(i))/D
      enddo

      end


      subroutine curgod_fort_2(nd, vmr, x, n_p, res)
Cf2py intent(in) nd, vmr, x, n_p
Cf2py intent(out) res

      implicit none
      include 'parameters.inc'

      real*8 nd(imxstp), vmr(imxstp), x(imxstp)
      real*8 res, dx, A, B, fu, D
      integer*4 n_p, i

      res = 0.0
      do i = 1, n_p-1
        dx = x(i+1)-x(i)
        A = nd(i)*vmr(i)
        B = nd(i)*(vmr(i+1)-vmr(i))/dx
        fu = nd(i+1)/nd(i)
        D = log(fu)/dx
        res = res + (A*D*(fu-1.)+B*fu*(D*dx-1.)+B)/D**2
      enddo

      end


      subroutine curgod_fort_3(nd, vmr, f, x, n_p, res)
Cf2py intent(in) nd, vmr, f, x, n_p
Cf2py intent(out) res

      implicit none
      include 'parameters.inc'

      real*8 nd(imxstp), vmr(imxstp), x(imxstp), f(imxstp)
      real*8 res, dx, A, cc, bb, B, C, fu, D
      integer*4 n_p, i

      res = 0.0
      do i = 1, n_p-1
        dx = x(i+1)-x(i)
        A = nd(i)*vmr(i)*f(i)
        cc = (vmr(i+1)-vmr(i))/dx
        bb = (f(i+1)-f(i))/dx
        B = nd(i)*(vmr(i)*bb+f(i)*cc)
        C = nd(i)*bb*cc
        fu = nd(i+1)/nd(i)
        D = log(fu)/dx
        res = res + (fu*(D*(A*D+B*(D*dx-1.))+C*(D*dx*(D*dx-2.)+2.))
     $ + D*(B-A*D)-2*C)/D**3
      enddo

      end


      subroutine curgod_fort_4(nd, vmr, f, x, n_p, res)
Cf2py intent(in) nd, vmr, f, x, n_p
Cf2py intent(out) res

      implicit none
      include 'parameters.inc'

      real*8 nd(imxstp), vmr(imxstp), x(imxstp), f(imxstp)
      real*8 res, dx, A, cc, B, fu, D
      integer*4 n_p, i

      res = 0.0
      do i = 1, n_p-1
        dx = x(i+1)-x(i)
        A = nd(i)*vmr(i)*f(i)
        cc = (vmr(i+1)-vmr(i))/dx
        B = nd(i)*f(i)*cc
        fu = nd(i+1)*f(i+1)/(nd(i)*f(i))
        D = log(fu)/dx
        res = res + (A*D*(fu-1.)+B*fu*(D*dx-1.)+B)/D**2
      enddo

      end

************************************************************************
