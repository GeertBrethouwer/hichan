!
! Verification of HiCHAN by the method of manufactured solutions (MMS)
!
! The exact solution (periodic in x and z, zero at the walls, divergence free, not polynomial in y):
!
!   wave 1 (wall-normal velocity), theta1 = a1 x + b1 z:
!     v = A(t) phi(y) cos(theta1),   (u, w) = -A(t) (a1, b1)/k1^2 phi'(y) sin(theta1)
!   wave 2 (wall-normal vorticity, v = 0), theta2 = a2 x + b2 z:
!     (u, w) = B(t) (b2, -a2) h(y) cos(theta2)
!   mean flow: u = C(t) U(y), w = D(t) W(y)
!   pressure:  P = E(t) S(y) cos(a1 x + b2 z) + 0.1 y^2
!
! with phi = (1-y^2)^2 cos(1.1y+0.3), h = (1-y^2) exp(y/2), U = (1-y^2)(1+0.2 sin 2y), W = (1-y^2) y,
! S = sin(1.5y). The body force
!
!   F = du/dt - u x omega + grad P - nu Lap u
!
! makes it an exact solution of HiCHAN's equations (rotational form: P is the total pressure; with the
! advective form the code's pressure is P - |u|^2/2, the velocity is the same). All products contain
! only a few Fourier modes, which the 3/2 rule resolves exactly: the errors come from y and from t only.
!
! Scalar n (nscal >= 1; scalar n has amplitude 1 + (n-1)/2), W = (1-y^2)^2 (zero value and slope at the walls):
!
!   s = Sb(y) + W [ c1(t) cos(1.3y) cos(theta1+0.7) + c2(t) exp(0.3y) sin(theta2) + c3(t) sin(y+0.2) ]
!
! Sb = A + B y + C y^2 meets the wall conditions of sbc_bot/sbc_top ('D': s = sval, 'N': ds/dy = sval); the
! source Q = ds/dt + u.grad s - kappa Lap s - ssrc_const - ssrc_u u/ubulk is added to the scalar right-hand side,
! and ri(n) s gvec to the momentum force (buoyancy is then tested too).
!
! Tests (&mms test = ...):
!   'operators' at t = 0: (1) diffusion add_lap against Lap u, (2) nonlinear terms (nonlinear with nu = 0)
!               against u x omega or -(u.grad)u, (3) pressure step: r = u + grad P must give u back, the
!               pressure phi must equal P, and the discrete divergence must vanish
!   'run'       time integration with the forcing from t = 0 to tend with fixed dt (all time schemes:
!               visc_theta = 0 or > 0, ns_method 'im' or 'vv'), error of u, v, w at tend
! Errors: max over all spectral coefficients |f - f_exact| / max |f_exact|, per component (interior rows
! 2..ny-1 for the velocity; walls are zero in both).
!
module mod_mms
  use mpi
  use mod_types
  use mod_param  , only: lx,lz,ny,mx,mz,visc,nl_form,nscal,ri,gvec,diff,ssrc_const,ssrc_u,ubulk, &
                         sbc_bot,sbc_top,sval_bot,sval_top
  use mod_decomp , only: nkx,nkz,nzA,nyA,zA_lo,yA_lo
  use mod_compact, only: yg
  use mod_fields , only: ap
  use mod_fft    , only: fwd_fft
  implicit none
  private
  public :: init_mms,to_spectral,mms_forcing,mms_scalar_forcing,errmax,Q_VEL,Q_LAP,Q_UXW,Q_ADV,Q_FORCE, &
            Q_GRADP,Q_P,Q_S,Q_SADV,Q_SFORCE
  integer, parameter :: Q_VEL = 1, Q_LAP = 2, Q_UXW = 3, Q_ADV = 4, Q_FORCE = 5, Q_GRADP = 6, Q_P = 7
  integer, parameter :: Q_S = 8, Q_SADV = 9, Q_SFORCE = 10   ! scalar: value, -u.grad s + sources, Q
  real(rp) :: a1,b1,a2,b2
  complex(rp), allocatable :: f1(:,:,:),f2(:,:,:),f3(:,:,:)
contains
  subroutine init_mms
    a1 = 2*pi/lx; b1 = 2*pi/lz
    a2 = 2*pi/lx; b2 = -2*(2*pi/lz)
    allocate(f1(nkx,nkz,ny),f2(nkx,nkz,ny),f3(nkx,nkz,ny))
    f1 = 0; f2 = 0; f3 = 0
    !$acc enter data copyin(f1,f2,f3)
  end subroutine init_mms
  !
  ! derivatives 0..3 of a product f*g from those of f and g
  pure function leib(f,g) result(p)
    real(rp), intent(in) :: f(0:3),g(0:3)
    real(rp) :: p(0:3)
    p(0) = f(0)*g(0)
    p(1) = f(1)*g(0) + f(0)*g(1)
    p(2) = f(2)*g(0) + 2*f(1)*g(1) + f(0)*g(2)
    p(3) = f(3)*g(0) + 3*f(2)*g(1) + 3*f(1)*g(2) + f(0)*g(3)
  end function leib
  !
  ! adds C(t) Y(y) T(theta) to component c (T = sin or cos of al x + be z): value, gradient, Laplacian, d/dt
  subroutine addterm(c,cc,cct,y0,y1,y2,is_sin,al,be,x,z,uv,g,lap,ut)
    integer , intent(in) :: c
    real(rp), intent(in) :: cc,cct,y0,y1,y2,al,be,x,z
    logical , intent(in) :: is_sin
    real(rp), intent(inout) :: uv(3),g(3,3),lap(3),ut(3)
    real(rp) :: th,t,tx,tz
    th = al*x + be*z
    if(is_sin) then
      t = sin(th); tx = al*cos(th); tz = be*cos(th)
    else
      t = cos(th); tx = -al*sin(th); tz = -be*sin(th)
    end if
    uv(c)  = uv(c)  + cc*y0*t
    g(c,1) = g(c,1) + cc*y0*tx
    g(c,2) = g(c,2) + cc*y1*t
    g(c,3) = g(c,3) + cc*y0*tz
    lap(c) = lap(c) + cc*(y2 - (al**2+be**2)*y0)*t
    ut(c)  = ut(c)  + cct*y0*t
  end subroutine addterm
  !
  ! exact velocity uv, gradient g(i,j) = d u_i / d x_j, Laplacian, du/dt, grad P and P at (x,y,z,t)
  subroutine exact(x,y,z,t,uv,g,lap,ut,gp,p)
    real(rp), intent(in) :: x,y,z,t
    real(rp), intent(out) :: uv(3),g(3,3),lap(3),ut(3),gp(3),p
    real(rp) :: pa(0:3),qa(0:3),ph(0:3),h(0:3),ub(0:3),wb(0:3),ea(0:3),s(0:3)
    real(rp) :: aa,aat,bb,bbt,cm,cmt,dm,dmt,ee,k1
    uv = 0; g = 0; lap = 0; ut = 0
    aa = 0.5_rp*(1+0.3_rp*sin(t));       aat = 0.15_rp*cos(t)
    bb = 0.4_rp*(1+0.2_rp*cos(1.3_rp*t)); bbt = -0.104_rp*sin(1.3_rp*t)
    cm = 1+0.25_rp*sin(0.8_rp*t);        cmt = 0.2_rp*cos(0.8_rp*t)
    dm = 0.3_rp*cos(0.5_rp*t);           dmt = -0.15_rp*sin(0.5_rp*t)
    ee = 0.2_rp*(1+sin(t))
    ! phi = (1-y^2)^2 cos(1.1y+0.3)
    pa = [(1-y**2)**2, -4*y*(1-y**2), -4+12*y**2, 24*y]
    qa = [cos(1.1_rp*y+0.3_rp), -1.1_rp*sin(1.1_rp*y+0.3_rp), -1.21_rp*cos(1.1_rp*y+0.3_rp), &
          1.331_rp*sin(1.1_rp*y+0.3_rp)]
    ph = leib(pa,qa)
    ! h = (1-y^2) exp(y/2), U = (1-y^2)(1+0.2 sin 2y), W = (1-y^2) y
    pa = [1-y**2, -2*y, -2._rp, 0._rp]
    ea = [exp(0.5_rp*y), 0.5_rp*exp(0.5_rp*y), 0.25_rp*exp(0.5_rp*y), 0.125_rp*exp(0.5_rp*y)]
    h  = leib(pa,ea)
    ub = leib(pa,[1+0.2_rp*sin(2*y), 0.4_rp*cos(2*y), -0.8_rp*sin(2*y), -1.6_rp*cos(2*y)])
    wb = leib(pa,[y, 1._rp, 0._rp, 0._rp])
    k1 = a1**2 + b1**2
    ! wave 1
    call addterm(2, aa, aat, ph(0),ph(1),ph(2),.false.,a1,b1,x,z,uv,g,lap,ut)
    call addterm(1,-aa*a1/k1,-aat*a1/k1,ph(1),ph(2),ph(3),.true.,a1,b1,x,z,uv,g,lap,ut)
    call addterm(3,-aa*b1/k1,-aat*b1/k1,ph(1),ph(2),ph(3),.true.,a1,b1,x,z,uv,g,lap,ut)
    ! wave 2
    call addterm(1, bb*b2, bbt*b2,h(0),h(1),h(2),.false.,a2,b2,x,z,uv,g,lap,ut)
    call addterm(3,-bb*a2,-bbt*a2,h(0),h(1),h(2),.false.,a2,b2,x,z,uv,g,lap,ut)
    ! mean flow
    call addterm(1, cm, cmt,ub(0),ub(1),ub(2),.false.,0._rp,0._rp,x,z,uv,g,lap,ut)
    call addterm(3, dm, dmt,wb(0),wb(1),wb(2),.false.,0._rp,0._rp,x,z,uv,g,lap,ut)
    ! pressure
    s = [sin(1.5_rp*y), 1.5_rp*cos(1.5_rp*y), 0._rp, 0._rp]
    p = ee*s(0)*cos(a1*x+b2*z) + 0.1_rp*y**2
    gp(1) = -ee*s(0)*a1*sin(a1*x+b2*z)
    gp(2) =  ee*s(1)*cos(a1*x+b2*z) + 0.2_rp*y
    gp(3) = -ee*s(0)*b2*sin(a1*x+b2*z)
  end subroutine exact
  !
  ! exact scalar n at (x,y,z,t): value sv, gradient sg, Laplacian sl, ds/dt st
  subroutine exact_scalar(n,x,y,z,t,sv,sg,sl,st)
    integer , intent(in) :: n
    real(rp), intent(in) :: x,y,z,t
    real(rp), intent(out) :: sv,sg(3),sl,st
    real(rp) :: v3(3),g3(3,3),l3(3),t3(3),pw(0:3),y1(0:3),y2(0:3),y3(0:3),f,c1,c1t,c2,c2t,c3,c3t
    real(rp) :: m(3,3),r(3),ab(3),vb,vt

    f = 1 + 0.5_rp*(n-1)
    c1 = f*0.4_rp*(1+0.3_rp*sin(1.1_rp*t)); c1t = f*0.132_rp*cos(1.1_rp*t)
    c2 = f*0.3_rp*cos(0.9_rp*t);            c2t = -f*0.27_rp*sin(0.9_rp*t)
    c3 = f*0.5_rp*(1+0.2_rp*sin(0.7_rp*t)); c3t = f*0.07_rp*cos(0.7_rp*t)
    pw = [(1-y**2)**2, -4*y*(1-y**2), -4+12*y**2, 24*y]
    y1 = leib(pw,[cos(1.3_rp*y), -1.3_rp*sin(1.3_rp*y), -1.69_rp*cos(1.3_rp*y), 2.197_rp*sin(1.3_rp*y)])
    y2 = leib(pw,[exp(0.3_rp*y), 0.3_rp*exp(0.3_rp*y), 0.09_rp*exp(0.3_rp*y), 0.027_rp*exp(0.3_rp*y)])
    y3 = leib(pw,[sin(y+0.2_rp), cos(y+0.2_rp), -sin(y+0.2_rp), -cos(y+0.2_rp)])
    v3 = 0; g3 = 0; l3 = 0; t3 = 0
    ! cos(theta1 + 0.7) = cos(0.7) cos(theta1) - sin(0.7) sin(theta1)
    call addterm(1, c1*cos(0.7_rp), c1t*cos(0.7_rp),y1(0),y1(1),y1(2),.false.,a1,b1,x,z,v3,g3,l3,t3)
    call addterm(1,-c1*sin(0.7_rp),-c1t*sin(0.7_rp),y1(0),y1(1),y1(2),.true. ,a1,b1,x,z,v3,g3,l3,t3)
    call addterm(1, c2, c2t,y2(0),y2(1),y2(2),.true. ,a2,b2,x,z,v3,g3,l3,t3)
    call addterm(1, c3, c3t,y3(0),y3(1),y3(2),.false.,0._rp,0._rp,x,z,v3,g3,l3,t3)
    ! Sb = A + B y + C y^2 from the wall conditions (third condition: C = 0.3, or A = 0 for Neumann-Neumann)
    vb = sval_bot(n); vt = sval_top(n)
    m = 0; r = 0
    if(sbc_bot(n) == 'N') then
      m(1,:) = [0._rp, 1._rp, -2._rp]
    else
      m(1,:) = [1._rp,-1._rp,  1._rp]
    end if
    if(sbc_top(n) == 'N') then
      m(2,:) = [0._rp, 1._rp,  2._rp]
    else
      m(2,:) = [1._rp, 1._rp,  1._rp]
    end if
    r(1:2) = [vb,vt]
    if(sbc_bot(n) == 'N' .and. sbc_top(n) == 'N') then
      m(3,:) = [1._rp,0._rp,0._rp]; r(3) = 0
    else
      m(3,:) = [0._rp,0._rp,1._rp]; r(3) = 0.3_rp
    end if
    ab = solve3(m,r)
    sv = v3(1) + ab(1) + ab(2)*y + ab(3)*y**2
    sg = g3(1,:); sg(2) = sg(2) + ab(2) + 2*ab(3)*y
    sl = l3(1) + 2*ab(3)
    st = t3(1)
  end subroutine exact_scalar
  !
  pure function solve3(m,r) result(x)
    real(rp), intent(in) :: m(3,3),r(3)
    real(rp) :: x(3),d
    integer :: c
    real(rp) :: mc(3,3)
    d = det3(m)
    do c=1,3
      mc = m; mc(:,c) = r
      x(c) = det3(mc)/d
    end do
  end function solve3
  pure real(rp) function det3(m)
    real(rp), intent(in) :: m(3,3)
    det3 = m(1,1)*(m(2,2)*m(3,3)-m(2,3)*m(3,2)) - m(1,2)*(m(2,1)*m(3,3)-m(2,3)*m(3,1)) &
         + m(1,3)*(m(2,1)*m(3,2)-m(2,2)*m(3,1))
  end function det3
  !
  ! quantity q (Q_*) of the exact solution at time t, as spectral fields fu, fv, fw (Q_P and the scalar
  ! quantities of scalar isc: fu only)
  subroutine to_spectral(q,t,fu,fv,fw,isc)
    integer , intent(in) :: q
    integer , intent(in), optional :: isc
    real(rp), intent(in) :: t
    complex(rp), intent(inout) :: fu(:,:,:),fv(:,:,:),fw(:,:,:)
    integer :: c,i,j,k,nc,n
    real(rp) :: x,y,z,uv(3),g(3,3),lap(3),ut(3),gp(3),p,om(3),val(3),sv,sg(3),sl,st
    nc = 3
    if(q == Q_P .or. q >= Q_S) nc = 1
    do c=1,nc
      !$acc update self(ap)
      do k=1,nyA
        y = yg(yA_lo+k-1)
        do j=1,nzA
          z = (zA_lo+j-2)*lz/mz
          do i=1,mx
            x = (i-1)*lx/mx
            call exact(x,y,z,t,uv,g,lap,ut,gp,p)
            om = [g(3,2)-g(2,3), g(1,3)-g(3,1), g(2,1)-g(1,2)]
            select case(q)
            case(Q_VEL)
              val = uv
            case(Q_LAP)
              val = lap
            case(Q_UXW)
              val = cross(uv,om)
            case(Q_ADV)
              val = -matmul(g,uv)
            case(Q_FORCE)
              val = ut - cross(uv,om) + gp - visc*lap
              do n=1,nscal   ! the code adds -ri s gvec (buoyancy)
                if(ri(n) /= 0._rp) then
                  call exact_scalar(n,x,y,z,t,sv,sg,sl,st)
                  val = val + ri(n)*sv*gvec
                end if
              end do
            case(Q_S,Q_SADV,Q_SFORCE)
              call exact_scalar(isc,x,y,z,t,sv,sg,sl,st)
              select case(q)
              case(Q_S)
                val = sv
              case(Q_SADV)
                val = -dot_product(uv,sg) + ssrc_const(isc) + ssrc_u(isc)/ubulk*uv(1)
              case default
                val = st + dot_product(uv,sg) - diff(isc)*sl - ssrc_const(isc) - ssrc_u(isc)/ubulk*uv(1)
              end select
            case(Q_GRADP)
              val = gp
            case default
              val = p
            end select
            ap(i,j,k) = val(c)
          end do
        end do
      end do
      !$acc update device(ap)
      select case(c)
      case(1)
        call fwd_fft(ap,fu)
      case(2)
        call fwd_fft(ap,fv)
      case(3)
        call fwd_fft(ap,fw)
      end select
    end do
  end subroutine to_spectral
  !
  pure function cross(a,b) result(c)
    real(rp), intent(in) :: a(3),b(3)
    real(rp) :: c(3)
    c = [a(2)*b(3)-a(3)*b(2), a(3)*b(1)-a(1)*b(3), a(1)*b(2)-a(2)*b(1)]
  end function cross
  !
  ! body force of the manufactured solution, added to the momentum right-hand sides (mod_rk extra_forcing)
  subroutine mms_forcing(t,fu,fv,fw)
    real(rp), intent(in) :: t
    complex(rp), intent(inout) :: fu(:,:,:),fv(:,:,:),fw(:,:,:)
    integer :: i,j,k
    call to_spectral(Q_FORCE,t,f1,f2,f3)
    !$acc parallel loop collapse(3) default(present) async(1)
    do k=1,ny
      do j=1,nkz
        do i=1,nkx
          fu(i,j,k) = fu(i,j,k) + f1(i,j,k)
          fv(i,j,k) = fv(i,j,k) + f2(i,j,k)
          fw(i,j,k) = fw(i,j,k) + f3(i,j,k)
        end do
      end do
    end do
  end subroutine mms_forcing
  !
  subroutine mms_scalar_forcing(t,n,fs)
    real(rp), intent(in) :: t
    integer , intent(in) :: n
    complex(rp), intent(inout) :: fs(:,:,:)
    integer :: i,j,k
    call to_spectral(Q_SFORCE,t,f1,f2,f3,n)
    !$acc parallel loop collapse(3) default(present) async(1)
    do k=1,ny
      do j=1,nkz
        do i=1,nkx
          fs(i,j,k) = fs(i,j,k) + f1(i,j,k)
        end do
      end do
    end do
  end subroutine mms_scalar_forcing
  !
  ! max |a - b| / max |b| over the coefficients of rows k1..k2 (all ranks); skip(i,j): columns left out
  real(rp) function errmax(a,b,k1,k2,skip)
    complex(rp), intent(in) :: a(:,:,:),b(:,:,:)
    integer, intent(in) :: k1,k2
    logical, intent(in), optional :: skip(:,:)
    real(rp) :: e(2),eg(2)
    integer :: i,j,k,ierr
    !$acc update self(a,b)
    e = 0
    do k=k1,k2
      do j=1,nkz
        do i=1,nkx
          if(present(skip)) then
            if(skip(i,j)) cycle
          end if
          e(1) = max(e(1),abs(a(i,j,k)-b(i,j,k)))
          e(2) = max(e(2),abs(b(i,j,k)))
        end do
      end do
    end do
    call MPI_ALLREDUCE(e,eg,2,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
    errmax = eg(1)/max(eg(2),tiny(1._rp))
  end function errmax
end module mod_mms
!
program mms_test
  use mpi
  use mod_types
  use mod_param
  use mod_decomp
  use mod_compact, only: init_grid,init_compact,yg
  use mod_fields
  use mod_fft
  use mod_ysolve
  use mod_rk
  use mod_mms
  implicit none
  character(len=16) :: test = 'run'
  real(rp) :: tend = 0.5_rp
  namelist /mms/ test,tend
  complex(rp), allocatable, dimension(:,:,:) :: eu,ev,ew
  integer :: ierr,iunit,ios,n,nsteps,kc1,kc2
  real(rp) :: time,dtcur,dtn,vsave,e(3),dsave(nscal_max),rsave(nscal_max),es
  character(len=1) :: cs
  call MPI_INIT(ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD,myid,ierr)
  call read_input(myid)
  open(newunit=iunit,file='input.nml',status='old',action='read')
  read(iunit,nml=mms,iostat=ios)
  close(iunit)
  if(forcing /= 'cpg' .or. dpdx /= 0._rp) call abort_run('mms: use forcing = ''cpg'', dpdx = 0')
  call init_decomp
  call init_grid(ny,grid_type,grid_alpha,grid_file)
  call init_compact(ny,fd_order)
  call alloc_fields
  call init_fft(up)
  call init_ysolve
  if(visc_theta > 0._rp) call init_implicit
  if(ns_method == 'vv') call init_vv
  call init_rk
  call init_mms
  allocate(eu(nkx,nkz,ny),ev(nkx,nkz,ny),ew(nkx,nkz,ny))
  eu = 0; ev = 0; ew = 0
  !$acc enter data copyin(eu,ev,ew)
  ! core of the channel, |y| <= 0.5 (away from the wall closures of the compact schemes)
  kc1 = 1; kc2 = ny
  do while(yg(kc1) < -0.5_rp); kc1 = kc1+1; end do
  do while(yg(kc2) >  0.5_rp); kc2 = kc2-1; end do
  select case(trim(test))
  case('operators')
    ! (1) diffusion
    call to_spectral(Q_VEL,0._rp,u,v,w)
    call to_spectral(Q_LAP,0._rp,eu,ev,ew)
    nu = 0; nv = 0; nw = 0
    !$acc update device(nu,nv,nw)
    call add_lap(u,nu,1._rp,tmp1)
    call add_lap(v,nv,1._rp,tmp1)
    call add_lap(w,nw,1._rp,tmp1)
    e = [errmax(nu,eu,2,ny-1),errmax(nv,ev,2,ny-1),errmax(nw,ew,2,ny-1)]
    call report('diffusion   (D2 - k^2)   ',e)
    e = [errmax(nu,eu,kc1,kc2),errmax(nv,ev,kc1,kc2),errmax(nw,ew,kc1,kc2)]
    call report('diffusion   core |y|<.5  ',e)
    ! (2) nonlinear terms, viscosity, diffusivity and buoyancy switched off
    do n=1,nscal
      call to_spectral(Q_S,0._rp,s(:,:,:,n),tmp2,tmp2,n)
    end do
    vsave = visc; visc = 0._rp
    dsave = diff; diff = 0._rp
    rsave = ri; ri = 0._rp
    call nonlinear(.false.,dtn)
    visc = vsave; diff = dsave; ri = rsave
    do n=1,nscal
      call to_spectral(Q_SADV,0._rp,eu,ev,ew,n)
      e = [errmax(ns(:,:,:,n),eu,1,ny),errmax(ns(:,:,:,n),eu,2,ny-1),errmax(ns(:,:,:,n),eu,kc1,kc2)]
      write(cs,'(i1)') n
      call report('scalar '//cs//' adv. '//scal_form(1:10),e)
    end do
    if(trim(nl_form) == 'advective') then
      call to_spectral(Q_ADV,0._rp,eu,ev,ew)
    else
      call to_spectral(Q_UXW,0._rp,eu,ev,ew)
    end if
    e = [errmax(nu,eu,1,ny),errmax(nv,ev,1,ny),errmax(nw,ew,1,ny)]
    call report('nonlinear   '//nl_form(1:10)//'   ',e)
    ! (3) pressure step: r = u + grad P
    call to_spectral(Q_GRADP,0._rp,nu,nv,nw)
    call to_spectral(Q_VEL,0._rp,eu,ev,ew)
    u = eu + nu; v = ev + nv; w = ew + nw
    !$acc update device(u,v,w)
    call pressure_step(u,v,w)
    e = [errmax(u,eu,1,ny),errmax(v,ev,1,ny),errmax(w,ew,1,ny)]
    call report('pressure    velocity     ',e)
    e = [errmax(u,eu,kc1,kc2),errmax(v,ev,kc1,kc2),errmax(w,ew,kc1,kc2)]
    call report('pressure    core |y|<.5  ',e)
    call to_spectral(Q_P,0._rp,eu,ev,ew)
    !$acc update self(phi)
    e(1) = errmax(phi,eu,1,ny,ismean)
    e(2:3) = divergence_max(u,v,w,tmp1)
    call report('pressure    phi, div     ',e(1:2))
  case('run')
    call to_spectral(Q_VEL,0._rp,u,v,w)
    call pressure_step(u,v,w)   ! discrete projection of the initial field
    if(ns_method == 'vv') call vv_state_from_velocity(u,v,w)
    do n=1,nscal
      call to_spectral(Q_S,0._rp,s(:,:,:,n),tmp2,tmp2,n)
    end do
    extra_forcing => mms_forcing
    extra_scalar_forcing => mms_scalar_forcing
    nsteps = nint(tend/dt)
    time = 0._rp; dtcur = dt
    do n=1,nsteps
      call rk_step(time,dtcur)
    end do
    call to_spectral(Q_VEL,time,eu,ev,ew)
    e = [errmax(u,eu,2,ny-1),errmax(v,ev,2,ny-1),errmax(w,ew,2,ny-1)]
    es = divergence_max(u,v,w,tmp1)   ! collectives (also in errmax) are called by all ranks
    e = [errmax(u,eu,2,ny-1),errmax(v,ev,2,ny-1),errmax(w,ew,2,ny-1)]
    if(myid == 0) write(*,'(a,i5,a,es10.3,a,i6,a,f6.3,a,3es11.3,a,es9.2)',advance='no') ' MMS run ny ',ny, &
      ' dt ',dt,' steps ',nsteps,' t ',time,'  err u v w ',e,'  div ',es
    e = [errmax(u,eu,kc1,kc2),errmax(v,ev,kc1,kc2),errmax(w,ew,kc1,kc2)]
    if(myid == 0) write(*,'(a,3es11.3)') '  core ',e
    do n=1,nscal
      call to_spectral(Q_S,time,eu,ev,ew,n)
      es = errmax(s(:,:,:,n),eu,1,ny)
      if(myid == 0) write(*,'(a,i5,a,es10.3,a,i1,a,a,a,a1,a1,a,f4.2,a,es11.3)') ' MMS run ny ',ny,' dt ',dt, &
        '  scalar ',n,' ',scal_form(1:10),' bc ',sbc_bot(n),sbc_top(n),' theta ',scal_theta(n),'  err ',es
    end do
  case default
    call abort_run('mms: test must be operators or run')
  end select
  call end_fft
  call finalize_decomp
  call MPI_FINALIZE(ierr)
contains
  subroutine report(what,e)
    character(len=*), intent(in) :: what
    real(rp), intent(in) :: e(:)
    if(myid == 0) write(*,'(a,i5,2x,a,3es11.3)') ' MMS ny ',ny,what,e
  end subroutine report
end program mms_test
