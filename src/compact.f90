!
! wall-normal grid and collocated compact finite-difference operators
!
!   A1 f' = B1 f ,   A2 f'' = B2 f        (Eqs. (6)-(8) of Lenaers et al. 2014)
!
! The coefficients are computed directly on the (non-uniform) grid by requiring
! exactness for polynomials of the highest possible degree. Schemes:
!   fd_order = 4 : (3-4-3)       tridiagonal LHS, 3rd-order boundary closure
!   fd_order = 6 : (3-5-6-5-3)   tridiagonal LHS, 3rd/5th-order closures
! A1 and A2 are tridiagonal; B1 and B2 are banded with offsets in [-3,3].
!
module mod_compact
  use mod_types
  use mod_param, only: abort_run
  implicit none
  private
  public :: init_grid,init_compact,mat_band_apply
  integer, parameter, public :: nbw = 3        ! max. half-bandwidth of B1/B2
  real(rp), allocatable, public :: yg(:)       ! grid points y(1:ny), y(1) = -1, y(ny) = 1
  real(rp), allocatable, public :: a1(:,:),a2(:,:) ! (-1:1   ,ny)
  real(rp), allocatable, public :: b1(:,:),b2(:,:) ! (-nbw:nbw,ny)
  real(rp), allocatable, public :: t1w(:),t1d(:),t1u(:) ! Thomas factors of A1 (multiplier, 1/pivot, super-diag.)
  real(rp), allocatable, public :: t2w(:),t2d(:),t2u(:) ! Thomas factors of A2
  real(rp), allocatable, public :: c1(:),cn(:)     ! first and last row of D1 = A1^{-1} B1 (dense)
  integer , public :: nc1,ncn                      ! effective extent of c1 (from the bottom) and cn (from the top)
  real(rp), allocatable, public :: wq(:)           ! quadrature weights, int_{-1}^{1} f dy = sum wq*f
  real(rp), allocatable, public :: dyc(:)          ! local minimum grid spacing (for the CFL condition)
  integer , public :: kl1,ku1,kl2,ku2              ! actual bandwidths of B1 and B2
  real(rp), public :: lam2max                      ! spectral radius of D2 with Dirichlet BCs
  real(rp), public :: kappa1                       ! max. modified wavenumber of D1 (uniform grid)
  real(rp), allocatable, public :: b1m(:,:)        ! (-nbw:nbw,ny) compact d/dxi on the uniform grid xi
  real(rp), allocatable, public :: t1mw(:),t1md(:),t1mu(:) ! Thomas factors of its LHS
  real(rp), allocatable, public :: rxi(:)          ! 1/(dy/dxi): D1m = diag(rxi) D_xi (metric first derivative)
  real(rp), public :: ew1(6),ewn(6)                ! explicit one-sided 6-point d/dy at y = -1 (points 1..6)
                                                   ! and y = +1 (points ny-5..ny)
contains
  subroutine init_grid(ny,grid_type,alpha,grid_file)
    implicit none
    integer , intent(in) :: ny
    character(len=*), intent(in) :: grid_type,grid_file
    real(rp), intent(in) :: alpha
    integer :: j,iunit,ios
    real(rp) :: xi
    allocate(yg(ny))
    select case(trim(grid_type))
    case('tanh')
      if(alpha <= 0._rp) call abort_run('grid_alpha must be > 0 for a tanh grid')
      do j=1,ny
        xi = (j-1._rp)/(ny-1._rp)
        yg(j) = tanh(alpha*(xi-0.5_rp))/tanh(0.5_rp*alpha)
      end do
    case('uniform')
      do j=1,ny
        yg(j) = -1._rp + 2._rp*(j-1._rp)/(ny-1._rp)
      end do
    case('file')
      open(newunit=iunit,file=trim(grid_file),status='old',action='read',iostat=ios)
      if(ios /= 0) call abort_run('cannot open grid file '//trim(grid_file))
      do j=1,ny
        read(iunit,*,iostat=ios) yg(j)
        if(ios /= 0) call abort_run('grid file too short')
      end do
      close(iunit)
    case default
      call abort_run('unknown grid_type '//trim(grid_type))
    end select
    yg(1) = -1._rp; yg(ny) = 1._rp
    do j=1,ny-1
      if(yg(j+1) <= yg(j)) call abort_run('grid points must be strictly increasing')
    end do
  end subroutine init_grid
  !
  subroutine init_compact(ny,order)
    implicit none
    integer, intent(in) :: ny,order
    integer :: i,j,l
    real(rp), allocatable :: z(:)
    real(rp) :: cmax
    !
    allocate(a1(-1:1,ny),a2(-1:1,ny),b1(-nbw:nbw,ny),b2(-nbw:nbw,ny))
    a1(:,:) = 0._rp; a2(:,:) = 0._rp; b1(:,:) = 0._rp; b2(:,:) = 0._rp
    do i=1,ny
      call row_coeffs(i,ny,order,1,a1(:,i),b1(:,i))
      call row_coeffs(i,ny,order,2,a2(:,i),b2(:,i))
    end do
    kl1 = 0; ku1 = 0; kl2 = 0; ku2 = 0
    do i=1,ny
      do l=-nbw,nbw
        if(b1(l,i) /= 0._rp) then
          kl1 = max(kl1,-l); ku1 = max(ku1,l)
        end if
        if(b2(l,i) /= 0._rp) then
          kl2 = max(kl2,-l); ku2 = max(ku2,l)
        end if
      end do
    end do
    !
    ! Thomas factorisation of the (k-independent) tridiagonal A1 and A2
    !
    allocate(t1w(ny),t1d(ny),t1u(ny),t2w(ny),t2d(ny),t2u(ny))
    call thomas_factor(ny,a1,t1w,t1d,t1u)
    call thomas_factor(ny,a2,t2w,t2d,t2u)
    !
    ! dense first and last rows of D1: c1 = B1^T A1^{-T} e_1
    !
    allocate(c1(ny),cn(ny),z(ny))
    call tri_transpose_solve(ny,a1,1 ,z)
    c1(:) = 0._rp
    do i=1,ny
      do l=-nbw,nbw
        j = i+l
        if(j >= 1 .and. j <= ny) c1(j) = c1(j) + z(i)*b1(l,i)
      end do
    end do
    call tri_transpose_solve(ny,a1,ny,z)
    cn(:) = 0._rp
    do i=1,ny
      do l=-nbw,nbw
        j = i+l
        if(j >= 1 .and. j <= ny) cn(j) = cn(j) + z(i)*b1(l,i)
      end do
    end do
    !
    ! truncate the (geometrically decaying) rows below round-off
    !
    cmax = maxval(abs(c1)); nc1 = ny
    do j=ny,1,-1
      if(abs(c1(j)) > 1.e-17_rp*cmax) exit
      nc1 = j-1
    end do
    cmax = maxval(abs(cn)); ncn = ny
    do j=1,ny
      if(abs(cn(j)) > 1.e-17_rp*cmax) exit
      ncn = ny-j
    end do
    nc1 = max(nc1,2); ncn = max(ncn,2)
    !
    call init_metric_derivative(ny,order)
    call init_quadrature(ny,order)
    allocate(dyc(ny))
    dyc(1)  = yg(2)-yg(1)
    dyc(ny) = yg(ny)-yg(ny-1)
    do j=2,ny-1
      dyc(j) = min(yg(j+1)-yg(j),yg(j)-yg(j-1))
    end do
    lam2max = spectral_radius_d2(ny)
    call wall_derivative(yg(1:6),yg(1),ew1)
    call wall_derivative(yg(ny-5:ny),yg(ny),ewn)
    if(order == 4) then
      kappa1 = 1.74_rp
    else
      kappa1 = 2.0_rp
    end if
  end subroutine init_compact
  !
  subroutine init_metric_derivative(ny,order)
    !
    ! first derivative through the uniform computational grid xi_j = -1 + 2(j-1)/(ny-1):
    ! D1m = diag(1/y_xi) A1u^{-1} B1u, with the compact scheme (A1u, B1u) of the uniform grid and
    ! y_xi = A1u^{-1} B1u y. Its interior part is skew-adjoint in the weights y_xi*dxi, which the direct
    ! compact D1 on a stretched grid is not; used in the skew-symmetric scalar advection.
    !
    implicit none
    integer, intent(in) :: ny,order
    real(rp), allocatable :: ysave(:),a1m(:,:),g(:)
    integer :: i
    allocate(b1m(-nbw:nbw,ny),a1m(-1:1,ny),t1mw(ny),t1md(ny),t1mu(ny),rxi(ny),g(ny))
    allocate(ysave,source=yg)
    do i=1,ny
      yg(i) = -1._rp + 2._rp*(i-1)/real(ny-1,rp)
    end do
    do i=1,ny
      call row_coeffs(i,ny,order,1,a1m(:,i),b1m(:,i))
    end do
    yg(:) = ysave(:)
    call thomas_factor(ny,a1m,t1mw,t1md,t1mu)
    call mat_band_apply(ny,nbw,b1m,yg,g)
    do i=2,ny
      g(i) = g(i) - t1mw(i)*g(i-1)
    end do
    g(ny) = g(ny)*t1md(ny)
    do i=ny-1,1,-1
      g(i) = (g(i) - t1mu(i)*g(i+1))*t1md(i)
    end do
    if(any(g <= 0._rp)) call abort_run('grid is not monotonic: dy/dxi <= 0')
    rxi(:) = 1._rp/g(:)
  end subroutine init_metric_derivative
  !
  subroutine wall_derivative(yp,x0,wts)
    ! weights of the derivative at x0 of the Lagrange polynomial through the points yp(1:6)
    implicit none
    real(rp), intent(in)  :: yp(6),x0
    real(rp), intent(out) :: wts(6)
    integer :: a,b,c,d
    real(rp) :: den,s,p
    do a=1,6
      den = 1._rp
      do b=1,6
        if(b /= a) den = den*(yp(a)-yp(b))
      end do
      s = 0._rp
      do c=1,6
        if(c == a) cycle
        p = 1._rp
        do d=1,6
          if(d /= a .and. d /= c) p = p*(x0-yp(d))
        end do
        s = s + p
      end do
      wts(a) = s/den
    end do
  end subroutine wall_derivative
  !
  subroutine stencil(i,ny,order,der,nl,lo,nr,ro)
    !
    ! offsets of the LHS (lo) and RHS (ro) stencils of row i
    !
    implicit none
    integer, intent(in)  :: i,ny,order,der
    integer, intent(out) :: nl,lo(3),nr,ro(7)
    integer :: k
    lo(:) = 0; ro(:) = 0
    if(i == 1) then
      nl = 2; lo(1:2) = [0,1]
      if(der == 1) then
        nr = 3; ro(1:3) = [0,1,2]
      else
        nr = 4; ro(1:4) = [0,1,2,3]
      end if
    else if(i == ny) then
      nl = 2; lo(1:2) = [-1,0]
      if(der == 1) then
        nr = 3; ro(1:3) = [-2,-1,0]
      else
        nr = 4; ro(1:4) = [-3,-2,-1,0]
      end if
    else
      nl = 3; lo(1:3) = [-1,0,1]
      if(order == 4) then
        nr = 3; ro(1:3) = [-1,0,1]
      else
        if(i == 2) then
          if(der == 1) then
            nr = 4; ro(1:4) = [-1,0,1,2]
          else
            nr = 5; ro(1:5) = [-1,0,1,2,3]
          end if
        else if(i == ny-1) then
          if(der == 1) then
            nr = 4; ro(1:4) = [-2,-1,0,1]
          else
            nr = 5; ro(1:5) = [-3,-2,-1,0,1]
          end if
        else
          nr = 5; ro(1:5) = [(k,k=-2,2)]
        end if
      end if
    end if
  end subroutine stencil
  !
  subroutine row_coeffs(i,ny,order,der,a,b)
    !
    ! sum_j a_j f^(der)(y_{i+j}) = sum_k b_k f(y_{i+k}), a_0 = 1, exact for polynomials
    ! of degree 0..nu-1 (nu = number of unknown coefficients)
    !
    implicit none
    integer , intent(in) :: i,ny,order,der
    real(rp), intent(out) :: a(-1:1),b(-nbw:nbw)
    integer :: nl,lo(3),nr,ro(7),nu,m,c,ia(2),na
    real(rp) :: mat(9,9),rhs(9),h,x
    call stencil(i,ny,order,der,nl,lo,nr,ro)
    h = (yg(min(i+1,ny))-yg(max(i-1,1)))
    na = 0
    do c=1,nl
      if(lo(c) /= 0) then
        na = na+1; ia(na) = lo(c)
      end if
    end do
    nu = na+nr
    do m=0,nu-1
      do c=1,na
        x = (yg(i+ia(c))-yg(i))/h
        mat(m+1,c) = dpoly(m,der,x)
      end do
      do c=1,nr
        x = (yg(i+ro(c))-yg(i))/h
        mat(m+1,na+c) = -x**m
      end do
      rhs(m+1) = -dpoly(m,der,0._rp)
    end do
    call dense_solve(nu,mat(1:nu,1:nu),rhs(1:nu))
    a(:) = 0._rp; b(:) = 0._rp
    a(0) = 1._rp
    do c=1,na
      a(ia(c)) = rhs(c)
    end do
    do c=1,nr
      b(ro(c)) = rhs(na+c)/h**der
    end do
  end subroutine row_coeffs
  !
  pure real(rp) function dpoly(m,der,x)
    ! der-th derivative of x**m
    implicit none
    integer , intent(in) :: m,der
    real(rp), intent(in) :: x
    integer :: q
    real(rp) :: f
    if(m < der) then
      dpoly = 0._rp
    else
      f = 1._rp
      do q=m-der+1,m
        f = f*q
      end do
      if(m == der) then
        dpoly = f
      else
        dpoly = f*x**(m-der)
      end if
    end if
  end function dpoly
  !
  subroutine dense_solve(n,mat,rhs)
    ! Gaussian elimination with partial pivoting (small systems)
    implicit none
    integer , intent(in) :: n
    real(rp), intent(inout) :: mat(n,n),rhs(n)
    integer :: i,k,p
    real(rp) :: f,tmp(n),t
    do k=1,n
      p = k-1+maxloc(abs(mat(k:n,k)),1)
      if(abs(mat(p,k)) == 0._rp) call abort_run('singular matrix computing compact coefficients')
      if(p /= k) then
        tmp(:) = mat(k,:); mat(k,:) = mat(p,:); mat(p,:) = tmp(:)
        t = rhs(k); rhs(k) = rhs(p); rhs(p) = t
      end if
      do i=k+1,n
        f = mat(i,k)/mat(k,k)
        mat(i,k:n) = mat(i,k:n) - f*mat(k,k:n)
        rhs(i) = rhs(i) - f*rhs(k)
      end do
    end do
    do k=n,1,-1
      rhs(k) = (rhs(k) - sum(mat(k,k+1:n)*rhs(k+1:n)))/mat(k,k)
    end do
  end subroutine dense_solve
  !
  subroutine thomas_factor(n,a,w,d,u)
    implicit none
    integer , intent(in) :: n
    real(rp), intent(in) :: a(-1:1,n)
    real(rp), intent(out) :: w(n),d(n),u(n)
    integer :: i
    real(rp) :: piv
    w(1) = 0._rp
    piv  = a(0,1)
    d(1) = 1._rp/piv
    u(:) = a(1,:)
    do i=2,n
      w(i) = a(-1,i)*d(i-1)
      piv  = a(0,i) - w(i)*a(1,i-1)
      if(abs(piv) < 1.e-12_rp) call abort_run('zero pivot in compact LHS matrix')
      d(i) = 1._rp/piv
    end do
  end subroutine thomas_factor
  !
  subroutine tri_transpose_solve(n,a,irow,z)
    ! solve A^T z = e_irow for tridiagonal A stored as a(-1:1,n)
    implicit none
    integer , intent(in) :: n,irow
    real(rp), intent(in) :: a(-1:1,n)
    real(rp), intent(out) :: z(n)
    real(rp) :: lo(n),di(n),up(n),c(n),r(n),den
    integer :: i
    ! (A^T)_{i,i-1} = A_{i-1,i} = a(1,i-1); (A^T)_{i,i+1} = A_{i+1,i} = a(-1,i+1)
    do i=1,n
      di(i) = a(0,i)
      lo(i) = 0._rp; up(i) = 0._rp
      if(i > 1) lo(i) = a(1,i-1)
      if(i < n) up(i) = a(-1,i+1)
    end do
    r(:) = 0._rp; r(irow) = 1._rp
    c(1) = up(1)/di(1)
    r(1) = r(1)/di(1)
    do i=2,n
      den  = di(i)-lo(i)*c(i-1)
      c(i) = up(i)/den
      r(i) = (r(i)-lo(i)*r(i-1))/den
    end do
    z(n) = r(n)
    do i=n-1,1,-1
      z(i) = r(i)-c(i)*z(i+1)
    end do
  end subroutine tri_transpose_solve
  !
  subroutine init_quadrature(ny,order)
    !
    ! composite interpolatory quadrature: on each interval [y_j,y_j+1] integrate
    ! the Lagrange polynomial through `order` neighbouring points
    !
    implicit none
    integer, intent(in) :: ny,order
    integer :: j,s,m,l,np
    real(rp) :: mat(8,8),rhs(8),h,xa,xb
    allocate(wq(ny))
    wq(:) = 0._rp
    np = min(order,ny)
    do j=1,ny-1
      s = j-np/2+1
      s = max(1,min(s,ny-np+1))
      h = yg(j+1)-yg(j)
      xa = 0._rp; xb = 1._rp
      do m=0,np-1
        do l=1,np
          mat(m+1,l) = ((yg(s+l-1)-yg(j))/h)**m
        end do
        rhs(m+1) = (xb**(m+1)-xa**(m+1))/(m+1)
      end do
      call dense_solve(np,mat(1:np,1:np),rhs(1:np))
      do l=1,np
        wq(s+l-1) = wq(s+l-1) + rhs(l)*h
      end do
    end do
  end subroutine init_quadrature
  !
  subroutine mat_band_apply(n,kb,b,f,g)
    ! g = B f for a banded matrix B(-kb:kb,n)
    implicit none
    integer , intent(in) :: n,kb
    real(rp), intent(in) :: b(-kb:kb,n),f(n)
    real(rp), intent(out) :: g(n)
    integer :: i,l
    do i=1,n
      g(i) = 0._rp
      do l=-kb,kb
        if(i+l >= 1 .and. i+l <= n) g(i) = g(i) + b(l,i)*f(i+l)
      end do
    end do
  end subroutine mat_band_apply
  !
  real(rp) function spectral_radius_d2(ny)
    !
    ! power iteration for the largest |eigenvalue| of D2 = A2^{-1} B2 with homogeneous Dirichlet BCs
    !
    implicit none
    integer, intent(in) :: ny
    real(rp) :: f(ny),g(ny),nrm,lam
    integer :: it,j
    do j=1,ny
      f(j) = sin(1.234_rp*j) + 0.5_rp*cos(7.77_rp*j*j)
    end do
    f(1) = 0._rp; f(ny) = 0._rp
    lam = 0._rp
    do it=1,400
      nrm = sqrt(sum(f**2))
      f(:) = f(:)/nrm
      call mat_band_apply(ny,nbw,b2,f,g)
      call thomas_solve(ny,t2w,t2d,t2u,g)
      g(1) = 0._rp; g(ny) = 0._rp
      lam = sqrt(sum(g**2))
      f(:) = g(:)
    end do
    spectral_radius_d2 = 1.05_rp*lam
  end function spectral_radius_d2
  !
  subroutine thomas_solve(n,w,d,u,r)
    implicit none
    integer , intent(in) :: n
    real(rp), intent(in) :: w(n),d(n),u(n)
    real(rp), intent(inout) :: r(n)
    integer :: i
    do i=2,n
      r(i) = r(i) - w(i)*r(i-1)
    end do
    r(n) = r(n)*d(n)
    do i=n-1,1,-1
      r(i) = (r(i) - u(i)*r(i+1))*d(i)
    end do
  end subroutine thomas_solve
end module mod_compact
