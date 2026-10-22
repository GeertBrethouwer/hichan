!
! wall-normal operations on spectral z-pencils f(nkx,nkz,ny): compact derivatives,
! the influence-matrix pressure/velocity solve, and the scalar Helmholtz solves
!
! All column operations are done by one GPU thread per wavenumber pair (kx,kz) that
! marches sequentially in y; consecutive threads access consecutive kx (coalesced).
! Only banded matrices (from the compact scheme) are factorised, on the fly, without
! pivoting; the dense operators D1 = A1^{-1}B1 and D2 = A2^{-1}B2 are never formed
! (Sec. 4 of Lenaers et al. 2014).
!
! Pressure step (explicit viscous terms, Appendix B.1), with phi = beta*dt*p:
!     u = rx - i kx phi,  w = rz - i kz phi,  v = ry - D1 phi   (interior points)
!     u = v = w = 0 at the walls,   i kx u + i kz w + D1 v = 0   at ALL points.
! The Poisson operator is factorised as D1^2 - k^2 = (D1 + k)(D1 - k), and with
! psi = (D1 - k) phi the problem becomes two first-order banded problems
!     (B1 + k A1) psi = A1 g      (row 1 replaced by psi(-1) = mu),
!     (B1 - k A1) phi = A1 psi    (row N replaced by phi(+1) = b)
! (Sec. 4.1, Appendix A). g equals the Poisson right-hand side at interior points,
! including the boundary-error terms sigma (Eqs. (21), (B.2)), and is free at the walls.
! All unknowns that the banded solves cannot impose (mu, b, the two wall values of g,
! and the two wall corrections sigma_b, sigma_t of v) are found by superposition
! with six homogeneous solutions (influence matrix, Eqs. (23)-(27), (B.9)-(B.22)),
! whose 6x6 matrix depends only on k and is inverted once at start-up. The discrete
! divergence then vanishes at every grid point (up to round-off).
!
module mod_ysolve
  use mod_types
  use mod_param , only: nxh,nz,ny,lx,lz,fd_order,abort_run,vv_neumann
  use mod_decomp, only: nkx,nkz,kx_lo,kz_lo
  use mod_compact, only: nbw,ca1 => a1,cb1 => b1,ca2 => a2,cb2 => b2,ct1w => t1w,ct1d => t1d,ct1u => t1u, &
                         ct2w => t2w,ct2d => t2d,ct2u => t2u,cc1 => c1,ccn => cn,cnc1 => nc1,cncn => ncn, &
                         cwq => wq,ckl1 => kl1,cku1 => ku1,ckl2 => kl2,cku2 => ku2,cew1 => ew1,cewn => ewn
  implicit none
  private
  public :: init_ysolve,deriv1,add_lap,pressure_step,scalar_solve,cfr_correct,divergence_max, &
            wall_gradient,mean_owner,i00,j00,kxv,kzv,kmod,ismean, &
            init_implicit,set_implicit_guess,pressure_step_implicit,cfr_correct_implicit,mean_gradient_max, &
            init_vv,vv_state_from_velocity,vv_rhs_terms,vv_step,vv_cfr
  real(rp), allocatable, public :: a1(:,:),b1(:,:),a2(:,:),b2(:,:)
  real(rp), allocatable :: t1w(:),t1d(:),t1u(:),t2w(:),t2d(:),t2u(:)
  real(rp), allocatable, public :: c1(:),cn(:),wq(:)
  integer :: nc1,ncn,klq,kuq,klh,kuh
  real(rp), allocatable :: kxv(:),kzv(:),kmod(:,:)
  logical , allocatable :: ismean(:,:),isnyq(:)
  real(rp), allocatable :: minv(:,:,:)
  real(rp), allocatable :: uf(:,:,:,:)
  complex(rp), allocatable :: psi(:,:,:)
  complex(rp), allocatable, public :: phi(:,:,:)
  complex(rp), allocatable :: sx(:,:,:)
  logical :: mean_owner
  integer :: i00 = -1, j00 = -1
  !
  ! implicit viscous terms (Appendix B.2): per RK substep q, cached for c_q = theta*beta_q*dt/Re
  !
  real(rp), allocatable :: minv_i(:,:,:,:)   ! (36,nkx,nkz,3) inverse 6x6 influence matrices
  real(rp), allocatable :: hinv_i(:,:,:,:)   ! (4 ,nkx,nkz,3) inverse 2x2 wall corrections of the Helmholtz solves
  real(rp) :: c_cached(3) = -1._rp
  complex(rp), allocatable :: vit(:,:,:),cvec(:,:,:),tw1(:,:,:)
  !
  ! velocity-vorticity method ('vv')
  !
  complex(rp), allocatable, public :: phv(:,:,:),om(:,:,:)  ! phi = lap(v) and omega_y (mean columns: W00, U00)
  complex(rp), allocatable :: wa(:,:,:),wb(:,:,:)
  real(rp), allocatable :: hinv_c(:,:,:,:),ginv_v(:,:,:,:),hinv_v(:,:,:)
  real(rp) :: c_cached_vv(3) = -1._rp
  real(rp), allocatable :: dw1(:),dwn(:)    ! wall-derivative functionals used for dv/dy = 0
  integer :: ndw1,ndwn
contains
  subroutine init_ysolve
    implicit none
    integer :: i,j,ig,jg,m
    allocate(a1,source=ca1); allocate(b1,source=cb1)
    allocate(a2,source=ca2); allocate(b2,source=cb2)
    allocate(t1w,source=ct1w); allocate(t1d,source=ct1d); allocate(t1u,source=ct1u)
    allocate(t2w,source=ct2w); allocate(t2d,source=ct2d); allocate(t2u,source=ct2u)
    allocate(c1,source=cc1); allocate(cn,source=ccn); allocate(wq,source=cwq)
    nc1 = cnc1; ncn = cncn
    klq = max(ckl1,1); kuq = max(cku1,1)
    klh = max(ckl2,1); kuh = max(cku2,1)
    !$acc enter data copyin(a1,b1,a2,b2,t1w,t1d,t1u,t2w,t2d,t2u,c1,cn,wq)
    allocate(kxv(nkx),kzv(nkz),kmod(nkx,nkz),ismean(nkx,nkz),isnyq(nkz))
    do i=1,nkx
      ig = kx_lo+i-1
      kxv(i) = 2*pi/lx*(ig-1)
    end do
    do j=1,nkz
      jg = kz_lo+j-1
      if(jg <= nz/2) then
        m = jg-1
      else
        m = jg-1-nz
      end if
      kzv(j) = 2*pi/lz*m
      isnyq(j) = (jg == nz/2+1)
    end do
    mean_owner = .false.
    do j=1,nkz
      do i=1,nkx
        kmod(i,j) = sqrt(kxv(i)**2+kzv(j)**2)
        ismean(i,j) = (kx_lo+i-1 == 1 .and. kz_lo+j-1 == 1)
        if(ismean(i,j)) then
          mean_owner = .true.; i00 = i; j00 = j
        end if
      end do
    end do
    !$acc enter data copyin(kxv,kzv,kmod,ismean,isnyq)
    allocate(minv(36,nkx,nkz),uf(nkx,nkz,ny,0:nbw),psi(nkx,nkz,ny),phi(nkx,nkz,ny),sx(nkx,nkz,ny))
    !$acc enter data create(minv,uf,psi,phi,sx)
    call init_influence_matrix
  end subroutine init_ysolve
  !
  ! ---------------------------------------------------------------------------
  ! column kernels (device routines)
  ! ---------------------------------------------------------------------------
  !
  subroutine bsolve(n1,n2,n,i,j,kl,ku,ca,cb,a,b,rep1,repn,x,uf)
    !
    ! solves (ca*A + cb*B) x = rhs for column (i,j), rhs given in x; A is tridiagonal,
    ! B banded with kl lower and ku upper diagonals; if rep1 (repn) the first (last)
    ! row is replaced by the identity row. Banded LU without pivoting; the U factor
    ! is stored in uf.
    !
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n2,n,i,j,kl,ku
    real(rp), intent(in) :: ca,cb
    real(rp), intent(in) :: a(-1:1,n),b(-nbw:nbw,n)
    logical , intent(in) :: rep1,repn
    complex(rp), intent(inout) :: x(n1,n2,n)
    real(rp), intent(inout) :: uf(n1,n2,n,0:nbw)
    real(rp) :: row(-nbw:nbw),f
    complex(rp) :: rr
    integer :: r,o,l,q
    do r=1,n
      do o=-nbw,nbw
        row(o) = 0._rp
      end do
      if((r == 1 .and. rep1) .or. (r == n .and. repn)) then
        row(0) = 1._rp
      else
        do o=-1,1
          row(o) = ca*a(o,r)
        end do
        do o=-kl,ku
          row(o) = row(o) + cb*b(o,r)
        end do
      end if
      rr = x(i,j,r)
      do l=max(1,r-kl),r-1
        o = l-r
        if(row(o) /= 0._rp) then
          f = row(o)/uf(i,j,l,0)
          do q=1,ku
            if(o+q <= ku) row(o+q) = row(o+q) - f*uf(i,j,l,q)
          end do
          rr = rr - f*x(i,j,l)
        end if
      end do
      do q=0,ku
        uf(i,j,r,q) = row(q)
      end do
      x(i,j,r) = rr
    end do
    x(i,j,n) = x(i,j,n)/uf(i,j,n,0)
    do r=n-1,1,-1
      rr = x(i,j,r)
      do q=1,min(ku,n-r)
        rr = rr - uf(i,j,r,q)*x(i,j,r+q)
      end do
      x(i,j,r) = rr/uf(i,j,r,0)
    end do
  end subroutine bsolve
  !
  pure complex(rp) function bmul(n1,n2,n,i,j,r,ca,cb,a,b,x)
    ! row r of (ca*A + cb*B) x
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n2,n,i,j,r
    real(rp), intent(in) :: ca,cb,a(-1:1,n),b(-nbw:nbw,n)
    complex(rp), intent(in) :: x(n1,n2,n)
    integer :: o
    complex(rp) :: s
    s = (0._rp,0._rp)
    do o=max(-nbw,1-r),min(nbw,n-r)
      if(abs(o) <= 1) then
        s = s + (ca*a(o,r)+cb*b(o,r))*x(i,j,r+o)
      else
        s = s + cb*b(o,r)*x(i,j,r+o)
      end if
    end do
    bmul = s
  end function bmul
  !
  pure real(rp) function bcol(n,r,c,ca,cb,a,b)
    ! entry (r,c) of ca*A + cb*B
    !$acc routine seq
    implicit none
    integer , intent(in) :: n,r,c
    real(rp), intent(in) :: ca,cb,a(-1:1,n),b(-nbw:nbw,n)
    integer :: o
    o = c-r
    bcol = 0._rp
    if(abs(o) <= nbw) bcol = cb*b(o,r)
    if(abs(o) <= 1  ) bcol = bcol + ca*a(o,r)
  end function bcol
  !
  subroutine pres_rhs(n1,n2,n,i,j,kx,kz,kk,a1,b1,u,v,w,x)
    ! A1 (i kx rx + i kz rz) + B1 ry, all rows
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n2,n,i,j
    real(rp), intent(in) :: kx,kz,kk,a1(-1:1,n),b1(-nbw:nbw,n)
    complex(rp), intent(in) :: u(n1,n2,n),v(n1,n2,n),w(n1,n2,n)
    complex(rp), intent(inout) :: x(n1,n2,n)
    integer :: r,o
    complex(rp) :: s
    do r=1,n
      s = (0._rp,0._rp)
      do o=max(-nbw,1-r),min(nbw,n-r)
        s = s + b1(o,r)*v(i,j,r+o)
        if(abs(o) <= 1) s = s + a1(o,r)*cmplx(0._rp,1._rp,rp)*(kx*u(i,j,r+o)+kz*w(i,j,r+o))
      end do
      x(i,j,r) = s
    end do
  end subroutine pres_rhs
  !
  subroutine pres_two_solves(n1,n2,n,i,j,kk,klq,kuq,a1,b1,mu,bt,psi,phi,uf,apsn)
    !
    ! given the full right-hand side A1 g in psi: psi <- solution of (B1+kA1) psi = A1 g (row 1: mu),
    ! phi <- solution of (B1-kA1) phi = A1 psi (row N: bt); apsn = (A1 psi)_N
    !
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n2,n,i,j,klq,kuq
    real(rp), intent(in) :: kk,a1(-1:1,n),b1(-nbw:nbw,n)
    complex(rp), intent(in) :: mu,bt
    complex(rp), intent(inout) :: psi(n1,n2,n),phi(n1,n2,n)
    real(rp), intent(inout) :: uf(n1,n2,n,0:nbw)
    complex(rp), intent(out) :: apsn
    integer :: r,o
    complex(rp) :: s
    psi(i,j,1) = mu
    call bsolve(n1,n2,n,i,j,klq,kuq,kk,1._rp,a1,b1,.true.,.false.,psi,uf)
    do r=1,n
      s = (0._rp,0._rp)
      do o=max(-1,1-r),min(1,n-r)
        s = s + a1(o,r)*psi(i,j,r+o)
      end do
      phi(i,j,r) = s
    end do
    apsn = phi(i,j,n)
    phi(i,j,n) = bt
    call bsolve(n1,n2,n,i,j,klq,kuq,-kk,1._rp,a1,b1,.false.,.true.,phi,uf)
  end subroutine pres_two_solves
  !
  subroutine pres_conds(n1,n2,n,i,j,kk,a1,b1,c1,cn,nc1,ncn,v,psi,phi,rhs1,apsn,sb,st,cc)
    !
    ! the six conditions that the superposed solution has to satisfy
    !
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n2,n,i,j,nc1,ncn
    real(rp), intent(in) :: kk,a1(-1:1,n),b1(-nbw:nbw,n),c1(n),cn(n)
    complex(rp), intent(in) :: v(n1,n2,n),psi(n1,n2,n),phi(n1,n2,n),rhs1,apsn,sb,st
    complex(rp), intent(out) :: cc(6)
    integer :: r
    complex(rp) :: s5,s6
    ! row 1 of (B1+kA1) psi = A1 g, and row N of (B1-kA1) phi = A1 psi
    cc(1) = bmul(n1,n2,n,i,j,1,kk,1._rp,a1,b1,psi) - rhs1
    cc(2) = bmul(n1,n2,n,i,j,n,-kk,1._rp,a1,b1,phi) - apsn
    ! wall corrections sigma_b,t = values of (ry - D1 phi) at the walls, D1 phi = psi + k phi
    cc(3) = sb - (v(i,j,1)-psi(i,j,1)-kk*phi(i,j,1))
    cc(4) = st - (v(i,j,n)-psi(i,j,n)-kk*phi(i,j,n))
    ! continuity at the walls: (D1 v)(+-1) = 0, with v = 0 at the walls
    s5 = (0._rp,0._rp); s6 = (0._rp,0._rp)
    do r=2,min(nc1,n-1)
      s5 = s5 + c1(r)*(v(i,j,r)-psi(i,j,r)-kk*phi(i,j,r))
    end do
    do r=max(n-ncn+1,2),n-1
      s6 = s6 + cn(r)*(v(i,j,r)-psi(i,j,r)-kk*phi(i,j,r))
    end do
    cc(5) = s5; cc(6) = s6
  end subroutine pres_conds
  !
  subroutine add_sources(n1,n2,n,i,j,a1,b1,sg,xs,x)
    ! x <- x + the right-hand sides of the homogeneous problems weighted by xs(1:4);
    ! sg = -1: wall corrections sigma of v (explicit), +1: delta of (B.32) (implicit)
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n2,n,i,j
    real(rp), intent(in) :: a1(-1:1,n),b1(-nbw:nbw,n),sg
    complex(rp), intent(in) :: xs(6)
    complex(rp), intent(inout) :: x(n1,n2,n)
    integer :: r
    do r=1,min(nbw+1,n)
      x(i,j,r) = x(i,j,r) + sg*xs(1)*bcol(n,r,1,0._rp,1._rp,a1,b1) + xs(3)*bcol(n,r,1,1._rp,0._rp,a1,b1)
    end do
    do r=max(n-nbw,1),n
      x(i,j,r) = x(i,j,r) + sg*xs(2)*bcol(n,r,n,0._rp,1._rp,a1,b1) + xs(4)*bcol(n,r,n,1._rp,0._rp,a1,b1)
    end do
  end subroutine add_sources
  !
  ! ---------------------------------------------------------------------------
  ! batched operations
  ! ---------------------------------------------------------------------------
  !
  subroutine init_influence_matrix
    !
    ! 6x6 influence matrix (inverted) for every wavenumber pair
    !
    implicit none
    integer :: i,j,m,r,q,p,l
    real(rp) :: kk,mat(6,6),inv(6,6),f,t
    complex(rp) :: cc(6),xs(6),apsn,rhs1,mu,bt
    !$acc parallel loop gang vector collapse(2) default(present) private(mat,inv,cc,xs) async(1)
    do j=1,nkz
      do i=1,nkx
        kk = kmod(i,j)
        if(ismean(i,j) .or. isnyq(j)) then
          do m=1,36
            minv(m,i,j) = 0._rp
          end do
        else
          do m=1,6
            do r=1,ny
              psi(i,j,r) = (0._rp,0._rp)
              sx(i,j,r)  = (0._rp,0._rp) ! homogeneous problems have ry = 0
            end do
            do q=1,6
              xs(q) = (0._rp,0._rp)
            end do
            xs(m) = (1._rp,0._rp)
            call add_sources(nkx,nkz,ny,i,j,a1,b1,-1._rp,xs,psi)
            rhs1 = psi(i,j,1)
            mu = xs(5); bt = xs(6)
            call pres_two_solves(nkx,nkz,ny,i,j,kk,klq,kuq,a1,b1,mu,bt,psi,phi,uf,apsn)
            call pres_conds(nkx,nkz,ny,i,j,kk,a1,b1,c1,cn,nc1,ncn,sx,psi,phi,rhs1,apsn,xs(1),xs(2),cc)
            do q=1,6
              mat(q,m) = real(cc(q),rp)
            end do
          end do
          !
          ! Gauss-Jordan inversion with partial pivoting
          !
          do q=1,6
            do p=1,6
              inv(q,p) = 0._rp
            end do
            inv(q,q) = 1._rp
          end do
          do q=1,6
            p = q
            do l=q+1,6
              if(abs(mat(l,q)) > abs(mat(p,q))) p = l
            end do
            if(p /= q) then
              do l=1,6
                t = mat(q,l); mat(q,l) = mat(p,l); mat(p,l) = t
                t = inv(q,l); inv(q,l) = inv(p,l); inv(p,l) = t
              end do
            end if
            f = 1._rp/mat(q,q)
            do l=1,6
              mat(q,l) = mat(q,l)*f; inv(q,l) = inv(q,l)*f
            end do
            do p=1,6
              if(p /= q) then
                f = mat(p,q)
                do l=1,6
                  mat(p,l) = mat(p,l) - f*mat(q,l)
                  inv(p,l) = inv(p,l) - f*inv(q,l)
                end do
              end if
            end do
          end do
          do q=1,6
            do p=1,6
              minv(p+6*(q-1),i,j) = inv(p,q)
            end do
          end do
        end if
      end do
    end do
    !$acc wait(1)
  end subroutine init_influence_matrix
  !
  subroutine pressure_step(u,v,w)
    !
    ! on input u,v,w hold the right-hand sides r^q (Eq. (B.1)); on output the
    ! divergence-free velocity at the new RK substep; phi = beta*dt*p
    !
    implicit none
    complex(rp), intent(inout), dimension(nkx,nkz,ny) :: u,v,w
    integer :: i,j,r,q
    real(rp) :: kx,kz,kk
    complex(rp) :: cc(6),xs(6),apsn,rhs1,zero
    zero = (0._rp,0._rp)
    !$acc parallel loop gang vector collapse(2) default(present) private(cc,xs) async(1)
    do j=1,nkz
      do i=1,nkx
        kx = kxv(i); kz = kzv(j); kk = kmod(i,j)
        if(ismean(i,j)) then
          do r=1,ny
            v(i,j,r) = zero
            phi(i,j,r) = zero
          end do
          u(i,j,1) = zero; u(i,j,ny) = zero
          w(i,j,1) = zero; w(i,j,ny) = zero
        else if(isnyq(j)) then
          do r=1,ny
            u(i,j,r) = zero; v(i,j,r) = zero; w(i,j,r) = zero; phi(i,j,r) = zero
          end do
        else
          !
          ! particular solution
          !
          call pres_rhs(nkx,nkz,ny,i,j,kx,kz,kk,a1,b1,u,v,w,psi)
          rhs1 = psi(i,j,1)
          call pres_two_solves(nkx,nkz,ny,i,j,kk,klq,kuq,a1,b1,zero,zero,psi,phi,uf,apsn)
          call pres_conds(nkx,nkz,ny,i,j,kk,a1,b1,c1,cn,nc1,ncn,v,psi,phi,rhs1,apsn,zero,zero,cc)
          !
          ! influence-matrix coefficients
          !
          do q=1,6
            xs(q) = zero
            do r=1,6
              xs(q) = xs(q) - minv(q+6*(r-1),i,j)*cc(r)
            end do
          end do
          !
          ! final solve with the corrected right-hand side and boundary values
          !
          call pres_rhs(nkx,nkz,ny,i,j,kx,kz,kk,a1,b1,u,v,w,psi)
          call add_sources(nkx,nkz,ny,i,j,a1,b1,-1._rp,xs,psi)
          call pres_two_solves(nkx,nkz,ny,i,j,kk,klq,kuq,a1,b1,xs(5),xs(6),psi,phi,uf,apsn)
          do r=2,ny-1
            u(i,j,r) = u(i,j,r) - cmplx(0._rp,kx,rp)*phi(i,j,r)
            w(i,j,r) = w(i,j,r) - cmplx(0._rp,kz,rp)*phi(i,j,r)
            v(i,j,r) = v(i,j,r) - psi(i,j,r) - kk*phi(i,j,r)
          end do
          u(i,j,1) = zero; u(i,j,ny) = zero
          v(i,j,1) = zero; v(i,j,ny) = zero
          w(i,j,1) = zero; w(i,j,ny) = zero
        end if
      end do
    end do
  end subroutine pressure_step
  !
  subroutine cfr_correct(u,ub_target,dpdx_eff,coef)
    !
    ! constant flow rate: uniform correction of the mean streamwise velocity
    ! (exact for explicit viscous terms); dpdx_eff = applied mean pressure gradient
    !
    implicit none
    complex(rp), intent(inout) :: u(nkx,nkz,ny)
    real(rp), intent(in) :: ub_target,coef
    real(rp), intent(out) :: dpdx_eff
    real(rp) :: ub,du
    integer :: r
    dpdx_eff = 0._rp
    if(.not.mean_owner) return
    ub = 0._rp
    !$acc parallel loop default(present) reduction(+:ub) async(1)
    do r=1,ny
      ub = ub + 0.5_rp*wq(r)*real(u(i00,j00,r),rp)
    end do
    !$acc wait(1)
    du = ub_target - ub
    du = du/(1._rp - 0.5_rp*(wq(1)+wq(ny)))
    !$acc parallel loop default(present) async(1)
    do r=2,ny-1
      u(i00,j00,r) = u(i00,j00,r) + du
    end do
    dpdx_eff = du/coef
  end subroutine cfr_correct
  !
  subroutine deriv1(f,df)
    ! df = D1 f = A1^{-1} B1 f
    implicit none
    complex(rp), intent(in)  :: f(nkx,nkz,ny)
    complex(rp), intent(out) :: df(nkx,nkz,ny)
    integer :: i,j,r,o
    complex(rp) :: s
    !$acc parallel loop gang vector collapse(2) default(present) private(s) async(1)
    do j=1,nkz
      do i=1,nkx
        do r=1,ny
          s = (0._rp,0._rp)
          do o=max(-nbw,1-r),min(nbw,ny-r)
            s = s + b1(o,r)*f(i,j,r+o)
          end do
          if(r > 1) s = s - t1w(r)*df(i,j,r-1)
          df(i,j,r) = s
        end do
        df(i,j,ny) = df(i,j,ny)*t1d(ny)
        do r=ny-1,1,-1
          df(i,j,r) = (df(i,j,r) - t1u(r)*df(i,j,r+1))*t1d(r)
        end do
      end do
    end do
  end subroutine deriv1
  !
  subroutine add_lap(f,nf,coef,tmp)
    ! nf <- nf + coef*(D2 - k^2) f
    implicit none
    complex(rp), intent(in) :: f(nkx,nkz,ny)
    complex(rp), intent(inout) :: nf(nkx,nkz,ny)
    real(rp), intent(in) :: coef
    complex(rp) :: tmp(nkx,nkz,ny)
    integer :: i,j,r,o
    real(rp) :: k2
    complex(rp) :: s
    !$acc parallel loop gang vector collapse(2) default(present) private(s,k2) async(1)
    do j=1,nkz
      do i=1,nkx
        k2 = kmod(i,j)**2
        do r=1,ny
          s = (0._rp,0._rp)
          do o=max(-nbw,1-r),min(nbw,ny-r)
            s = s + b2(o,r)*f(i,j,r+o)
          end do
          if(r > 1) s = s - t2w(r)*tmp(i,j,r-1)
          tmp(i,j,r) = s
        end do
        tmp(i,j,ny) = tmp(i,j,ny)*t2d(ny)
        nf(i,j,ny) = nf(i,j,ny) + coef*(tmp(i,j,ny) - k2*f(i,j,ny))
        do r=ny-1,1,-1
          tmp(i,j,r) = (tmp(i,j,r) - t2u(r)*tmp(i,j,r+1))*t2d(r)
          nf(i,j,r) = nf(i,j,r) + coef*(tmp(i,j,r) - k2*f(i,j,r))
        end do
      end do
    end do
  end subroutine add_lap
  !
  subroutine scalar_solve(s,c,bcb,bct,gb,gt)
    !
    ! theta-method for a scalar:  [ (1 + c k^2) I - c D2 ] s = r  at interior points,
    ! with Dirichlet ('D', s = g) or Neumann ('N', D1 s = g) conditions at the walls
    ! (g only for the mean mode). On input s holds r. Premultiplying by A2 gives the
    ! banded matrix H = (1+ck^2) A2 - c B2; the unknown wall values tau_1, tau_N of
    ! L s enter rows 2 and N-1 (cf. the sigma-correction of Sec. 3), and are determined
    ! together with the wall values of s by a small influence-matrix system.
    !
    implicit none
    complex(rp), intent(inout) :: s(nkx,nkz,ny)
    real(rp), intent(in) :: c,gb,gt
    character(len=1), intent(in) :: bcb,bct
    integer :: i,j,r,m,q,p,l,nh
    real(rp) :: k2,ca,cb
    complex(rp) :: e(4,5),zz(4),f,t,ra1,ran,zero
    logical :: isnb,isnt,isd
    zero = (0._rp,0._rp)
    isnb = (bcb == 'N'); isnt = (bct == 'N')
    isd  = .not.(isnb.or.isnt)
    nh = 3
    if(.not.isd) nh = 5
    !$acc parallel loop gang vector collapse(2) default(present) private(e,zz) async(1)
    do j=1,nkz
      do i=1,nkx
        if(isnyq(j)) then
          do r=1,ny
            s(i,j,r) = zero
          end do
        else
          k2 = kmod(i,j)**2
          ca = 1._rp + c*k2
          cb = -c
          ! A2 r~ (r~ = r with zero wall values)
          ra1 = a2(1,1)*s(i,j,2)
          ran = a2(-1,ny)*s(i,j,ny-1)
          !
          ! m = 1: particular; 2: tau_1; 3: tau_N; 4: s(-1) = 1; 5: s(+1) = 1
          !
          do m=1,nh
            if(m == 1) then
              do r=2,ny-1
                sx(i,j,r) = zero
                do q=max(-1,2-r),min(1,ny-1-r)
                  sx(i,j,r) = sx(i,j,r) + a2(q,r)*s(i,j,r+q)
                end do
              end do
              sx(i,j,1) = zero; sx(i,j,ny) = zero
              if(ismean(i,j)) then
                if(.not.isnb) sx(i,j,1)  = gb
                if(.not.isnt) sx(i,j,ny) = gt
              end if
            else
              do r=1,ny
                sx(i,j,r) = zero
              end do
              if(m == 2) sx(i,j,2)    = a2(-1,2)
              if(m == 3) sx(i,j,ny-1) = a2( 1,ny-1)
              if(m == 4) sx(i,j,1)    = 1._rp
              if(m == 5) sx(i,j,ny)   = 1._rp
            end if
            call bsolve(nkx,nkz,ny,i,j,klh,kuh,ca,cb,a2,b2,.true.,.true.,sx,uf)
            ! conditions: rows 1 and N of H s = A2 g, and the boundary conditions
            e(1,m) = bmul(nkx,nkz,ny,i,j,1 ,ca,cb,a2,b2,sx)
            e(2,m) = bmul(nkx,nkz,ny,i,j,ny,ca,cb,a2,b2,sx)
            if(m == 1) then
              e(1,m) = e(1,m) - ra1
              e(2,m) = e(2,m) - ran
            end if
            if(m == 2) e(1,m) = e(1,m) - a2(0,1)
            if(m == 3) e(2,m) = e(2,m) - a2(0,ny)
            if(isnb) then
              f = zero
              do r=1,nc1
                f = f + c1(r)*sx(i,j,r)
              end do
              e(3,m) = f
              if(m == 1 .and. ismean(i,j)) e(3,m) = e(3,m) - gb
            else
              e(3,m) = zero
              if(m == 4) e(3,m) = (1._rp,0._rp)
            end if
            if(isnt) then
              f = zero
              do r=ny-ncn+1,ny
                f = f + cn(r)*sx(i,j,r)
              end do
              e(4,m) = f
              if(m == 1 .and. ismean(i,j)) e(4,m) = e(4,m) - gt
            else
              e(4,m) = zero
              if(m == 5) e(4,m) = (1._rp,0._rp)
            end if
          end do
          !
          ! solve sum_m e(:,m+1) zz(m) = -e(:,1)  (Gaussian elimination, partial pivoting)
          !
          if(isd) then
            ! Dirichlet: only tau_1 and tau_N are unknown (2x2)
            f = e(1,2)*e(2,3) - e(1,3)*e(2,2)
            zz(1) = (-e(1,1)*e(2,3) + e(1,3)*e(2,1))/f
            zz(2) = (-e(2,1)*e(1,2) + e(2,2)*e(1,1))/f
            zz(3) = zero; zz(4) = zero
          else
            do q=1,4
              e(q,1) = -e(q,1)
            end do
            do q=1,4
              p = q
              do l=q+1,4
                if(abs(e(l,q+1)) > abs(e(p,q+1))) p = l
              end do
              if(p /= q) then
                do l=1,5
                  t = e(q,l); e(q,l) = e(p,l); e(p,l) = t
                end do
              end if
              do p=q+1,4
                f = e(p,q+1)/e(q,q+1)
                do l=1,5
                  e(p,l) = e(p,l) - f*e(q,l)
                end do
              end do
            end do
            do q=4,1,-1
              f = e(q,1)
              do l=q+1,4
                f = f - e(q,l+1)*zz(l)
              end do
              zz(q) = f/e(q,q+1)
            end do
          end if
          !
          ! final solve with the combined right-hand side
          !
          do r=2,ny-1
            sx(i,j,r) = zero
            do q=max(-1,2-r),min(1,ny-1-r)
              sx(i,j,r) = sx(i,j,r) + a2(q,r)*s(i,j,r+q)
            end do
          end do
          sx(i,j,2)    = sx(i,j,2)    + zz(1)*a2(-1,2)
          sx(i,j,ny-1) = sx(i,j,ny-1) + zz(2)*a2( 1,ny-1)
          sx(i,j,1)  = zz(3)
          sx(i,j,ny) = zz(4)
          if(ismean(i,j)) then
            if(.not.isnb) sx(i,j,1)  = gb
            if(.not.isnt) sx(i,j,ny) = gt
          end if
          call bsolve(nkx,nkz,ny,i,j,klh,kuh,ca,cb,a2,b2,.true.,.true.,sx,uf)
          do r=1,ny
            s(i,j,r) = sx(i,j,r)
          end do
        end if
      end do
    end do
  end subroutine scalar_solve
  !
  real(rp) function divergence_max(u,v,w,tmp)
    use mpi
    implicit none
    complex(rp), intent(in) :: u(nkx,nkz,ny),v(nkx,nkz,ny),w(nkx,nkz,ny)
    complex(rp) :: tmp(nkx,nkz,ny)
    real(rp) :: dm
    integer :: i,j,r,ierr
    call deriv1(v,tmp)
    dm = 0._rp
    !$acc parallel loop collapse(3) default(present) reduction(max:dm) async(1)
    do r=1,ny
      do j=1,nkz
        do i=1,nkx
          dm = max(dm,abs(cmplx(0._rp,kxv(i),rp)*u(i,j,r)+cmplx(0._rp,kzv(j),rp)*w(i,j,r)+tmp(i,j,r)))
        end do
      end do
    end do
    !$acc wait(1)
    call MPI_ALLREDUCE(MPI_IN_PLACE,dm,1,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
    divergence_max = dm
  end function divergence_max
  !
  subroutine wall_gradient(f,gb,gt)
    ! dF/dy at both walls of the mean mode (valid on the mean owner only)
    implicit none
    complex(rp), intent(in) :: f(nkx,nkz,ny)
    real(rp), intent(out) :: gb,gt
    integer :: r
    real(rp) :: sb,st
    gb = 0._rp; gt = 0._rp
    if(.not.mean_owner) return
    sb = 0._rp; st = 0._rp
    !$acc parallel loop default(present) reduction(+:sb,st) async(1)
    do r=1,ny
      if(r <= nc1) sb = sb + c1(r)*real(f(i00,j00,r),rp)
      if(r >= ny-ncn+1) st = st + cn(r)*real(f(i00,j00,r),rp)
    end do
    !$acc wait(1)
    gb = sb; gt = st
  end subroutine wall_gradient
  !
  ! ===========================================================================
  ! implicit viscous terms (theta-method), Appendix B.2
  !
  ! Discrete problem per wavenumber pair (c = theta*beta_q*dt/Re, L = (1+c k^2) I - c D2):
  !     L u + i kx phi = r_x,  L v + D1 phi = r_y,  L w + i kz phi = r_z   (interior points)
  !     u = v = w = 0 at the walls,   i kx u + i kz w + D1 v = 0         (all points)
  ! Taking the divergence gives the pressure equation (cf. Eq. (B.32))
  !     (D1^2 - k^2) phi = R + C v + delta_b d1(:,1) + delta_t d1(:,N),   C = c (D1 D2 - D2 D1),
  ! with delta_b,t = (L v)(+-1) - (r_y - D1 phi)(+-1) the wall-row (sigma) corrections.
  ! As in Appendix B.2 the commutator term C v is lagged and iterated upon; everything
  ! else (factorised Poisson solve, the A2-premultiplied Helmholtz solves with their
  ! boundary errors, the wall corrections and the wall continuity conditions) is solved
  ! exactly by superposition with six homogeneous solutions whose 6x6 influence matrix
  ! depends on k and c only (cached per RK substep). At convergence the iteration
  ! reproduces the discrete theta-method solution exactly (divergence at round-off).
  ! ===========================================================================
  !
  subroutine init_implicit
    implicit none
    allocate(minv_i(36,nkx,nkz,3),hinv_i(4,nkx,nkz,3),vit(nkx,nkz,ny),cvec(nkx,nkz,ny),tw1(nkx,nkz,ny))
    minv_i = 0._rp; hinv_i = 0._rp; vit = 0._rp; cvec = 0._rp; tw1 = 0._rp
    !$acc enter data copyin(minv_i,hinv_i,vit,cvec,tw1)
    c_cached(:) = -1._rp
  end subroutine init_implicit
  !
  subroutine dcol(n1,n2,n,i,j,b,tw,td,tu,f,df)
    ! df = A^{-1} B f for column (i,j) (A tridiagonal, Thomas factors tw,td,tu); df /= f
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n2,n,i,j
    real(rp), intent(in) :: b(-nbw:nbw,n),tw(n),td(n),tu(n)
    complex(rp), intent(in) :: f(n1,n2,n)
    complex(rp), intent(inout) :: df(n1,n2,n)
    integer :: r,o
    complex(rp) :: s
    do r=1,n
      s = (0._rp,0._rp)
      do o=max(-nbw,1-r),min(nbw,n-r)
        s = s + b(o,r)*f(i,j,r+o)
      end do
      if(r > 1) s = s - tw(r)*df(i,j,r-1)
      df(i,j,r) = s
    end do
    df(i,j,n) = df(i,j,n)*td(n)
    do r=n-1,1,-1
      df(i,j,r) = (df(i,j,r) - tu(r)*df(i,j,r+1))*td(r)
    end do
  end subroutine dcol
  !
  subroutine a2rhs(n1,n2,n,i,j,a2,g,x)
    ! x = A2 g~ at rows 2..N-1 (g~ = g with zero wall values), x = 0 at rows 1 and N
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n2,n,i,j
    real(rp), intent(in) :: a2(-1:1,n)
    complex(rp), intent(in) :: g(n1,n2,n)
    complex(rp), intent(inout) :: x(n1,n2,n)
    integer :: r,q
    complex(rp) :: s
    do r=2,n-1
      s = (0._rp,0._rp)
      do q=max(-1,2-r),min(1,n-1-r)
        s = s + a2(q,r)*g(i,j,r+q)
      end do
      x(i,j,r) = s
    end do
    x(i,j,1) = (0._rp,0._rp); x(i,j,n) = (0._rp,0._rp)
  end subroutine a2rhs
  !
  subroutine hsetup(n1,n2,n,i,j,kl,ku,ca,cb,a2,b2,x,uf,h)
    !
    ! inverse of the 2x2 matrix that determines the unknown wall values tau_1, tau_N of
    ! L s in the A2-premultiplied Helmholtz problem (the boundary error of Sec. 3/B.2)
    !
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n2,n,i,j,kl,ku
    real(rp), intent(in) :: ca,cb,a2(-1:1,n),b2(-nbw:nbw,n)
    complex(rp), intent(inout) :: x(n1,n2,n)
    real(rp), intent(inout) :: uf(n1,n2,n,0:nbw)
    real(rp), intent(out) :: h(4)
    real(rp) :: m11,m12,m21,m22,det
    integer :: r
    do r=1,n
      x(i,j,r) = (0._rp,0._rp)
    end do
    x(i,j,2) = a2(-1,2)
    call bsolve(n1,n2,n,i,j,kl,ku,ca,cb,a2,b2,.true.,.true.,x,uf)
    m11 = real(bmul(n1,n2,n,i,j,1,ca,cb,a2,b2,x),rp) - a2(0,1)
    m21 = real(bmul(n1,n2,n,i,j,n,ca,cb,a2,b2,x),rp)
    do r=1,n
      x(i,j,r) = (0._rp,0._rp)
    end do
    x(i,j,n-1) = a2(1,n-1)
    call bsolve(n1,n2,n,i,j,kl,ku,ca,cb,a2,b2,.true.,.true.,x,uf)
    m12 = real(bmul(n1,n2,n,i,j,1,ca,cb,a2,b2,x),rp)
    m22 = real(bmul(n1,n2,n,i,j,n,ca,cb,a2,b2,x),rp) - a2(0,n)
    det = m11*m22 - m12*m21
    h(1) =  m22/det; h(2) = -m21/det; h(3) = -m12/det; h(4) = m11/det
  end subroutine hsetup
  !
  subroutine hsolve(n1,n2,n,i,j,kl,ku,ca,cb,a2,b2,h,g,x,uf,t1,tn)
    !
    ! exact solution of  L x = g  at interior points, x = 0 at the walls, with
    ! L = A2^{-1} (ca A2 + cb B2); t1, tn are the wall values of L x
    !
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n2,n,i,j,kl,ku
    real(rp), intent(in) :: ca,cb,a2(-1:1,n),b2(-nbw:nbw,n),h(4)
    complex(rp), intent(in) :: g(n1,n2,n)
    complex(rp), intent(inout) :: x(n1,n2,n)
    real(rp), intent(inout) :: uf(n1,n2,n,0:nbw)
    complex(rp), intent(out) :: t1,tn
    complex(rp) :: e1,e2
    call a2rhs(n1,n2,n,i,j,a2,g,x)
    call bsolve(n1,n2,n,i,j,kl,ku,ca,cb,a2,b2,.true.,.true.,x,uf)
    e1 = bmul(n1,n2,n,i,j,1,ca,cb,a2,b2,x) - a2( 1,1)*g(i,j,2)
    e2 = bmul(n1,n2,n,i,j,n,ca,cb,a2,b2,x) - a2(-1,n)*g(i,j,n-1)
    t1 = -(h(1)*e1 + h(3)*e2)
    tn = -(h(2)*e1 + h(4)*e2)
    call a2rhs(n1,n2,n,i,j,a2,g,x)
    x(i,j,2)   = x(i,j,2)   + t1*a2(-1,2)
    x(i,j,n-1) = x(i,j,n-1) + tn*a2( 1,n-1)
    call bsolve(n1,n2,n,i,j,kl,ku,ca,cb,a2,b2,.true.,.true.,x,uf)
  end subroutine hsolve
  !
  subroutine add_a1(n1,n2,n,i,j,a1,f,x)
    ! x <- x + A1 f (all rows)
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n2,n,i,j
    real(rp), intent(in) :: a1(-1:1,n)
    complex(rp), intent(in) :: f(n1,n2,n)
    complex(rp), intent(inout) :: x(n1,n2,n)
    integer :: r,o
    do r=1,n
      do o=max(-1,1-r),min(1,n-r)
        x(i,j,r) = x(i,j,r) + a1(o,r)*f(i,j,r+o)
      end do
    end do
  end subroutine add_a1
  !
  subroutine imp_conds(n1,n2,n,i,j,kk,klh,kuh,ca,cb,a1,b1,a2,b2,c1,cn,nc1,ncn,h,psi,phi,sx,uf, &
                       rhs1,apsn,db,dt_,cc)
    !
    ! on input psi holds g = r_y - D1 phi; computes v (in sx) from L v = g (interior), v = 0
    ! (walls), and conditions 3-6: definitions of delta_b,t and continuity at the walls
    !
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n2,n,i,j,klh,kuh,nc1,ncn
    real(rp), intent(in) :: kk,ca,cb,a1(-1:1,n),b1(-nbw:nbw,n),a2(-1:1,n),b2(-nbw:nbw,n),c1(n),cn(n),h(4)
    complex(rp), intent(inout) :: psi(n1,n2,n),phi(n1,n2,n),sx(n1,n2,n)
    real(rp), intent(inout) :: uf(n1,n2,n,0:nbw)
    complex(rp), intent(in) :: rhs1,apsn,db,dt_
    complex(rp), intent(inout) :: cc(6)
    complex(rp) :: t1,tn,s5,s6
    integer :: r
    call hsolve(n1,n2,n,i,j,klh,kuh,ca,cb,a2,b2,h,psi,sx,uf,t1,tn)
    cc(3) = db  - (t1 - psi(i,j,1))
    cc(4) = dt_ - (tn - psi(i,j,n))
    s5 = (0._rp,0._rp); s6 = (0._rp,0._rp)
    do r=2,min(nc1,n-1)
      s5 = s5 + c1(r)*sx(i,j,r)
    end do
    do r=max(n-ncn+1,2),n-1
      s6 = s6 + cn(r)*sx(i,j,r)
    end do
    cc(5) = s5; cc(6) = s6
  end subroutine imp_conds
  !
  subroutine inv6(mat,inv)
    ! Gauss-Jordan inversion with partial pivoting of a 6x6 matrix (mat is destroyed)
    !$acc routine seq
    implicit none
    real(rp), intent(inout) :: mat(6,6)
    real(rp), intent(out) :: inv(6,6)
    integer :: q,p,l
    real(rp) :: f,t
    do q=1,6
      do p=1,6
        inv(q,p) = 0._rp
      end do
      inv(q,q) = 1._rp
    end do
    do q=1,6
      p = q
      do l=q+1,6
        if(abs(mat(l,q)) > abs(mat(p,q))) p = l
      end do
      if(p /= q) then
        do l=1,6
          t = mat(q,l); mat(q,l) = mat(p,l); mat(p,l) = t
          t = inv(q,l); inv(q,l) = inv(p,l); inv(p,l) = t
        end do
      end if
      f = 1._rp/mat(q,q)
      do l=1,6
        mat(q,l) = mat(q,l)*f; inv(q,l) = inv(q,l)*f
      end do
      do p=1,6
        if(p /= q) then
          f = mat(p,q)
          do l=1,6
            mat(p,l) = mat(p,l) - f*mat(q,l)
            inv(p,l) = inv(p,l) - f*inv(q,l)
          end do
        end if
      end do
    end do
  end subroutine inv6
  !
  subroutine implicit_matrices(q,c)
    !
    ! influence matrices for substep q and c = theta*beta_q*dt/Re
    !
    implicit none
    integer , intent(in) :: q
    real(rp), intent(in) :: c
    integer :: i,j,m,r,l,p
    real(rp) :: kk,ca,cb,h(4),mat(6,6),inv(6,6)
    complex(rp) :: cc(6),xs(6),apsn,rhs1
    !$acc parallel loop gang vector collapse(2) default(present) private(h,mat,inv,cc,xs) async(1)
    do j=1,nkz
      do i=1,nkx
        kk = kmod(i,j)
        ca = 1._rp + c*kk**2
        cb = -c
        if(isnyq(j)) then
          do l=1,4
            hinv_i(l,i,j,q) = 0._rp
          end do
          do l=1,36
            minv_i(l,i,j,q) = 0._rp
          end do
        else
          call hsetup(nkx,nkz,ny,i,j,klh,kuh,ca,cb,a2,b2,sx,uf,h)
          do l=1,4
            hinv_i(l,i,j,q) = h(l)
          end do
          if(ismean(i,j)) then
            do l=1,36
              minv_i(l,i,j,q) = 0._rp
            end do
          else
            do m=1,6
              do r=1,ny
                psi(i,j,r) = (0._rp,0._rp)
              end do
              do l=1,6
                xs(l) = (0._rp,0._rp)
              end do
              xs(m) = (1._rp,0._rp)
              call add_sources(nkx,nkz,ny,i,j,a1,b1,1._rp,xs,psi)
              rhs1 = psi(i,j,1)
              call pres_two_solves(nkx,nkz,ny,i,j,kk,klq,kuq,a1,b1,xs(5),xs(6),psi,phi,uf,apsn)
              cc(1) = bmul(nkx,nkz,ny,i,j,1 , kk,1._rp,a1,b1,psi) - rhs1
              cc(2) = bmul(nkx,nkz,ny,i,j,ny,-kk,1._rp,a1,b1,phi) - apsn
              do r=1,ny
                psi(i,j,r) = -psi(i,j,r) - kk*phi(i,j,r)   ! g = r_y - D1 phi with r_y = 0
              end do
              call imp_conds(nkx,nkz,ny,i,j,kk,klh,kuh,ca,cb,a1,b1,a2,b2,c1,cn,nc1,ncn,h,psi,phi,sx,uf, &
                             rhs1,apsn,xs(1),xs(2),cc)
              do l=1,6
                mat(l,m) = real(cc(l),rp)
              end do
            end do
            call inv6(mat,inv)
            do p=1,6
              do l=1,6
                minv_i(l+6*(p-1),i,j,q) = inv(l,p)
              end do
            end do
          end if
        end if
      end do
    end do
    !$acc wait(1)
  end subroutine implicit_matrices
  !
  subroutine set_implicit_guess(v)
    ! initial guess of the iteration: v at the previous substep (v^q)
    implicit none
    complex(rp), intent(in) :: v(nkx,nkz,ny)
    integer :: i,j,k
    !$acc parallel loop collapse(3) default(present) async(1)
    do k=1,ny
      do j=1,nkz
        do i=1,nkx
          vit(i,j,k) = v(i,j,k)
        end do
      end do
    end do
  end subroutine set_implicit_guess
  !
  subroutine pressure_step_implicit(u,v,w,q,c,nit,dvrel)
    !
    ! on input u,v,w hold r^q (Eq. (13)); the initial guess of v (v^q) must have been set with
    ! set_implicit_guess. On output the velocity at the new substep; phi = beta*dt*p
    !
    use mpi
    use mod_param, only: impl_maxit,impl_tol
    implicit none
    complex(rp), intent(inout), dimension(nkx,nkz,ny) :: u,v,w
    integer , intent(in) :: q
    real(rp), intent(in) :: c
    integer , intent(out) :: nit
    real(rp), intent(out) :: dvrel
    integer :: i,j,r,l,m,ierr
    real(rp) :: kx,kz,kk,ca,cb,h(4),dvm,vm,buf(2)
    complex(rp) :: cc(6),xs(6),apsn,rhs1,zero,t1,tn
    zero = (0._rp,0._rp)
    if(c /= c_cached(q)) then
      call implicit_matrices(q,c)
      c_cached(q) = c
    end if
    nit = 0
    dvrel = 0._rp
    do m=1,impl_maxit
      dvm = 0._rp; vm = 0._rp
      !$acc parallel loop gang vector collapse(2) default(present) private(h,cc,xs) &
      !$acc reduction(max:dvm,vm) async(1)
      do j=1,nkz
        do i=1,nkx
          if(.not.(ismean(i,j) .or. isnyq(j))) then
            kx = kxv(i); kz = kzv(j); kk = kmod(i,j)
            ca = 1._rp + c*kk**2
            cb = -c
            do l=1,4
              h(l) = hinv_i(l,i,j,q)
            end do
            !
            ! lagged commutator term  C v^m = c (D1 D2 - D2 D1) v^m
            !
            call dcol(nkx,nkz,ny,i,j,b2,t2w,t2d,t2u,vit,tw1)
            call dcol(nkx,nkz,ny,i,j,b1,t1w,t1d,t1u,tw1,cvec)
            call dcol(nkx,nkz,ny,i,j,b1,t1w,t1d,t1u,vit,tw1)
            call dcol(nkx,nkz,ny,i,j,b2,t2w,t2d,t2u,tw1,psi)
            do r=1,ny
              cvec(i,j,r) = c*(cvec(i,j,r) - psi(i,j,r))
            end do
            !
            ! particular solution
            !
            call pres_rhs(nkx,nkz,ny,i,j,kx,kz,kk,a1,b1,u,v,w,psi)
            call add_a1(nkx,nkz,ny,i,j,a1,cvec,psi)
            rhs1 = psi(i,j,1)
            call pres_two_solves(nkx,nkz,ny,i,j,kk,klq,kuq,a1,b1,zero,zero,psi,phi,uf,apsn)
            cc(1) = bmul(nkx,nkz,ny,i,j,1 , kk,1._rp,a1,b1,psi) - rhs1
            cc(2) = bmul(nkx,nkz,ny,i,j,ny,-kk,1._rp,a1,b1,phi) - apsn
            do r=1,ny
              psi(i,j,r) = v(i,j,r) - psi(i,j,r) - kk*phi(i,j,r)
            end do
            call imp_conds(nkx,nkz,ny,i,j,kk,klh,kuh,ca,cb,a1,b1,a2,b2,c1,cn,nc1,ncn,h,psi,phi,sx,uf, &
                           rhs1,apsn,zero,zero,cc)
            do l=1,6
              xs(l) = zero
              do r=1,6
                xs(l) = xs(l) - minv_i(l+6*(r-1),i,j,q)*cc(r)
              end do
            end do
            !
            ! corrected solution
            !
            call pres_rhs(nkx,nkz,ny,i,j,kx,kz,kk,a1,b1,u,v,w,psi)
            call add_a1(nkx,nkz,ny,i,j,a1,cvec,psi)
            call add_sources(nkx,nkz,ny,i,j,a1,b1,1._rp,xs,psi)
            call pres_two_solves(nkx,nkz,ny,i,j,kk,klq,kuq,a1,b1,xs(5),xs(6),psi,phi,uf,apsn)
            do r=1,ny
              psi(i,j,r) = v(i,j,r) - psi(i,j,r) - kk*phi(i,j,r)
            end do
            call hsolve(nkx,nkz,ny,i,j,klh,kuh,ca,cb,a2,b2,h,psi,sx,uf,t1,tn)
            do r=1,ny
              dvm = max(dvm,abs(sx(i,j,r)-vit(i,j,r)))
              vm  = max(vm ,abs(sx(i,j,r)))
              vit(i,j,r) = sx(i,j,r)
            end do
          end if
        end do
      end do
      !$acc wait(1)
      buf(:) = [dvm,vm]
      call MPI_ALLREDUCE(MPI_IN_PLACE,buf,2,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
      nit = m
      dvrel = buf(1)/max(buf(2),tiny(1._rp))
      if(dvrel <= impl_tol) exit
    end do
    !
    ! final velocities: L u = r_x - i kx phi, L w = r_z - i kz phi, v from the last iterate
    !
    !$acc parallel loop gang vector collapse(2) default(present) private(h) async(1)
    do j=1,nkz
      do i=1,nkx
        kx = kxv(i); kz = kzv(j); kk = kmod(i,j)
        ca = 1._rp + c*kk**2
        cb = -c
        if(isnyq(j)) then
          do r=1,ny
            u(i,j,r) = zero; v(i,j,r) = zero; w(i,j,r) = zero; phi(i,j,r) = zero
          end do
        else
          do l=1,4
            h(l) = hinv_i(l,i,j,q)
          end do
          if(ismean(i,j)) then
            do r=1,ny
              phi(i,j,r) = zero
              vit(i,j,r) = zero
            end do
          end if
          do r=1,ny
            psi(i,j,r) = u(i,j,r) - cmplx(0._rp,kx,rp)*phi(i,j,r)
          end do
          call hsolve(nkx,nkz,ny,i,j,klh,kuh,ca,cb,a2,b2,h,psi,sx,uf,t1,tn)
          do r=1,ny
            u(i,j,r) = sx(i,j,r)
            psi(i,j,r) = w(i,j,r) - cmplx(0._rp,kz,rp)*phi(i,j,r)
          end do
          call hsolve(nkx,nkz,ny,i,j,klh,kuh,ca,cb,a2,b2,h,psi,sx,uf,t1,tn)
          do r=1,ny
            w(i,j,r) = sx(i,j,r)
            v(i,j,r) = vit(i,j,r)
          end do
        end if
      end do
    end do
  end subroutine pressure_step_implicit
  !
  subroutine cfr_correct_implicit(u,ub_target,dpdx_eff,coef,c,q)
    !
    ! constant flow rate with implicit viscous terms: the response of the mean flow to a
    ! uniform pressure gradient f over the substep is coef*f*h, with L h = 1, h = 0 at the walls
    !
    implicit none
    complex(rp), intent(inout) :: u(nkx,nkz,ny)
    real(rp), intent(in) :: ub_target,coef,c
    integer , intent(in) :: q
    real(rp), intent(out) :: dpdx_eff
    real(rp) :: ub,hb,alpha,h(4)
    complex(rp) :: t1,tn
    integer :: r,l
    dpdx_eff = 0._rp
    if(.not.mean_owner) return
    !$acc serial default(present) private(h)
    do l=1,4
      h(l) = hinv_i(l,i00,j00,q)
    end do
    do r=1,ny
      psi(i00,j00,r) = (1._rp,0._rp)
    end do
    call hsolve(nkx,nkz,ny,i00,j00,klh,kuh,1._rp,-c,a2,b2,h,psi,sx,uf,t1,tn)
    !$acc end serial
    ub = 0._rp; hb = 0._rp
    !$acc parallel loop default(present) reduction(+:ub,hb) async(1)
    do r=1,ny
      ub = ub + 0.5_rp*wq(r)*real(u(i00,j00,r),rp)
      hb = hb + 0.5_rp*wq(r)*real(sx(i00,j00,r),rp)
    end do
    !$acc wait(1)
    alpha = (ub_target - ub)/hb
    !$acc parallel loop default(present) async(1)
    do r=1,ny
      u(i00,j00,r) = u(i00,j00,r) + alpha*sx(i00,j00,r)
    end do
    dpdx_eff = alpha/coef
  end subroutine cfr_correct_implicit
  !
  real(rp) function mean_gradient_max(f)
    !
    ! max_y |dF/dy| of the mean (kx = kz = 0) profile, on all ranks (for the buoyancy time-step limit)
    !
    use mpi
    implicit none
    complex(rp), intent(in) :: f(nkx,nkz,ny)
    real(rp) :: gm
    integer :: r,ierr
    gm = 0._rp
    if(mean_owner) then
      !$acc serial default(present) async(1)
      call dcol(nkx,nkz,ny,i00,j00,b1,t1w,t1d,t1u,f,psi)
      !$acc end serial
      !$acc parallel loop default(present) reduction(max:gm) async(1)
      do r=1,ny
        gm = max(gm,abs(real(psi(i00,j00,r),rp)))
      end do
      !$acc wait(1)
    end if
    call MPI_ALLREDUCE(MPI_IN_PLACE,gm,1,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
    mean_gradient_max = gm
  end function mean_gradient_max
  !
  ! ===========================================================================
  ! velocity-vorticity method ('vv'): F. Lluesma-Rodriguez, F. Alcantara-Avila, M.J. Perez-Quiles,
  ! S. Hoyas, "A code for simulating heat transfer in turbulent channel flow", Mathematics 9 (2021) 756
  ! (Kim-Moin-Moser formulation, compact finite differences in y, SMR RK3 with implicit viscous terms)
  !
  !   d/dt phi     = h_v + nu (D2 - k^2) phi ,  (D2 - k^2) v = phi ,  v = D1 v = 0 at the walls
  !   d/dt omega_y = h_g + nu (D2 - k^2) omega_y ,                    omega_y = 0 at the walls
  !   h_v = -k^2 N_y - D1 (i kx N_x + i kz N_z) ,   h_g = i kz N_x - i kx N_z
  !   u = (i kx D1 v - i kz omega_y)/k^2 ,   w = (i kz D1 v + i kx omega_y)/k^2   (Eqs. (12)-(13))
  ! N = nonlinear term + body forces (buoyancy, mean pressure gradient). The (0,0) modes of u and w
  ! obey d/dt U00 = N_x00 + nu D2 U00 (and W00), stored in the mean columns of om and phv.
  ! The Neumann condition D1 v = 0 is imposed with two homogeneous solutions (phi = 1 at one wall)
  ! whose 2x2 matrix depends on k and dt only (cached per RK substep). The discrete divergence
  ! i kx u + i kz w + D1 v vanishes identically.
  ! ===========================================================================
  !
  subroutine init_vv
    implicit none
    integer :: i,j,l
    real(rp) :: h(4)
    allocate(phv(nkx,nkz,ny),om(nkx,nkz,ny),wa(nkx,nkz,ny),wb(nkx,nkz,ny))
    allocate(hinv_c(4,nkx,nkz,3),ginv_v(4,nkx,nkz,3),hinv_v(4,nkx,nkz))
    phv = 0; om = 0; wa = 0; wb = 0; hinv_c = 0; ginv_v = 0; hinv_v = 0
    !$acc enter data copyin(phv,om,wa,wb,hinv_c,ginv_v,hinv_v)
    c_cached_vv(:) = -1._rp
    allocate(dw1(ny),dwn(ny))
    dw1(:) = 0._rp; dwn(:) = 0._rp
    if(vv_neumann == 'explicit') then
      dw1(1:6) = cew1(:); dwn(ny-5:ny) = cewn(:)
      ndw1 = 6; ndwn = 6
    else
      dw1(:) = c1(:); dwn(:) = cn(:)
      ndw1 = nc1; ndwn = ncn
    end if
    !$acc enter data copyin(dw1,dwn)
    !
    ! wall corrections of the (k-only) operator  k^2 - D2  used for (D2 - k^2) v = phi
    !
    !$acc parallel loop gang vector collapse(2) default(present) private(h) async(1)
    do j=1,nkz
      do i=1,nkx
        if(.not.(ismean(i,j) .or. isnyq(j))) then
          call hsetup(nkx,nkz,ny,i,j,klh,kuh,kmod(i,j)**2,-1._rp,a2,b2,sx,uf,h)
          do l=1,4
            hinv_v(l,i,j) = h(l)
          end do
        end if
      end do
    end do
    !$acc wait(1)
    !
    ! no pressure in this formulation (statistics of p are zero)
    !
    !$acc kernels default(present) async(1)
    phi(:,:,:) = (0._rp,0._rp)
    !$acc end kernels
    !$acc wait(1)
  end subroutine init_vv
  !
  subroutine hsolve_ab(n1,n2,n,i,j,kl,ku,ca,cb,a2,b2,h,g,x,uf,wa_,wb_)
    !
    ! as hsolve, but with wall values x(1) = wa_, x(n) = wb_
    !
    !$acc routine seq
    implicit none
    integer , intent(in) :: n1,n2,n,i,j,kl,ku
    real(rp), intent(in) :: ca,cb,a2(-1:1,n),b2(-nbw:nbw,n),h(4)
    complex(rp), intent(in) :: g(n1,n2,n),wa_,wb_
    complex(rp), intent(inout) :: x(n1,n2,n)
    real(rp), intent(inout) :: uf(n1,n2,n,0:nbw)
    complex(rp) :: e1,e2,t1,tn
    call a2rhs(n1,n2,n,i,j,a2,g,x)
    x(i,j,1) = wa_; x(i,j,n) = wb_
    call bsolve(n1,n2,n,i,j,kl,ku,ca,cb,a2,b2,.true.,.true.,x,uf)
    e1 = bmul(n1,n2,n,i,j,1,ca,cb,a2,b2,x) - a2( 1,1)*g(i,j,2)
    e2 = bmul(n1,n2,n,i,j,n,ca,cb,a2,b2,x) - a2(-1,n)*g(i,j,n-1)
    t1 = -(h(1)*e1 + h(3)*e2)
    tn = -(h(2)*e1 + h(4)*e2)
    call a2rhs(n1,n2,n,i,j,a2,g,x)
    x(i,j,2)   = x(i,j,2)   + t1*a2(-1,2)
    x(i,j,n-1) = x(i,j,n-1) + tn*a2( 1,n-1)
    x(i,j,1) = wa_; x(i,j,n) = wb_
    call bsolve(n1,n2,n,i,j,kl,ku,ca,cb,a2,b2,.true.,.true.,x,uf)
  end subroutine hsolve_ab
  !
  subroutine vv_state_from_velocity(u,v,w)
    !
    ! phi = (D2 - k^2) v and omega_y = i kz u - i kx w (mean columns: U00 in om, W00 in phv)
    !
    implicit none
    complex(rp), intent(in) :: u(nkx,nkz,ny),v(nkx,nkz,ny),w(nkx,nkz,ny)
    integer :: i,j,r
    real(rp) :: k2
    !$acc parallel loop gang vector collapse(2) default(present) private(k2) async(1)
    do j=1,nkz
      do i=1,nkx
        if(ismean(i,j)) then
          do r=1,ny
            om(i,j,r) = u(i,j,r); phv(i,j,r) = w(i,j,r)
          end do
        else
          k2 = kmod(i,j)**2
          call dcol(nkx,nkz,ny,i,j,b2,t2w,t2d,t2u,v,phv)
          do r=1,ny
            phv(i,j,r) = phv(i,j,r) - k2*v(i,j,r)
            om(i,j,r)  = cmplx(0._rp,kzv(j),rp)*u(i,j,r) - cmplx(0._rp,kxv(i),rp)*w(i,j,r)
          end do
        end if
      end do
    end do
    !$acc wait(1)
  end subroutine vv_state_from_velocity
  !
  subroutine vv_rhs_terms(nx_,ny_,nz_)
    !
    ! (N_x,N_y,N_z) -> h_g (in nx_), h_v (in ny_); mean columns: N_x00 stays in nx_, N_z00 -> ny_
    !
    implicit none
    complex(rp), intent(inout) :: nx_(nkx,nkz,ny),ny_(nkx,nkz,ny),nz_(nkx,nkz,ny)
    integer :: i,j,r
    real(rp) :: kx,kz,k2
    complex(rp) :: zero
    zero = (0._rp,0._rp)
    !$acc parallel loop gang vector collapse(2) default(present) private(kx,kz,k2) async(1)
    do j=1,nkz
      do i=1,nkx
        if(isnyq(j)) then
          do r=1,ny
            nx_(i,j,r) = zero; ny_(i,j,r) = zero
          end do
        else if(ismean(i,j)) then
          do r=1,ny
            ny_(i,j,r) = nz_(i,j,r)
          end do
        else
          kx = kxv(i); kz = kzv(j); k2 = kmod(i,j)**2
          do r=1,ny
            wb(i,j,r) = cmplx(0._rp,kx,rp)*nx_(i,j,r) + cmplx(0._rp,kz,rp)*nz_(i,j,r)
          end do
          call dcol(nkx,nkz,ny,i,j,b1,t1w,t1d,t1u,wb,wa)
          do r=1,ny
            ny_(i,j,r) = -k2*ny_(i,j,r) - wa(i,j,r)
            nx_(i,j,r) = cmplx(0._rp,kz,rp)*nx_(i,j,r) - cmplx(0._rp,kx,rp)*nz_(i,j,r)
          end do
        end if
      end do
    end do
  end subroutine vv_rhs_terms
  !
  subroutine vv_matrices(q,c)
    !
    ! for substep q, c = b_q dt nu: wall corrections of (1 + c k^2) - c D2 and the inverse 2x2 matrix
    ! [D1 v_a(-1) D1 v_b(-1); D1 v_a(1) D1 v_b(1)] of the homogeneous solutions phi_a (1 at y = -1),
    ! phi_b (1 at y = +1), v_a,b from (D2 - k^2) v = phi, v = 0 at the walls
    !
    implicit none
    integer , intent(in) :: q
    real(rp), intent(in) :: c
    integer :: i,j,r,l,m
    real(rp) :: kk,ca,cb,h(4),hv(4),mm(2,2),det,s1,s2
    complex(rp) :: one,zero,wa_,wb_
    one = (1._rp,0._rp); zero = (0._rp,0._rp)
    !$acc parallel loop gang vector collapse(2) default(present) private(h,hv,mm) async(1)
    do j=1,nkz
      do i=1,nkx
        if(.not.isnyq(j)) then
          kk = kmod(i,j)
          ca = 1._rp + c*kk**2
          cb = -c
          call hsetup(nkx,nkz,ny,i,j,klh,kuh,ca,cb,a2,b2,sx,uf,h)
          do l=1,4
            hinv_c(l,i,j,q) = h(l)
          end do
          if(.not.ismean(i,j)) then
            do l=1,4
              hv(l) = hinv_v(l,i,j)
            end do
            do m=1,2
              do r=1,ny
                wb(i,j,r) = zero
              end do
              wa_ = zero; wb_ = zero
              if(m == 1) wa_ = one
              if(m == 2) wb_ = one
              call hsolve_ab(nkx,nkz,ny,i,j,klh,kuh,ca,cb,a2,b2,h,wb,sx,uf,wa_,wb_)
              do r=1,ny
                psi(i,j,r) = -sx(i,j,r)
              end do
              call hsolve_ab(nkx,nkz,ny,i,j,klh,kuh,kk**2,-1._rp,a2,b2,hv,psi,wa,uf,zero,zero)
              s1 = 0._rp; s2 = 0._rp
              do r=1,ndw1
                s1 = s1 + dw1(r)*real(wa(i,j,r),rp)
              end do
              do r=ny-ndwn+1,ny
                s2 = s2 + dwn(r)*real(wa(i,j,r),rp)
              end do
              mm(1,m) = s1; mm(2,m) = s2
            end do
            det = mm(1,1)*mm(2,2) - mm(1,2)*mm(2,1)
            ginv_v(1,i,j,q) =  mm(2,2)/det; ginv_v(2,i,j,q) = -mm(2,1)/det
            ginv_v(3,i,j,q) = -mm(1,2)/det; ginv_v(4,i,j,q) =  mm(1,1)/det
          end if
        end if
      end do
    end do
    !$acc wait(1)
  end subroutine vv_matrices
  !
  subroutine vv_step(u,v,w,q,c)
    !
    ! on input om and phv hold the right-hand sides of Eqs. (30)-(32); on output om, phv, and the
    ! velocity u, v, w at the new substep
    !
    implicit none
    complex(rp), intent(inout), dimension(nkx,nkz,ny) :: u,v,w
    integer , intent(in) :: q
    real(rp), intent(in) :: c
    integer :: i,j,r,l
    real(rp) :: kx,kz,kk,k2,ca,cb,h(4),hv(4)
    complex(rp) :: zero,d1,dn,cp,cm
    zero = (0._rp,0._rp)
    if(c /= c_cached_vv(q)) then
      call vv_matrices(q,c)
      c_cached_vv(q) = c
    end if
    !$acc parallel loop gang vector collapse(2) default(present) private(h,hv) async(1)
    do j=1,nkz
      do i=1,nkx
        if(isnyq(j)) then
          do r=1,ny
            u(i,j,r) = zero; v(i,j,r) = zero; w(i,j,r) = zero; om(i,j,r) = zero; phv(i,j,r) = zero
          end do
        else
          kx = kxv(i); kz = kzv(j); kk = kmod(i,j); k2 = kk**2
          ca = 1._rp + c*k2
          cb = -c
          do l=1,4
            h(l) = hinv_c(l,i,j,q)
          end do
          !
          ! omega_y (U00 in the mean column)
          !
          call hsolve_ab(nkx,nkz,ny,i,j,klh,kuh,ca,cb,a2,b2,h,om,sx,uf,zero,zero)
          do r=1,ny
            om(i,j,r) = sx(i,j,r)
          end do
          if(ismean(i,j)) then
            ! W00 in the mean column of phv
            call hsolve_ab(nkx,nkz,ny,i,j,klh,kuh,ca,cb,a2,b2,h,phv,sx,uf,zero,zero)
            do r=1,ny
              phv(i,j,r) = sx(i,j,r)
              u(i,j,r) = om(i,j,r); w(i,j,r) = sx(i,j,r); v(i,j,r) = zero
            end do
          else
            do l=1,4
              hv(l) = hinv_v(l,i,j)
            end do
            !
            ! particular solution: phi_p = 0 at the walls, v_p = 0 at the walls
            !
            call hsolve_ab(nkx,nkz,ny,i,j,klh,kuh,ca,cb,a2,b2,h,phv,sx,uf,zero,zero)
            do r=1,ny
              psi(i,j,r) = -sx(i,j,r)
            end do
            call hsolve_ab(nkx,nkz,ny,i,j,klh,kuh,k2,-1._rp,a2,b2,hv,psi,wa,uf,zero,zero)
            d1 = zero; dn = zero
            do r=1,ndw1
              d1 = d1 + dw1(r)*wa(i,j,r)
            end do
            do r=ny-ndwn+1,ny
              dn = dn + dwn(r)*wa(i,j,r)
            end do
            ! wall values of phi such that D1 v = 0 at both walls
            cp = -(ginv_v(1,i,j,q)*d1 + ginv_v(3,i,j,q)*dn)
            cm = -(ginv_v(2,i,j,q)*d1 + ginv_v(4,i,j,q)*dn)
            call hsolve_ab(nkx,nkz,ny,i,j,klh,kuh,ca,cb,a2,b2,h,phv,sx,uf,cp,cm)
            do r=1,ny
              psi(i,j,r) = -sx(i,j,r)
              phv(i,j,r) = sx(i,j,r)
            end do
            call hsolve_ab(nkx,nkz,ny,i,j,klh,kuh,k2,-1._rp,a2,b2,hv,psi,v,uf,zero,zero)
            !
            ! u and w from continuity and omega_y (Eqs. (12)-(13))
            !
            call dcol(nkx,nkz,ny,i,j,b1,t1w,t1d,t1u,v,wa)
            do r=1,ny
              u(i,j,r) = (cmplx(0._rp,kx,rp)*wa(i,j,r) - cmplx(0._rp,kz,rp)*om(i,j,r))/k2
              w(i,j,r) = (cmplx(0._rp,kz,rp)*wa(i,j,r) + cmplx(0._rp,kx,rp)*om(i,j,r))/k2
            end do
            u(i,j,1) = zero; u(i,j,ny) = zero; w(i,j,1) = zero; w(i,j,ny) = zero
          end if
        end if
      end do
    end do
  end subroutine vv_step
  !
  subroutine vv_cfr(u,ub_target,dpdx_eff,coef,c,q)
    !
    ! constant flow rate: U00 <- U00 + alpha h with (1 - c D2) h = 1, h = 0 at the walls
    !
    implicit none
    complex(rp), intent(inout) :: u(nkx,nkz,ny)
    real(rp), intent(in) :: ub_target,coef,c
    integer , intent(in) :: q
    real(rp), intent(out) :: dpdx_eff
    real(rp) :: ub,hb,alpha,h(4)
    complex(rp) :: zero
    integer :: r,l
    dpdx_eff = 0._rp
    if(.not.mean_owner) return
    zero = (0._rp,0._rp)
    !$acc serial default(present) private(h)
    do l=1,4
      h(l) = hinv_c(l,i00,j00,q)
    end do
    do r=1,ny
      wb(i00,j00,r) = (1._rp,0._rp)
    end do
    call hsolve_ab(nkx,nkz,ny,i00,j00,klh,kuh,1._rp,-c,a2,b2,h,wb,sx,uf,zero,zero)
    !$acc end serial
    ub = 0._rp; hb = 0._rp
    !$acc parallel loop default(present) reduction(+:ub,hb) async(1)
    do r=1,ny
      ub = ub + 0.5_rp*wq(r)*real(u(i00,j00,r),rp)
      hb = hb + 0.5_rp*wq(r)*real(sx(i00,j00,r),rp)
    end do
    !$acc wait(1)
    alpha = (ub_target - ub)/hb
    !$acc parallel loop default(present) async(1)
    do r=1,ny
      u(i00,j00,r)  = u(i00,j00,r)  + alpha*sx(i00,j00,r)
      om(i00,j00,r) = om(i00,j00,r) + alpha*sx(i00,j00,r)
    end do
    dpdx_eff = alpha/coef
  end subroutine vv_cfr
end module mod_ysolve
