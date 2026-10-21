!
! nonlinear terms and the low-storage RK3 time step
!
!   u^{q+1} = u^q + dt [ gam_q N^q + zet_q N^{q-1} ] - beta_q dt grad p^{q+1}
!
! N contains the advection terms (pseudo-spectral, 3/2-rule dealiasing in x and z),
! the explicit viscous terms and the mean pressure gradient (Sec. 2.3, Appendix B.1).
! Scalars use the theta-method for diffusion (Eqs. (9)-(13)); theta = 0 puts the
! diffusion into N (RK3).
!
module mod_rk
  use mpi
  use mod_types
  use mod_param
  use mod_decomp , only: nkx,nkz,nzA,nyA,yA_lo,zA_lo
  use mod_fields
  use mod_fft    , only: fwd_fft,bwd_fft
  use mod_ysolve , only: deriv1,add_lap,pressure_step,scalar_solve,cfr_correct,kxv,kzv,mean_owner,i00,j00, &
                         set_implicit_guess,pressure_step_implicit,cfr_correct_implicit,mean_gradient_max
  use mod_compact, only: kappa1,lam2max
  implicit none
  private
  public :: init_rk,rk_step,dpdx_applied,dyc_d,impl_nit,impl_res
  real(rp) :: dpdx_applied = 0._rp
  integer  :: impl_nit = 0          ! iterations of the implicit coupling in the last substep
  real(rp) :: impl_res = 0._rp      ! relative change of v in the last iteration
  real(rp), allocatable :: dyc_d(:)
contains
  subroutine init_rk
    use mod_compact, only: dyc
    implicit none
    allocate(dyc_d,source=dyc)
    !$acc enter data copyin(dyc_d)
  end subroutine init_rk
  !
  subroutine spec_mul(f,g,ax,az)
    ! g = i (ax*kx + az*kz) f
    implicit none
    complex(rp), intent(in)  :: f(nkx,nkz,ny)
    complex(rp), intent(out) :: g(nkx,nkz,ny)
    real(rp), intent(in) :: ax,az
    integer :: i,j,k
    !$acc parallel loop collapse(3) default(present) async(1)
    do k=1,ny
      do j=1,nkz
        do i=1,nkx
          g(i,j,k) = cmplx(0._rp,ax*kxv(i)+az*kzv(j),rp)*f(i,j,k)
        end do
      end do
    end do
  end subroutine spec_mul
  !
  subroutine spec_axpy(f,g,ax,az,c)
    ! g = c*g + i (ax*kx + az*kz) f
    implicit none
    complex(rp), intent(in)    :: f(nkx,nkz,ny)
    complex(rp), intent(inout) :: g(nkx,nkz,ny)
    real(rp), intent(in) :: ax,az,c
    integer :: i,j,k
    !$acc parallel loop collapse(3) default(present) async(1)
    do k=1,ny
      do j=1,nkz
        do i=1,nkx
          g(i,j,k) = c*g(i,j,k) + cmplx(0._rp,ax*kxv(i)+az*kzv(j),rp)*f(i,j,k)
        end do
      end do
    end do
  end subroutine spec_axpy
  !
  subroutine phys_prod(a,b,c,fac,accum)
    ! c = fac*a*b (accum = .false.) or c = c + fac*a*b
    implicit none
    real(rp), intent(in) :: a(mx,nzA,nyA),b(mx,nzA,nyA),fac
    real(rp), intent(inout) :: c(mx,nzA,nyA)
    logical, intent(in) :: accum
    integer :: i,j,k
    if(accum) then
      !$acc parallel loop collapse(3) default(present) async(1)
      do k=1,nyA
        do j=1,nzA
          do i=1,mx
            c(i,j,k) = c(i,j,k) + fac*a(i,j,k)*b(i,j,k)
          end do
        end do
      end do
    else
      !$acc parallel loop collapse(3) default(present) async(1)
      do k=1,nyA
        do j=1,nzA
          do i=1,mx
            c(i,j,k) = fac*a(i,j,k)*b(i,j,k)
          end do
        end do
      end do
    end if
  end subroutine phys_prod
  !
  subroutine advect(f)
    ! ap = -(u df/dx + v df/dy + w df/dz) in physical space (advective form)
    implicit none
    complex(rp), intent(in)  :: f(nkx,nkz,ny)
    call spec_mul(f,tmp1,1._rp,0._rp)
    call bwd_fft(tmp1,gp(:,:,:,1))
    call phys_prod(up,gp(:,:,:,1),ap,-1._rp,.false.)
    call deriv1(f,tmp1)
    call bwd_fft(tmp1,gp(:,:,:,1))
    call phys_prod(vp,gp(:,:,:,1),ap,-1._rp,.true.)
    call spec_mul(f,tmp1,0._rp,1._rp)
    call bwd_fft(tmp1,gp(:,:,:,1))
    call phys_prod(wp,gp(:,:,:,1),ap,-1._rp,.true.)
  end subroutine advect
  !
  subroutine nonlinear(is_first,dtnew)
    implicit none
    logical , intent(in)  :: is_first
    real(rp), intent(out) :: dtnew
    integer :: n,i,j,k
    real(rp) :: qc,qu
    call bwd_fft(u,up)
    call bwd_fft(v,vp)
    call bwd_fft(w,wp)
    dtnew = dt
    if(is_first .and. .not.fixed_dt) dtnew = compute_dt()
    !
    ! scalars first (only up, vp, wp are needed)
    !
    do n=1,nscal
      call advect(s(:,:,:,n))
      qc = ssrc_const(n); qu = ssrc_u(n)/ubulk
      if(qc /= 0._rp .or. qu /= 0._rp) then
        !$acc parallel loop collapse(3) default(present) async(1)
        do k=1,nyA
          do j=1,nzA
            do i=1,mx
              ap(i,j,k) = ap(i,j,k) + qc + qu*up(i,j,k)
            end do
          end do
        end do
      end if
      call fwd_fft(ap,ns(:,:,:,n))
      if(scal_theta(n) == 0._rp) call add_lap(s(:,:,:,n),ns(:,:,:,n),diff(n),tmp1)
    end do
    !
    ! momentum
    !
    if(trim(nl_form) == 'advective') then
      call advect(u)
      call fwd_fft(ap,nu)
      call advect(v)
      call fwd_fft(ap,nv)
      call advect(w)
      call fwd_fft(ap,nw)
    else
      ! omega_x = dw/dy - dv/dz
      call deriv1(w,tmp1)
      call spec_axpy(v,tmp1,0._rp,-1._rp,1._rp)
      call bwd_fft(tmp1,gp(:,:,:,1))
      ! omega_y = du/dz - dw/dx
      call spec_mul(u,tmp1,0._rp,1._rp)
      call spec_axpy(w,tmp1,-1._rp,0._rp,1._rp)
      call bwd_fft(tmp1,gp(:,:,:,2))
      ! omega_z = dv/dx - du/dy
      call deriv1(u,tmp1)
      call spec_axpy(v,tmp1,1._rp,0._rp,-1._rp)
      call bwd_fft(tmp1,gp(:,:,:,3))
      ! N = u x omega
      call phys_prod(vp,gp(:,:,:,3),ap, 1._rp,.false.)
      call phys_prod(wp,gp(:,:,:,2),ap,-1._rp,.true.)
      call fwd_fft(ap,nu)
      call phys_prod(wp,gp(:,:,:,1),ap, 1._rp,.false.)
      call phys_prod(up,gp(:,:,:,3),ap,-1._rp,.true.)
      call fwd_fft(ap,nv)
      call phys_prod(up,gp(:,:,:,2),ap, 1._rp,.false.)
      call phys_prod(vp,gp(:,:,:,1),ap,-1._rp,.true.)
      call fwd_fft(ap,nw)
    end if
    if(visc_theta == 0._rp) then
      call add_lap(u,nu,visc,tmp1)
      call add_lap(v,nv,visc,tmp1)
      call add_lap(w,nw,visc,tmp1)
    end if
    !
    ! buoyancy (Boussinesq): - sum_n ri(n) s_n e_g, explicit (RK3) with the scalars of this stage
    !
    do n=1,nscal
      if(ri(n) /= 0._rp) call add_buoyancy(s(:,:,:,n),-ri(n)*gvec(:))
    end do
    if(forcing == 'cpg' .and. mean_owner) then
      !$acc parallel loop default(present) async(1)
      do k=1,ny
        nu(i00,j00,k) = nu(i00,j00,k) + dpdx
      end do
    end if
  end subroutine nonlinear
  !
  subroutine add_buoyancy(f,b)
    ! (nu,nv,nw) <- (nu,nv,nw) + b(1:3)*f
    implicit none
    complex(rp), intent(in) :: f(nkx,nkz,ny)
    real(rp), intent(in) :: b(3)
    real(rp) :: bx,by,bz
    integer :: i,j,k
    bx = b(1); by = b(2); bz = b(3)
    !$acc parallel loop collapse(3) default(present) async(1)
    do k=1,ny
      do j=1,nkz
        do i=1,nkx
          nu(i,j,k) = nu(i,j,k) + bx*f(i,j,k)
          nv(i,j,k) = nv(i,j,k) + by*f(i,j,k)
          nw(i,j,k) = nw(i,j,k) + bz*f(i,j,k)
        end do
      end do
    end do
  end subroutine add_buoyancy
  !
  real(rp) function compute_dt()
    !
    ! RK3 stability: |lambda dt| < sqrt(3) on the imaginary axis (advection) and
    ! < 2.51 on the real axis (explicit diffusion)
    !
    implicit none
    real(rp) :: cmax,kxm,kzm,dta,dtv,nmax
    integer :: i,j,k,n,ierr
    kxm = pi*nx/lx
    kzm = pi*nz/lz
    cmax = 0._rp
    !$acc parallel loop collapse(3) default(present) reduction(max:cmax) async(1)
    do k=1,nyA
      do j=1,nzA
        do i=1,mx
          cmax = max(cmax,abs(up(i,j,k))*kxm + abs(wp(i,j,k))*kzm + abs(vp(i,j,k))*kappa1/dyc_d(yA_lo+k-1))
        end do
      end do
    end do
    !$acc wait(1)
    call MPI_ALLREDUCE(MPI_IN_PLACE,cmax,1,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,ierr)
    dta = cfl/max(cmax,1.e-12_rp)
    !
    ! the theta-method is unconditionally stable for theta >= 1/2; for theta < 1/2 the
    ! explicit part limits the time step (effective diffusivity (1-2 theta)*nu)
    !
    nmax = visc*max(0._rp,1._rp-2._rp*visc_theta)
    do n=1,nscal
      nmax = max(nmax,diff(n)*max(0._rp,1._rp-2._rp*scal_theta(n)))
    end do
    if(nmax > 0._rp) then
      dtv = cfl_visc/(nmax*(kxm**2+kzm**2+lam2max))
    else
      dtv = huge(1._rp)
    end if
    compute_dt = min(dta,dtv,dt_max)
    !
    ! internal gravity waves: |lambda| <= N, N^2 = |ri| |dS/dy| (mean stratification)
    !
    do n=1,nscal
      if(ri(n) /= 0._rp) then
        nmax = sqrt(abs(ri(n))*mean_gradient_max(s(:,:,:,n)))
        if(nmax > 0._rp) compute_dt = min(compute_dt,cfl/nmax)
      end if
    end do
  end function compute_dt
  !
  subroutine update(f,nf,nfo,dtg,dtz)
    ! f = f + dtg*nf + dtz*nfo ; nfo = nf
    implicit none
    complex(rp), intent(inout) :: f(nkx,nkz,ny),nfo(nkx,nkz,ny)
    complex(rp), intent(in) :: nf(nkx,nkz,ny)
    real(rp), intent(in) :: dtg,dtz
    integer :: i,j,k
    !$acc parallel loop collapse(3) default(present) async(1)
    do k=1,ny
      do j=1,nkz
        do i=1,nkx
          f(i,j,k)   = f(i,j,k) + dtg*nf(i,j,k) + dtz*nfo(i,j,k)
          nfo(i,j,k) = nf(i,j,k)
        end do
      end do
    end do
  end subroutine update
  !
  subroutine rhs_theta(f,nf,nfo,dtg,dtz,cexp)
    !
    ! right-hand side of the theta-method, Eq. (13):
    !   f <- f + dtg*nf + dtz*nfo + cexp*(D2 - k^2) f ,   nfo <- nf
    !
    implicit none
    complex(rp), intent(inout) :: f(nkx,nkz,ny),nfo(nkx,nkz,ny)
    complex(rp), intent(in) :: nf(nkx,nkz,ny)
    real(rp), intent(in) :: dtg,dtz,cexp
    integer :: i,j,k
    !$acc parallel loop collapse(3) default(present) async(1)
    do k=1,ny
      do j=1,nkz
        do i=1,nkx
          tmp2(i,j,k) = f(i,j,k) + dtg*nf(i,j,k) + dtz*nfo(i,j,k)
          nfo(i,j,k)  = nf(i,j,k)
        end do
      end do
    end do
    if(cexp /= 0._rp) call add_lap(f,tmp2,cexp,tmp1)
    !$acc parallel loop collapse(3) default(present) async(1)
    do k=1,ny
      do j=1,nkz
        do i=1,nkx
          f(i,j,k) = tmp2(i,j,k)
        end do
      end do
    end do
  end subroutine rhs_theta
  !
  subroutine rk_step(time,dtcur)
    implicit none
    real(rp), intent(inout) :: time,dtcur
    integer :: q,n
    real(rp) :: dtn,dtg,dtz,dtb,c,dpe,th
    do q=1,3
      call nonlinear(q == 1,dtn)
      if(q == 1) then
        if(visc_theta > 0._rp .and. .not.fixed_dt) then
          !
          ! keep dt piecewise constant: the influence matrices of the implicit scheme
          ! depend on dt and are recomputed only when it changes
          !
          if(dtcur > dtn .or. dtcur < 0.8_rp*dtn) dtcur = 0.95_rp*dtn
        else
          dtcur = dtn
        end if
      end if
      dtg = rk_gam(q)*dtcur
      dtz = rk_zet(q)*dtcur
      dtb = rk_bet(q)*dtcur
      !
      ! velocity: r^q, then pressure / influence-matrix step
      !
      if(visc_theta == 0._rp) then
        call update(u,nu,nuo,dtg,dtz)
        call update(v,nv,nvo,dtg,dtz)
        call update(w,nw,nwo,dtg,dtz)
        call pressure_step(u,v,w)
        if(forcing == 'cfr') call cfr_correct(u,ubulk,dpe,dtb)
      else
        !
        ! implicit viscous terms (Appendix B.2)
        !
        call set_implicit_guess(v)
        call rhs_theta(u,nu,nuo,dtg,dtz,(1._rp-visc_theta)*dtb*visc)
        call rhs_theta(v,nv,nvo,dtg,dtz,(1._rp-visc_theta)*dtb*visc)
        call rhs_theta(w,nw,nwo,dtg,dtz,(1._rp-visc_theta)*dtb*visc)
        c = visc_theta*dtb*visc
        call pressure_step_implicit(u,v,w,q,c,impl_nit,impl_res)
        if(forcing == 'cfr') call cfr_correct_implicit(u,ubulk,dpe,dtb,c,q)
      end if
      if(forcing == 'cfr') then
        if(q == 3) dpdx_applied = dpe
      else
        dpdx_applied = dpdx
      end if
      !
      ! scalars
      !
      do n=1,nscal
        th = scal_theta(n)
        if(th > 0._rp) then
          ! theta-method (Eqs. (9)-(13), B.2): r = s + dt(gam N + zet N_old) + (1-theta) beta dt kappa Lap(s)
          call rhs_theta(s(:,:,:,n),ns(:,:,:,n),nso(:,:,:,n),dtg,dtz,(1._rp-th)*dtb*diff(n))
          c = th*dtb*diff(n)
        else
          call update(s(:,:,:,n),ns(:,:,:,n),nso(:,:,:,n),dtg,dtz)
          c = 0._rp
        end if
        call scalar_solve(s(:,:,:,n),c,sbc_bot(n),sbc_top(n),sval_bot(n),sval_top(n))
      end do
    end do
    !$acc wait(1)
    time = time + dtcur
  end subroutine rk_step
end module mod_rk
