!
! initial conditions (set on the host, then copied to the device)
!
module mod_initflow
  use mod_types
  use mod_param
  use mod_decomp , only: myid,nkx,nkz,kx_lo,kz_lo
  use mod_compact, only: yg
  implicit none
  private
  public :: init_fields
contains
  pure real(rp) function hash01(a,b,c,d)
    ! deterministic pseudo-random number in [0,1) from four integers (decomposition independent)
    implicit none
    integer, intent(in) :: a,b,c,d
    integer(i8) :: h
    h = 1469598103_i8
    h = ieor(h*1099511_i8, int(a,i8)+7919_i8)
    h = ieor(h*1099511_i8, int(b,i8)+104729_i8)
    h = ieor(h*1099511_i8, int(c,i8)+1299709_i8)
    h = ieor(h*1099511_i8, int(d,i8)+15485863_i8)
    h = ieor(h,ishft(h,-29)); h = h*2654435761_i8; h = ieor(h,ishft(h,-32))
    hash01 = real(modulo(h,2147483647_i8),rp)/2147483647._rp
  end function hash01
  !
  complex(rp) function rnd_profile(mx_,mz_,k,comp,seed)
    ! smooth random wall-normal profile of mode (mx_,mz_), Hermitian for mx_ = 0
    implicit none
    integer, intent(in) :: mx_,mz_,k,comp,seed
    integer :: mzz,q
    real(rp) :: y,ph(3),am(3)
    complex(rp) :: f
    logical :: conj_
    mzz = mz_; conj_ = .false.
    if(mx_ == 0 .and. mz_ < 0) then
      mzz = -mz_; conj_ = .true.
    end if
    do q=1,3
      am(q) = hash01(mx_,mzz,comp,seed+10*q) - 0.5_rp
      ph(q) = 2*pi*hash01(mx_,mzz,comp,seed+10*q+5)
    end do
    y = yg(k)
    f = (1._rp-y*y)*(am(1)*exp(cmplx(0._rp,ph(1),rp)) + am(2)*y*exp(cmplx(0._rp,ph(2),rp)) + &
                     am(3)*(2*y*y-1._rp)*exp(cmplx(0._rp,ph(3),rp)))
    if(conj_) f = conjg(f)
    rnd_profile = f
  end function rnd_profile
  !
  complex(rp) function rnd_profile_hf(mx_,mz_,k,comp,seed,nmodes)
    ! random wall-normal profile with up to nmodes half-waves, zero at the walls (for small-scale noise)
    implicit none
    integer, intent(in) :: mx_,mz_,k,comp,seed,nmodes
    integer :: mzz,q,nq
    real(rp) :: y,a,ph
    complex(rp) :: f
    logical :: conj_
    mzz = mz_; conj_ = .false.
    if(mx_ == 0 .and. mz_ < 0) then
      mzz = -mz_; conj_ = .true.
    end if
    y = yg(k)
    f = (0._rp,0._rp)
    do q=1,4
      nq = 1 + int(hash01(mx_,mzz,comp,seed+100*q)*nmodes)
      a  = hash01(mx_,mzz,comp,seed+100*q+7) - 0.5_rp
      ph = 2*pi*hash01(mx_,mzz,comp,seed+100*q+13)
      f = f + a*exp(cmplx(0._rp,ph,rp))*sin(nq*pi*(y+1._rp)/2._rp)
    end do
    if(conj_) f = conjg(f)
    rnd_profile_hf = f
  end function rnd_profile_hf
  !
  subroutine disturbance(u,v,w,kind)
    !
    ! random velocity disturbance (no mean mode), kind = 'large': modes with ls_nmin <= |m| <= ls_nmax
    ! wavelengths per box length in x and/or z (the other index 0 or also in the band), smooth in y;
    ! kind = 'noise': all modes with |mx| <= nx/3, |mz| <= nz/3, broadband in y
    !
    implicit none
    complex(rp), intent(out) :: u(nkx,nkz,ny),v(nkx,nkz,ny),w(nkx,nkz,ny)
    character(len=*), intent(in) :: kind
    integer :: i,j,k,jg,mxx,mzz,nmx,nmz,sd,nyw
    logical :: inx,inz
    u(:,:,:) = 0; v(:,:,:) = 0; w(:,:,:) = 0
    nmx = min(ls_nmax,nx/2-1); nmz = min(ls_nmax,nz/2-1)
    nyw = max(2,ny/6)
    sd = init_seed + merge(0,1000,kind == 'large')
    do k=2,ny-1
      do j=1,nkz
        jg = kz_lo+j-1
        if(jg == nz/2+1) cycle
        mzz = jg-1; if(jg > nz/2) mzz = jg-1-nz
        do i=1,nkx
          mxx = kx_lo+i-2
          if(mxx == 0 .and. mzz == 0) cycle
          if(kind == 'large') then
            inx = (mxx == 0) .or. (mxx >= ls_nmin .and. mxx <= nmx)
            inz = (mzz == 0) .or. (abs(mzz) >= ls_nmin .and. abs(mzz) <= nmz)
            if(.not.(inx .and. inz)) cycle
            u(i,j,k) = rnd_profile(mxx,mzz,k,1,sd)
            v(i,j,k) = rnd_profile(mxx,mzz,k,2,sd)
            w(i,j,k) = rnd_profile(mxx,mzz,k,3,sd)
          else
            if(mxx > nx/3 .or. abs(mzz) > nz/3) cycle
            u(i,j,k) = rnd_profile_hf(mxx,mzz,k,1,sd,nyw)
            v(i,j,k) = rnd_profile_hf(mxx,mzz,k,2,sd,nyw)
            w(i,j,k) = rnd_profile_hf(mxx,mzz,k,3,sd,nyw)
          end if
        end do
      end do
    end do
  end subroutine disturbance
  !
  subroutine project_scale(u,v,w,rms)
    ! make (u,v,w) divergence-free and zero at the walls, then scale the rms to the given value
    use mod_ysolve, only: pressure_step
    implicit none
    complex(rp), intent(inout) :: u(nkx,nkz,ny),v(nkx,nkz,ny),w(nkx,nkz,ny)
    real(rp), intent(in) :: rms
    real(rp) :: e,fac
    ! u,v,w may be the field arrays (already on the device) or host temporaries: 'create' allocates
    ! device copies only for the latter, 'delete' frees only those
    !$acc enter data create(u,v,w)
    !$acc update device(u,v,w)
    call pressure_step(u,v,w)
    !$acc wait(1)
    !$acc update self(u,v,w)
    !$acc exit data delete(u,v,w)
    call fluct_energy(u,v,w,e)
    fac = 0._rp
    if(e > 0._rp) fac = rms/sqrt(e)
    u(:,:,:) = fac*u(:,:,:); v(:,:,:) = fac*v(:,:,:); w(:,:,:) = fac*w(:,:,:)
  end subroutine project_scale
  !
  subroutine init_fields(u,v,w,s)
    use mod_ysolve, only: pressure_step
    implicit none
    complex(rp), intent(inout) :: u(nkx,nkz,ny),v(nkx,nkz,ny),w(nkx,nkz,ny)
    complex(rp), intent(inout), contiguous :: s(:,:,:,:)
    integer :: i,j,k,n,ig,jg,mxx,mzz,iunit,ios,mmax_x,mmax_z
    real(rp) :: y,amp,ulam,e,fac,uref
    real(rp) :: ur,ui,vr,vi,wr,wi
    complex(rp), allocatable :: mode(:,:),un(:,:,:),vn(:,:,:),wn(:,:,:)
    u(:,:,:) = 0; v(:,:,:) = 0; w(:,:,:) = 0; s(:,:,:,:) = 0
    !
    ! laminar profile in the mean mode
    !
    if(forcing == 'cfr') then
      ulam = 1.5_rp*ubulk
    else
      ulam = 0.5_rp*dpdx*re
    end if
    if(kx_lo == 1 .and. kz_lo == 1) then
      do k=1,ny
        u(1,1,k) = ulam*(1._rp-yg(k)**2)
      end do
    end if
    !
    select case(trim(init_type))
    case('laminar','restart')
    case('noise')
      mmax_x = max(1,nx/6); mmax_z = max(1,nz/6)
      do k=2,ny-1
        do j=1,nkz
          jg = kz_lo+j-1
          mzz = jg-1; if(jg > nz/2) mzz = jg-1-nz
          if(jg == nz/2+1 .or. abs(mzz) > mmax_z) cycle
          do i=1,nkx
            ig = kx_lo+i-1
            mxx = ig-1
            if(mxx > mmax_x .or. (mxx == 0 .and. mzz == 0)) cycle
            amp = 1._rp/(1._rp+0.2_rp*(mxx+abs(mzz)))
            u(i,j,k) = u(i,j,k) + amp*rnd_profile(mxx,mzz,k,1,init_seed)
            v(i,j,k) = v(i,j,k) + amp*rnd_profile(mxx,mzz,k,2,init_seed)
            w(i,j,k) = w(i,j,k) + amp*rnd_profile(mxx,mzz,k,3,init_seed)
          end do
        end do
      end do
      !
      ! rescale: rms of the velocity fluctuations = init_amp*U_b, with U_b the laminar bulk velocity
      ! (2/3 of the centreline value), or the velocity unit if there is no mean flow (e.g. convection)
      !
      call fluct_energy(u,v,w,e)
      uref = (2._rp/3._rp)*abs(ulam)
      if(uref == 0._rp) uref = 1._rp
      fac = init_amp*uref/sqrt(max(e,tiny(1._rp)))
      do k=1,ny
        do j=1,nkz
          do i=1,nkx
            if(kx_lo+i-1 == 1 .and. kz_lo+j-1 == 1) cycle
            u(i,j,k) = fac*u(i,j,k); v(i,j,k) = fac*v(i,j,k); w(i,j,k) = fac*w(i,j,k)
          end do
        end do
      end do
    case('largescale')
      !
      ! large-scale disturbances (wavelengths L/ls_nmax ... L/ls_nmin in x and z, rms init_amp) plus
      ! small-scale noise (rms init_noise), both divergence-free; velocities relative to U_b
      ! (or to the velocity unit without mean flow)
      !
      uref = (2._rp/3._rp)*abs(ulam)
      if(uref == 0._rp) uref = 1._rp
      allocate(un(nkx,nkz,ny),vn(nkx,nkz,ny),wn(nkx,nkz,ny))
      call disturbance(un,vn,wn,'noise')
      call project_scale(un,vn,wn,init_noise*uref)
      call disturbance(u,v,w,'large')
      call project_scale(u,v,w,init_amp*uref)
      u(:,:,:) = u(:,:,:) + un(:,:,:); v(:,:,:) = v(:,:,:) + vn(:,:,:); w(:,:,:) = w(:,:,:) + wn(:,:,:)
      if(kx_lo == 1 .and. kz_lo == 1) then
        do k=1,ny
          u(1,1,k) = ulam*(1._rp-yg(k)**2)
        end do
        v(1,1,:) = 0; w(1,1,:) = 0
      end if
    case('mode')
      !
      ! a single Fourier mode (e.g. an Orr-Sommerfeld eigenfunction) read from mode_file:
      ! ny lines with Re(u) Im(u) Re(v) Im(v) Re(w) Im(w); amplitude init_amp
      !
      allocate(mode(ny,3))
      open(newunit=iunit,file=trim(mode_file),status='old',action='read',iostat=ios)
      if(ios /= 0) call abort_run('cannot open mode file '//trim(mode_file))
      do k=1,ny
        read(iunit,*,iostat=ios) ur,ui,vr,vi,wr,wi
        if(ios /= 0) call abort_run('mode file too short')
        mode(k,:) = [cmplx(ur,ui,rp),cmplx(vr,vi,rp),cmplx(wr,wi,rp)]
      end do
      close(iunit)
      do j=1,nkz
        jg = kz_lo+j-1
        mzz = jg-1; if(jg > nz/2) mzz = jg-1-nz
        do i=1,nkx
          mxx = kx_lo+i-1-1
          if(mxx == mode_kx .and. mzz == mode_kz) then
            u(i,j,:) = u(i,j,:) + init_amp*mode(:,1)
            v(i,j,:) = v(i,j,:) + init_amp*mode(:,2)
            w(i,j,:) = w(i,j,:) + init_amp*mode(:,3)
          end if
          if(mode_kx == 0 .and. mxx == 0 .and. mzz == -mode_kz .and. mode_kz /= 0) then
            u(i,j,:) = u(i,j,:) + init_amp*conjg(mode(:,1))
            v(i,j,:) = v(i,j,:) + init_amp*conjg(mode(:,2))
            w(i,j,:) = w(i,j,:) + init_amp*conjg(mode(:,3))
          end if
        end do
      end do
    case default
      call abort_run('unknown init_type '//trim(init_type))
    end select
    !
    ! scalars
    !
    do n=1,nscal
      if(kx_lo == 1 .and. kz_lo == 1) then
        do k=1,ny
          y = yg(k)
          select case(trim(sinit(n)))
          case('linear','noise')
            if(sbc_bot(n) == 'D' .and. sbc_top(n) == 'D') then
              s(1,1,k,n) = 0.5_rp*(sval_bot(n)*(1._rp-y) + sval_top(n)*(1._rp+y))
            end if
          case('zero')
          case default
            call abort_run('unknown sinit')
          end select
        end do
      end if
      if(trim(sinit(n)) == 'noise') then
        do k=2,ny-1
          do j=1,nkz
            jg = kz_lo+j-1
            mzz = jg-1; if(jg > nz/2) mzz = jg-1-nz
            if(jg == nz/2+1 .or. abs(mzz) > max(1,nz/6)) cycle
            do i=1,nkx
              mxx = kx_lo+i-2
              if(mxx > max(1,nx/6) .or. (mxx == 0 .and. mzz == 0)) cycle
              s(i,j,k,n) = s(i,j,k,n) + init_amp*rnd_profile(mxx,mzz,k,3+n,init_seed)
            end do
          end do
        end do
      end if
    end do
    !$acc update device(u,v,w,s)
    !
    ! make the initial velocity field divergence-free (and zero at the walls)
    !
    call pressure_step(u,v,w)
    !$acc wait(1)
    !$acc update self(u,v,w)
  end subroutine init_fields
  !
  subroutine fluct_energy(u,v,w,e)
    ! e = < u'u' + v'v' + w'w' >/3 (volume average, Parseval)
    use mpi
    use mod_compact, only: wq
    implicit none
    complex(rp), intent(in) :: u(nkx,nkz,ny),v(nkx,nkz,ny),w(nkx,nkz,ny)
    real(rp), intent(out) :: e
    integer :: i,j,k,ierr
    real(rp) :: wt
    e = 0._rp
    do k=1,ny
      do j=1,nkz
        do i=1,nkx
          wt = 2._rp
          if(kx_lo+i-1 == 1) wt = 1._rp
          if(kx_lo+i-1 == 1 .and. kz_lo+j-1 == 1) wt = 0._rp
          e = e + 0.5_rp*wq(k)*wt*(abs(u(i,j,k))**2+abs(v(i,j,k))**2+abs(w(i,j,k))**2)/3._rp
        end do
      end do
    end do
    call MPI_ALLREDUCE(MPI_IN_PLACE,e,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,ierr)
  end subroutine fluct_energy
end module mod_initflow
