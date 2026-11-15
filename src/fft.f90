!
! transforms between the physical (dealiased) x-pencil and the spectral z-pencil
!
!   physical : real    f(mx , nzA, nyA)     x in [0,lx), z in [0,lz) on the (3/2-)padded grid
!   spectral : complex fh(nkx, nkz, ny)     kx = 0..nx/2-1, kz = 0..nz/2-1, (Nyquist = 0), -nz/2+1..-1
!
!   f(x,z,y) = sum_{kx,kz} fh(kx,kz,y) exp(i(kx x + kz z)) , kx >= 0 with Hermitian symmetry
!
! GPU: cuFFT batched plans on the OpenACC stream (as in CaNS); CPU: FFTW3.
!
module mod_fft
  use, intrinsic :: iso_c_binding
  use mod_types
  use mod_param , only: nxh,nz,ny,mx,mz,abort_run,fft_2d
  use mod_decomp, only: nzA,nyA,nkx,nkz,tr_xa2ya,tr_ya2xa,tr_yb2zb,tr_zb2yb,p1,myid
  use mod_timer , only: tic,toc,tm_fft,tm_tr
#if defined(_OPENACC)
  use cufft
  use mod_decomp, only: istream
#endif
  implicit none
#if !defined(_OPENACC)
  include 'fftw3.f03'
#endif
  private
  public :: init_fft,fwd_fft,bwd_fft,end_fft,fft_buffer_bytes
  !
  ! work buffers, shared by stages that are never active at the same time:
  !   bufp: cxw(mx/2+1,nzA,nyA) (x-FFT, spectral side)  and  yaw(mz,nyA,nkx) (z-FFT)
  !   bufa: xaw(nxh,nzA,nyA)    (x-pencil, grid A)       and  ybw(nz,nyA,nkx) (y-pencil, grid B)
  ! On GPUs bufa is also the work area of all cuFFT plans: it is idle during every cuFFT call.
  !
  complex(rp), allocatable :: bufp(:),bufa(:)
#if defined(_OPENACC)
  integer :: plan_xf,plan_xb,plan_z
#else
  type(c_ptr) :: plan_xf,plan_xb,plan_zf,plan_zb
#endif
  real(rp) :: norm
  !
  ! use2d (fft_2d = .true. and p1 = 1, i.e. slabs in y): x and z are both local in the x-pencil, so
  ! the two 1D transforms and the (local) x<->y transpose between them are replaced by one batched
  ! 2D transform (plan_xf, plan_xb) and one copy that truncates/pads in x and z and reorders:
  !   phys(mx,mz,nyA) <-> cxw(mx/2+1,mz,nyA) <-> ybw(nz,nyA,nxh)
  !
  logical :: use2d = .false.
contains
  subroutine init_fft(phys)
    implicit none
    real(rp), intent(inout) :: phys(mx,nzA,nyA) ! only used for planning (not modified)
    integer(i8) :: np_,na_
#if defined(_OPENACC)
    integer :: istat
    integer(i8) :: ws(3)
#endif
    np_ = max(int(mx/2+1,i8)*nzA*nyA,int(mz,i8)*nyA*nkx)
    na_ = max(int(nxh,i8)*nzA*nyA,int(nz,i8)*nyA*nkx)
    norm = 1._rp/(real(mx,rp)*real(mz,rp))
    use2d = fft_2d .and. p1 == 1 .and. nzA == mz .and. nkx == nxh
    if(myid == 0 .and. fft_2d) then
      if(use2d) then
        write(*,'(a)') ' FFTs: batched 2D transforms in x and z (fft_2d, slabs in y)'
      else
        write(*,'(a)') ' FFTs: fft_2d needs p1 = 1 (pdims = 1, P); using the 1D transforms'
      end if
    end if
#if defined(_OPENACC)
    ws(:) = 0
#if defined(_CUFFT_AUTO_WORK)
    if(use2d) then
      istat = cufftPlanMany(plan_xf,2,[mz,mx],[mz,mx],1,mx*mz,[mz,mx/2+1],1,(mx/2+1)*mz,CUFFT_D2Z,nyA)
      if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (2D, forward)')
      istat = cufftPlanMany(plan_xb,2,[mz,mx],[mz,mx/2+1],1,(mx/2+1)*mz,[mz,mx],1,mx*mz,CUFFT_Z2D,nyA)
      if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (2D, backward)')
    else
      istat = cufftPlanMany(plan_xf,1,[mx],[mx],1,mx,[mx/2+1],1,mx/2+1,CUFFT_D2Z,nzA*nyA)
      if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (x, forward)')
      istat = cufftPlanMany(plan_xb,1,[mx],[mx/2+1],1,mx/2+1,[mx],1,mx,CUFFT_Z2D,nzA*nyA)
      if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (x, backward)')
      istat = cufftPlanMany(plan_z ,1,[mz],[mz],1,mz,[mz],1,mz,CUFFT_Z2Z,nyA*nkx)
      if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (z)')
    end if
#else
    ! plans without their own work areas (as in CaNS); one shared area is set below
    istat = cufftCreate(plan_xf); istat = cufftSetAutoAllocation(plan_xf,0)
    istat = cufftCreate(plan_xb); istat = cufftSetAutoAllocation(plan_xb,0)
    if(use2d) then
      istat = cufftMakePlanMany(plan_xf,2,[mz,mx],[mz,mx],1,mx*mz,[mz,mx/2+1],1,(mx/2+1)*mz,CUFFT_D2Z,nyA,ws(1))
      if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (2D, forward)')
      istat = cufftMakePlanMany(plan_xb,2,[mz,mx],[mz,mx/2+1],1,(mx/2+1)*mz,[mz,mx],1,mx*mz,CUFFT_Z2D,nyA,ws(2))
      if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (2D, backward)')
    else
      istat = cufftMakePlanMany(plan_xf,1,[mx],[mx],1,mx,[mx/2+1],1,mx/2+1,CUFFT_D2Z,nzA*nyA,ws(1))
      if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (x, forward)')
      istat = cufftMakePlanMany(plan_xb,1,[mx],[mx/2+1],1,mx/2+1,[mx],1,mx,CUFFT_Z2D,nzA*nyA,ws(2))
      if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (x, backward)')
      istat = cufftCreate(plan_z ); istat = cufftSetAutoAllocation(plan_z ,0)
      istat = cufftMakePlanMany(plan_z ,1,[mz],[mz],1,mz,[mz],1,mz,CUFFT_Z2Z,nyA*nkx,ws(3))
      if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (z)')
    end if
    na_ = max(na_,(maxval(ws)+15)/16)
#endif
#endif
    allocate(bufp(np_),bufa(na_))
    !$acc enter data create(bufp,bufa)
#if defined(_OPENACC)
#if !defined(_CUFFT_AUTO_WORK)
    !$acc host_data use_device(bufa)
    istat = cufftSetWorkArea(plan_xf,bufa)
    istat = istat + cufftSetWorkArea(plan_xb,bufa)
    if(.not.use2d) istat = istat + cufftSetWorkArea(plan_z ,bufa)
    !$acc end host_data
    if(istat /= CUFFT_SUCCESS) call abort_run('cufftSetWorkArea failed')
#endif
    istat = cufftSetStream(plan_xf,istream)
    istat = cufftSetStream(plan_xb,istream)
    if(.not.use2d) istat = cufftSetStream(plan_z ,istream)
#else
    call plan_cpu(phys,bufp,bufp)
#endif
  end subroutine init_fft
  !
#if !defined(_OPENACC)
  subroutine plan_cpu(phys,cxw,yaw)
    implicit none
    real(rp), intent(inout) :: phys(mx,nzA,nyA)
    complex(rp), intent(inout) :: cxw(mx/2+1,nzA,nyA),yaw(mz,nyA,nkx)
    integer(c_int) :: flags
    ! the x-plans are also executed on gp(:,:,:,2), which starts mx*nzA*nyA reals after an aligned
    ! address: not 16-byte aligned if that number is odd
    flags = FFTW_ESTIMATE
    if(mod(mx,2) == 1 .and. mod(nzA,2) == 1 .and. mod(nyA,2) == 1) flags = ior(flags,FFTW_UNALIGNED)
    if(use2d) then
      plan_xf = fftw_plan_many_dft_r2c(2,[mz,mx],nyA,phys,[mz,mx],1,mx*mz,cxw,[mz,mx/2+1],1,(mx/2+1)*mz,flags)
      plan_xb = fftw_plan_many_dft_c2r(2,[mz,mx],nyA,cxw,[mz,mx/2+1],1,(mx/2+1)*mz,phys,[mz,mx],1,mx*mz,flags)
      if(.not.(c_associated(plan_xf).and.c_associated(plan_xb))) call abort_run('FFTW planning failed')
      return
    end if
    plan_xf = fftw_plan_many_dft_r2c(1,[mx],nzA*nyA,phys,[mx],1,mx,cxw,[mx/2+1],1,mx/2+1,flags)
    plan_xb = fftw_plan_many_dft_c2r(1,[mx],nzA*nyA,cxw,[mx/2+1],1,mx/2+1,phys,[mx],1,mx,flags)
    plan_zf = fftw_plan_many_dft(1,[mz],nyA*nkx,yaw,[mz],1,mz,yaw,[mz],1,mz,FFTW_FORWARD ,FFTW_ESTIMATE)
    plan_zb = fftw_plan_many_dft(1,[mz],nyA*nkx,yaw,[mz],1,mz,yaw,[mz],1,mz,FFTW_BACKWARD,FFTW_ESTIMATE)
    if(.not.(c_associated(plan_xf).and.c_associated(plan_xb).and.c_associated(plan_zf).and.c_associated(plan_zb))) &
      call abort_run('FFTW planning failed')
  end subroutine plan_cpu
#endif
  !
  integer(i8) function fft_buffer_bytes()
    implicit none
    fft_buffer_bytes = 16_i8*(size(bufp,kind=i8)+size(bufa,kind=i8))
  end function fft_buffer_bytes
  !
  subroutine end_fft
    implicit none
#if defined(_OPENACC)
    integer :: istat
    istat = cufftDestroy(plan_xf); istat = cufftDestroy(plan_xb)
    if(.not.use2d) istat = cufftDestroy(plan_z)
#else
    call fftw_destroy_plan(plan_xf); call fftw_destroy_plan(plan_xb)
    if(.not.use2d) then
      call fftw_destroy_plan(plan_zf); call fftw_destroy_plan(plan_zb)
    end if
#endif
    !$acc exit data delete(bufp,bufa)
  end subroutine end_fft
  !
  subroutine fwd_fft(phys,fh)
    !
    ! physical (x-pencil) -> spectral (z-pencil); `phys` may be overwritten
    !
    implicit none
    real(rp)   , intent(inout) :: phys(mx,nzA,nyA)
    complex(rp), intent(out)   :: fh(nkx,nkz,ny)
    call tic(tm_fft)
    if(use2d) then
      call fwd_xz(phys,bufp,bufa)                                   ! phys -> cxw -> ybw
    else
      call fwd_x(phys,bufp,bufa)                                    ! phys -> cxw -> xaw
      call tic(tm_tr); call tr_xa2ya(bufa,bufp); call toc(tm_tr)   ! xaw -> yaw
      call fwd_z(bufp,bufa)                                         ! yaw -> ybw
    end if
    call tic(tm_tr); call tr_yb2zb(bufa,fh); call toc(tm_tr)     ! ybw -> fh
    call toc(tm_fft)
  end subroutine fwd_fft
  !
  subroutine bwd_fft(fh,phys)
    !
    ! spectral (z-pencil) -> physical (x-pencil), with zero padding
    !
    implicit none
    complex(rp), intent(in)  :: fh(nkx,nkz,ny)
    real(rp)   , intent(out) :: phys(mx,nzA,nyA)
    call tic(tm_fft)
    call tic(tm_tr); call tr_zb2yb(fh,bufa); call toc(tm_tr)     ! fh -> ybw
    if(use2d) then
      call bwd_zx(bufa,bufp,phys)                                   ! ybw -> cxw -> phys
    else
      call bwd_z(bufa,bufp)                                         ! ybw -> yaw
      call tic(tm_tr); call tr_ya2xa(bufp,bufa); call toc(tm_tr)   ! yaw -> xaw
      call bwd_x(bufa,bufp,phys)                                    ! xaw -> cxw -> phys
    end if
    call toc(tm_fft)
  end subroutine bwd_fft
  !
  subroutine fwd_xz(phys,cxw,ybw)
    ! 2D transform in x and z (slabs in y), then truncation in x and z and reordering to the y-pencil
    implicit none
    real(rp)   , intent(inout) :: phys(mx,mz,nyA)
    complex(rp), intent(inout) :: cxw(mx/2+1,mz,nyA)
    complex(rp), intent(out)   :: ybw(nz,nyA,nxh)
    integer :: i,j,k,nzh,off
#if defined(_OPENACC)
    integer :: istat
    !$acc host_data use_device(phys,cxw)
    istat = cufftExecD2Z(plan_xf,phys,cxw)
    !$acc end host_data
#else
    call fftw_execute_dft_r2c(plan_xf,phys,cxw)
#endif
    nzh = nz/2
    off = mz-nz
    !$acc parallel loop collapse(3) default(present) async(1)
    do i=1,nxh
      do k=1,nyA
        do j=1,nz
          if(j <= nzh) then
            ybw(j,k,i) = cxw(i,j,k)*norm
          else if(j == nzh+1) then
            ybw(j,k,i) = (0._rp,0._rp)
          else
            ybw(j,k,i) = cxw(i,j+off,k)*norm
          end if
        end do
      end do
    end do
  end subroutine fwd_xz
  !
  subroutine bwd_zx(ybw,cxw,phys)
    ! zero padding in x and z and reordering to the x-pencil, then the 2D transform
    implicit none
    complex(rp), intent(in)    :: ybw(nz,nyA,nxh)
    complex(rp), intent(inout) :: cxw(mx/2+1,mz,nyA)
    real(rp)   , intent(out)   :: phys(mx,mz,nyA)
    integer :: i,j,k,nzh,off
#if defined(_OPENACC)
    integer :: istat
#endif
    nzh = nz/2
    off = mz-nz
    !$acc parallel loop collapse(3) default(present) async(1)
    do k=1,nyA
      do j=1,mz
        do i=1,mx/2+1
          if(i <= nxh .and. j <= nzh) then
            cxw(i,j,k) = ybw(j,k,i)
          else if(i <= nxh .and. j > off+nzh+1) then
            cxw(i,j,k) = ybw(j-off,k,i)
          else
            cxw(i,j,k) = (0._rp,0._rp)
          end if
        end do
      end do
    end do
#if defined(_OPENACC)
    !$acc host_data use_device(phys,cxw)
    istat = cufftExecZ2D(plan_xb,cxw,phys)
    !$acc end host_data
#else
    call fftw_execute_dft_c2r(plan_xb,cxw,phys)
#endif
  end subroutine bwd_zx
  !
  subroutine fwd_x(phys,cxw,xaw)
    implicit none
    real(rp)   , intent(inout) :: phys(mx,nzA,nyA)
    complex(rp), intent(inout) :: cxw(mx/2+1,nzA,nyA)
    complex(rp), intent(out)   :: xaw(nxh,nzA,nyA)
    integer :: i,j,k
#if defined(_OPENACC)
    integer :: istat
    !$acc host_data use_device(phys,cxw)
    istat = cufftExecD2Z(plan_xf,phys,cxw)
    !$acc end host_data
#else
    call fftw_execute_dft_r2c(plan_xf,phys,cxw)
#endif
    !$acc parallel loop collapse(3) default(present) async(1)
    do k=1,nyA
      do j=1,nzA
        do i=1,nxh
          xaw(i,j,k) = cxw(i,j,k)*norm
        end do
      end do
    end do
  end subroutine fwd_x
  !
  subroutine fwd_z(yaw,ybw)
    implicit none
    complex(rp), intent(inout) :: yaw(mz,nyA,nkx)
    complex(rp), intent(out)   :: ybw(nz,nyA,nkx)
    integer :: i,j,k,nzh,off
#if defined(_OPENACC)
    integer :: istat
    !$acc host_data use_device(yaw)
    istat = cufftExecZ2Z(plan_z,yaw,yaw,CUFFT_FORWARD)
    !$acc end host_data
#else
    call fftw_execute_dft(plan_zf,yaw,yaw)
#endif
    nzh = nz/2
    off = mz-nz
    !$acc parallel loop collapse(3) default(present) async(1)
    do i=1,nkx
      do k=1,nyA
        do j=1,nz
          if(j <= nzh) then
            ybw(j,k,i) = yaw(j,k,i)
          else if(j == nzh+1) then
            ybw(j,k,i) = (0._rp,0._rp)
          else
            ybw(j,k,i) = yaw(j+off,k,i)
          end if
        end do
      end do
    end do
  end subroutine fwd_z
  !
  subroutine bwd_z(ybw,yaw)
    implicit none
    complex(rp), intent(in)  :: ybw(nz,nyA,nkx)
    complex(rp), intent(out) :: yaw(mz,nyA,nkx)
    integer :: i,j,k,nzh,off
#if defined(_OPENACC)
    integer :: istat
#endif
    nzh = nz/2
    off = mz-nz
    !$acc parallel loop collapse(3) default(present) async(1)
    do i=1,nkx
      do k=1,nyA
        do j=1,mz
          if(j <= nzh) then
            yaw(j,k,i) = ybw(j,k,i)
          else if(j > off+nzh+1) then
            yaw(j,k,i) = ybw(j-off,k,i)
          else
            yaw(j,k,i) = (0._rp,0._rp)
          end if
        end do
      end do
    end do
#if defined(_OPENACC)
    !$acc host_data use_device(yaw)
    istat = cufftExecZ2Z(plan_z,yaw,yaw,CUFFT_INVERSE)
    !$acc end host_data
#else
    call fftw_execute_dft(plan_zb,yaw,yaw)
#endif
  end subroutine bwd_z
  !
  subroutine bwd_x(xaw,cxw,phys)
    implicit none
    complex(rp), intent(in)    :: xaw(nxh,nzA,nyA)
    complex(rp), intent(inout) :: cxw(mx/2+1,nzA,nyA)
    real(rp)   , intent(out)   :: phys(mx,nzA,nyA)
    integer :: i,j,k
#if defined(_OPENACC)
    integer :: istat
#endif
    !$acc parallel loop collapse(3) default(present) async(1)
    do k=1,nyA
      do j=1,nzA
        do i=1,mx/2+1
          if(i <= nxh) then
            cxw(i,j,k) = xaw(i,j,k)
          else
            cxw(i,j,k) = (0._rp,0._rp)
          end if
        end do
      end do
    end do
#if defined(_OPENACC)
    !$acc host_data use_device(phys,cxw)
    istat = cufftExecZ2D(plan_xb,cxw,phys)
    !$acc end host_data
#else
    call fftw_execute_dft_c2r(plan_xb,cxw,phys)
#endif
  end subroutine bwd_x
end module mod_fft
