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
  use mod_param , only: nxh,nz,ny,mx,mz,abort_run
  use mod_decomp, only: nzA,nyA,nkx,nkz,tr_xa2ya,tr_ya2xa,tr_yb2zb,tr_zb2yb
#if defined(_OPENACC)
  use cufft
  use mod_decomp, only: istream
#endif
  implicit none
#if !defined(_OPENACC)
  include 'fftw3.f03'
#endif
  private
  public :: init_fft,fwd_fft,bwd_fft,end_fft
  complex(rp), allocatable :: cxw(:,:,:),xaw(:,:,:),yaw(:,:,:),ybw(:,:,:)
#if defined(_OPENACC)
  integer :: plan_xf,plan_xb,plan_z
#else
  type(c_ptr) :: plan_xf,plan_xb,plan_zf,plan_zb
#endif
  real(rp) :: norm
contains
  subroutine init_fft(phys)
    implicit none
    real(rp), intent(inout) :: phys(mx,nzA,nyA) ! only used for planning (not modified)
    integer :: istat
    allocate(cxw(mx/2+1,nzA,nyA),xaw(nxh,nzA,nyA),yaw(mz,nyA,nkx),ybw(nz,nyA,nkx))
    !$acc enter data create(cxw,xaw,yaw,ybw)
    norm = 1._rp/(real(mx,rp)*real(mz,rp))
#if defined(_OPENACC)
    istat = cufftPlanMany(plan_xf,1,[mx],[mx],1,mx,[mx/2+1],1,mx/2+1,CUFFT_D2Z,nzA*nyA)
    if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (x, forward)')
    istat = cufftPlanMany(plan_xb,1,[mx],[mx/2+1],1,mx/2+1,[mx],1,mx,CUFFT_Z2D,nzA*nyA)
    if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (x, backward)')
    istat = cufftPlanMany(plan_z ,1,[mz],[mz],1,mz,[mz],1,mz,CUFFT_Z2Z,nyA*nkx)
    if(istat /= CUFFT_SUCCESS) call abort_run('cuFFT plan (z)')
    istat = cufftSetStream(plan_xf,istream)
    istat = cufftSetStream(plan_xb,istream)
    istat = cufftSetStream(plan_z ,istream)
#else
    istat = 0
    plan_xf = fftw_plan_many_dft_r2c(1,[mx],nzA*nyA,phys,[mx],1,mx,cxw,[mx/2+1],1,mx/2+1,FFTW_ESTIMATE)
    plan_xb = fftw_plan_many_dft_c2r(1,[mx],nzA*nyA,cxw,[mx/2+1],1,mx/2+1,phys,[mx],1,mx,FFTW_ESTIMATE)
    plan_zf = fftw_plan_many_dft(1,[mz],nyA*nkx,yaw,[mz],1,mz,yaw,[mz],1,mz,FFTW_FORWARD ,FFTW_ESTIMATE)
    plan_zb = fftw_plan_many_dft(1,[mz],nyA*nkx,yaw,[mz],1,mz,yaw,[mz],1,mz,FFTW_BACKWARD,FFTW_ESTIMATE)
    if(.not.(c_associated(plan_xf).and.c_associated(plan_xb).and.c_associated(plan_zf).and.c_associated(plan_zb))) &
      call abort_run('FFTW planning failed')
#endif
  end subroutine init_fft
  !
  subroutine end_fft
    implicit none
#if defined(_OPENACC)
    integer :: istat
    istat = cufftDestroy(plan_xf); istat = cufftDestroy(plan_xb); istat = cufftDestroy(plan_z)
#else
    call fftw_destroy_plan(plan_xf); call fftw_destroy_plan(plan_xb)
    call fftw_destroy_plan(plan_zf); call fftw_destroy_plan(plan_zb)
#endif
    !$acc exit data delete(cxw,xaw,yaw,ybw)
  end subroutine end_fft
  !
  subroutine fwd_fft(phys,fh)
    !
    ! physical (x-pencil) -> spectral (z-pencil); `phys` may be overwritten
    !
    implicit none
    real(rp)   , intent(inout) :: phys(mx,nzA,nyA)
    complex(rp), intent(out)   :: fh(nkx,nkz,ny)
    integer :: i,j,k,nzh,off
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
    call tr_xa2ya(xaw,yaw)
#if defined(_OPENACC)
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
    call tr_yb2zb(ybw,fh)
  end subroutine fwd_fft
  !
  subroutine bwd_fft(fh,phys)
    !
    ! spectral (z-pencil) -> physical (x-pencil), with zero padding
    !
    implicit none
    complex(rp), intent(in)  :: fh(nkx,nkz,ny)
    real(rp)   , intent(out) :: phys(mx,nzA,nyA)
    integer :: i,j,k,nzh,off
#if defined(_OPENACC)
    integer :: istat
#endif
    call tr_zb2yb(fh,ybw)
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
    call tr_ya2xa(yaw,xaw)
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
  end subroutine bwd_fft
end module mod_fft
