!
! input parameters (namelist file input.nml) and derived quantities
!
module mod_param
  use mpi
  use mod_types
  implicit none
  public
  integer, parameter :: nscal_max = 16
  !
  ! &grid
  !
  integer  :: nx = 64, ny = 65, nz = 64 ! Fourier modes in x and z (even), points in y (walls included)
  real(rp) :: lx = 2*pi, lz = pi        ! box size, the channel half-height is 1 (y in [-1,1])
  character(len=16)  :: grid_type = 'tanh' ! 'tanh' (Orlandi stretching), 'uniform' or 'file'
  real(rp) :: grid_alpha = 2._rp           ! stretching parameter alpha_s of Eq. (41)
  character(len=256) :: grid_file = 'grid.in'
  integer  :: fd_order = 6                 ! 4 -> (3-4-3), 6 -> (3-5-6-5-3) compact schemes
  logical  :: dealias = .true.             ! 3/2 rule in x and z
  !
  ! &physics
  !
  real(rp) :: re = 2800._rp                ! Reynolds number U_ref*h/nu
  character(len=3)  :: forcing = 'cfr'     ! 'cfr' (constant flow rate) or 'cpg' (constant pressure gradient)
  real(rp) :: ubulk = 1._rp                ! bulk velocity for 'cfr'
  real(rp) :: dpdx  = 0._rp                ! -dp/dx for 'cpg' (=1 with re=Re_tau in wall units)
  real(rp) :: visc_theta = 0._rp          ! theta of the viscous term (0: explicit, Appendix B.1; >0: implicit, B.2)
  integer  :: impl_maxit = 10              ! max. iterations of the implicit pressure/velocity coupling (B.2)
  real(rp) :: impl_tol = 1.e-12_rp         ! convergence: max|v^{m+1}-v^m| < impl_tol*max|v|
  character(len=16) :: nl_form = 'rotational' ! 'rotational' (energy conserving, 9 FFTs) or 'advective' (as in the paper, 15 FFTs)
  !
  ! &time
  !
  integer  :: nstep = 1000
  real(rp) :: tmax  = huge(1._rp)
  real(rp) :: wall_time_max = huge(1._rp)  ! in seconds; the run stops (with a restart file) before this
  real(rp) :: dt    = 1.e-3_rp
  logical  :: fixed_dt = .false.
  real(rp) :: cfl = 1.0_rp, cfl_visc = 1.5_rp ! RK3: |lambda dt| < sqrt(3) (imag.) and 2.51 (real)
  !
  ! &init
  !
  character(len=16) :: init_type = 'laminar' ! 'laminar', 'noise', 'restart', 'mode'
  real(rp) :: init_amp = 0.1_rp
  integer  :: init_seed = 1
  character(len=256) :: mode_file = 'mode.in'
  integer  :: mode_kx = 1, mode_kz = 0
  character(len=256) :: restart_file = 'restart'
  logical  :: reset_time = .false.
  !
  ! &output
  !
  character(len=256) :: data_dir = 'data'
  integer  :: iout_log = 10, iout_stats = 10, iout_save_stats = 1000, iout_restart = 5000, iout_field = 0
  real(rp) :: tstats_start = 0._rp
  integer  :: mode_log_kx = -1, mode_log_kz = 0 ! log the amplitude of one Fourier mode (e.g. for TS waves)
  !
  ! &scalars
  !
  integer  :: nscal = 0
  real(rp) :: pr(nscal_max) = 1._rp              ! Prandtl/Schmidt numbers
  real(rp) :: scal_theta(nscal_max) = 0.5_rp     ! theta of the theta-method for diffusion (0 -> explicit RK3)
  character(len=1) :: sbc_bot(nscal_max) = 'D', sbc_top(nscal_max) = 'D' ! 'D' Dirichlet or 'N' Neumann
  real(rp) :: sval_bot(nscal_max) = 1._rp, sval_top(nscal_max) = -1._rp ! wall value or wall-normal gradient
  real(rp) :: ssrc_const(nscal_max) = 0._rp      ! volumetric source  Q_c
  real(rp) :: ssrc_u(nscal_max) = 0._rp          ! source  Q_u*u/ubulk (e.g. uniform-heat-flux channels)
  character(len=16) :: sinit(nscal_max) = 'linear' ! 'linear', 'zero', 'noise'
  !
  ! &parallel
  !
  integer :: pdims(2) = [0,0]              ! process grid (0,0 -> automatic / cuDecomp autotuning)
  integer :: cudecomp_t_backend = 0        ! 0 -> autotune, otherwise CUDECOMP_TRANSPOSE_COMM_* value
  logical :: cudecomp_enable_nccl = .true., cudecomp_enable_nvshmem = .true.
  !
  ! derived quantities
  !
  integer  :: nxh, mx, mz                  ! number of kx modes kept; physical (dealiased) grid sizes
  real(rp) :: visc
  real(rp) :: diff(nscal_max)
  !
  ! low-storage RK3 coefficients (Table 1)
  !
  real(rp), parameter :: rk_gam(3) = [ 8._rp/15._rp, 5._rp/12._rp,  3._rp/4._rp]
  real(rp), parameter :: rk_zet(3) = [ 0._rp      ,-17._rp/60._rp,-5._rp/12._rp]
  real(rp), parameter :: rk_bet(3) = rk_gam(:) + rk_zet(:)
contains
  subroutine read_input(myid)
    implicit none
    integer, intent(in) :: myid
    integer :: iunit,ios
    namelist /grid/ nx,ny,nz,lx,lz,grid_type,grid_alpha,grid_file,fd_order,dealias
    namelist /physics/ re,forcing,ubulk,dpdx,nl_form,visc_theta,impl_maxit,impl_tol
    namelist /time/ nstep,tmax,wall_time_max,dt,fixed_dt,cfl,cfl_visc
    namelist /init/ init_type,init_amp,init_seed,mode_file,mode_kx,mode_kz,restart_file,reset_time
    namelist /output/ data_dir,iout_log,iout_stats,iout_save_stats,iout_restart,iout_field,tstats_start, &
                      mode_log_kx,mode_log_kz
    namelist /scalars/ nscal,pr,scal_theta,sbc_bot,sbc_top,sval_bot,sval_top,ssrc_const,ssrc_u,sinit
    namelist /parallel/ pdims,cudecomp_t_backend,cudecomp_enable_nccl,cudecomp_enable_nvshmem
    !
    open(newunit=iunit,file='input.nml',status='old',action='read',iostat=ios)
    if(ios /= 0) call abort_run('cannot open input.nml')
    read(iunit,nml=grid    ,iostat=ios); if(ios /= 0) call abort_run('error reading namelist &grid')
    rewind(iunit)
    read(iunit,nml=physics ,iostat=ios); if(ios /= 0) call abort_run('error reading namelist &physics')
    rewind(iunit)
    read(iunit,nml=time    ,iostat=ios); if(ios /= 0) call abort_run('error reading namelist &time')
    rewind(iunit)
    read(iunit,nml=init    ,iostat=ios); if(ios /= 0) call abort_run('error reading namelist &init')
    rewind(iunit)
    read(iunit,nml=output  ,iostat=ios); if(ios /= 0) call abort_run('error reading namelist &output')
    rewind(iunit)
    read(iunit,nml=scalars ,iostat=ios); if(ios > 0 ) call abort_run('error reading namelist &scalars')
    rewind(iunit)
    read(iunit,nml=parallel,iostat=ios); if(ios > 0 ) call abort_run('error reading namelist &parallel')
    close(iunit)
    !
    if(mod(nx,2) /= 0 .or. mod(nz,2) /= 0) call abort_run('nx and nz must be even')
    if(ny < 9) call abort_run('ny must be at least 9')
    if(all(fd_order /= [4,6])) call abort_run('fd_order must be 4 or 6')
    if(visc_theta < 0._rp .or. visc_theta > 1._rp) call abort_run('0 <= visc_theta <= 1 required')
    if(any(scal_theta(:) < 0._rp .or. scal_theta(:) > 1._rp)) call abort_run('0 <= scal_theta <= 1 required')
    if(nscal < 0 .or. nscal > nscal_max) call abort_run('0 <= nscal <= 16 required')
    if(all(forcing /= ['cfr','cpg'])) call abort_run('forcing must be cfr or cpg')
    if(all(nl_form /= ['advective ','rotational'])) call abort_run('nl_form must be advective or rotational')
    nxh = nx/2
    if(dealias) then
      mx = 3*nx/2
      mz = 3*nz/2
    else
      mx = nx
      mz = nz
    end if
    visc = 1._rp/re
    diff(:) = visc/pr(:)
    if(myid == 0) then
      write(*,nml=grid)
      write(*,nml=physics)
      write(*,nml=time)
      write(*,nml=init)
      write(*,nml=output)
      write(*,nml=scalars)
      write(*,nml=parallel)
    end if
  end subroutine read_input
  !
  subroutine abort_run(msg)
    implicit none
    character(len=*), intent(in) :: msg
    integer :: ierr
    write(*,'(a)') 'HiCHAN ERROR: '//trim(msg)
    flush(6)
    call MPI_ABORT(MPI_COMM_WORLD,1,ierr)
    stop 1
  end subroutine abort_run
end module mod_param
