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
  character(len=3) :: ns_method = 'im'     ! 'im': primitive variables + influence matrix (Lenaers et al. 2014)
                                           ! 'vv': velocity-vorticity, phi = lap(v) and omega_y, implicit
                                           !       viscous terms (Lluesma-Rodriguez et al., Mathematics 2021)
  character(len=8) :: vv_neumann = 'explicit' ! 'vv': wall derivative for dv/dy = 0: 'explicit' (6-point one-
                                           !   sided, 5th order, more accurate) or 'compact' (row of D1:
                                           !   divergence exactly zero also at the walls)
  real(rp) :: visc_theta = 0._rp          ! theta of the viscous term (0: explicit, Appendix B.1; >0: implicit, B.2)
  integer  :: impl_maxit = 10              ! max. iterations of the implicit pressure/velocity coupling (B.2)
  real(rp) :: impl_tol = 1.e-12_rp         ! convergence: max|v^{m+1}-v^m| < impl_tol*max|v|
  character(len=16) :: nl_form = 'rotational' ! 'rotational' (energy conserving, 9 FFTs), 'advective' (as in the paper,
                                              ! 15 FFTs) or 'divergence' (div(uu), conserves momentum, 9 FFTs)
  !
  ! &time
  !
  integer  :: nstep = 1000
  real(rp) :: tmax  = huge(1._rp)
  real(rp) :: wall_time_max = huge(1._rp)  ! in seconds; the run stops (with a restart file) before this
  real(rp) :: dt    = 1.e-5_rp              ! fixed time step, or the first step of an adaptive run
  logical  :: fixed_dt = .false.
  real(rp) :: dt_max = 1._rp                ! upper bound of the adaptive time step
  real(rp) :: dt_growth = 1.1_rp            ! adaptive dt may grow by at most this factor per step
  real(rp) :: cfl = 1.0_rp, cfl_visc = 1.5_rp ! RK3: |lambda dt| < sqrt(3) (imag.) and 2.51 (real)
  !
  ! &init
  !
  character(len=16) :: init_type = 'laminar' ! 'laminar', 'noise', 'restart', 'mode'
  real(rp) :: init_amp = 0.1_rp              ! 'noise'/'mode': amplitude; 'largescale': rms of the large scales
  real(rp) :: init_noise = 0.01_rp           ! 'largescale': rms of the superposed small-scale noise
  integer  :: ls_nmin = 10, ls_nmax = 40     ! 'largescale': wavelengths L/ls_nmax ... L/ls_nmin in x and z
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
  logical  :: timing = .false.       ! timers: report at the end, time per step in timing_steps.dat
  logical  :: timing_sync = .true.   ! timers wait for the GPU (exact attribution, small overhead)
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
  character(len=12) :: scal_form = 'advective' ! scalar advection: 'advective' u.grad(s), 'divergence'
                                               ! div(u s), or 'skew' (average of both: conserves the
                                               ! scalar variance, more robust, 8 instead of 4 FFTs)
  !
  ! &buoyancy (Boussinesq): momentum source  - sum_n ri(n) * s_n * e_g
  !
  real(rp) :: gravity(3) = [0._rp,-1._rp,0._rp]  ! direction of gravity (x,y,z), normalised internally
  real(rp) :: ri(nscal_max) = 0._rp              ! Richardson number of scalar n (0 -> passive scalar)
  !
  ! &parallel
  !
  integer :: pdims(2) = [0,0]              ! process grid (0,0 -> automatic / cuDecomp autotuning)
  character(len=12) :: transpose_lib = 'auto' ! 'auto', 'cudecomp', 'diezdecomp' or 'mpi'
  character(len=8)  :: transpose_prec = 'double' ! data sent in the transposes: 'double' or 'single'
                                                 ! (rounded to single precision; computation stays double)
  integer :: cudecomp_t_backend = 0        ! 0 -> autotune, otherwise CUDECOMP_TRANSPOSE_COMM_* value
  integer :: tr_batch = 1                  ! fields per transpose message in the batched transforms
                                           ! (1: one field per message)
  logical :: tr_overlap = .false.          ! pipeline the batched transforms: non-blocking transposes of one
                                           ! group of fields overlap the FFTs of the previous group
  logical :: cudecomp_enable_nccl = .true., cudecomp_enable_nvshmem = .false.
  !
  ! derived quantities
  !
  integer  :: nxh, mx, mz                  ! number of kx modes kept; physical (dealiased) grid sizes
  real(rp) :: visc
  real(rp) :: diff(nscal_max)
  real(rp) :: gvec(3) = 0._rp              ! unit vector in the direction of gravity
  logical  :: is_buoyant = .false.         ! at least one active scalar
  !
  ! low-storage RK3 coefficients (Table 1)
  !
  real(rp), parameter :: rk_gam(3) = [ 8._rp/15._rp, 5._rp/12._rp,  3._rp/4._rp]
  real(rp), parameter :: rk_zet(3) = [ 0._rp      ,-17._rp/60._rp,-5._rp/12._rp]
  real(rp), parameter :: rk_bet(3) = rk_gam(:) + rk_zet(:)
  !
  ! implicit/explicit split of the linear terms in the 'vv' method (Spalart, Moser & Rogers 1991,
  ! Eqs. (30)-(32) of Lluesma-Rodriguez et al. 2021): (I - b_q dt L) f^{q+1} = (I + a_q dt L) f^q + ...
  !
  real(rp), parameter :: rk_asmr(3) = [29._rp/96._rp, -3._rp/40._rp, 1._rp/6._rp]
  real(rp), parameter :: rk_bsmr(3) = [37._rp/160._rp, 5._rp/24._rp, 1._rp/6._rp]
contains
  subroutine read_input(myid)
    implicit none
    integer, intent(in) :: myid
    integer :: iunit,ios
    namelist /grid/ nx,ny,nz,lx,lz,grid_type,grid_alpha,grid_file,fd_order,dealias
    namelist /physics/ re,forcing,ubulk,dpdx,ns_method,vv_neumann,nl_form,visc_theta,impl_maxit,impl_tol
    namelist /time/ nstep,tmax,wall_time_max,dt,fixed_dt,dt_max,dt_growth,cfl,cfl_visc
    namelist /init/ init_type,init_amp,init_noise,ls_nmin,ls_nmax,init_seed,mode_file,mode_kx,mode_kz,restart_file,reset_time
    namelist /output/ data_dir,iout_log,iout_stats,iout_save_stats,iout_restart,iout_field,tstats_start, &
                      mode_log_kx,mode_log_kz,timing,timing_sync
    namelist /scalars/ nscal,pr,scal_theta,sbc_bot,sbc_top,sval_bot,sval_top,ssrc_const,ssrc_u,sinit,scal_form
    namelist /buoyancy/ gravity,ri
    namelist /parallel/ pdims,transpose_lib,transpose_prec,cudecomp_t_backend,cudecomp_enable_nccl,cudecomp_enable_nvshmem, &
                       tr_batch,tr_overlap
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
    read(iunit,nml=buoyancy,iostat=ios); if(ios > 0 ) call abort_run('error reading namelist &buoyancy')
    rewind(iunit)
    read(iunit,nml=parallel,iostat=ios); if(ios > 0 ) call abort_run('error reading namelist &parallel')
    close(iunit)
    !
    if(mod(nx,2) /= 0 .or. mod(nz,2) /= 0) call abort_run('nx and nz must be even')
    if(ny < 9) call abort_run('ny must be at least 9')
    if(all(fd_order /= [4,6])) call abort_run('fd_order must be 4 or 6')
    if(dt <= 0._rp) call abort_run('dt must be > 0')
    if(ls_nmin < 1 .or. ls_nmax < ls_nmin) call abort_run('&init: 1 <= ls_nmin <= ls_nmax required')
    if(dt_growth < 1._rp) call abort_run('dt_growth must be >= 1')
    if(all(ns_method /= ['im ','vv '])) call abort_run("ns_method must be 'im' or 'vv'")
    if(all(vv_neumann /= ['explicit','compact '])) call abort_run("vv_neumann must be 'explicit' or 'compact'")
    if(visc_theta < 0._rp .or. visc_theta > 1._rp) call abort_run('0 <= visc_theta <= 1 required')
    if(any(scal_theta(:) < 0._rp .or. scal_theta(:) > 1._rp)) call abort_run('0 <= scal_theta <= 1 required')
    if(nscal < 0 .or. nscal > nscal_max) call abort_run('0 <= nscal <= 16 required')
    if(all(forcing /= ['cfr','cpg'])) call abort_run('forcing must be cfr or cpg')
    if(all(nl_form /= ['advective ','rotational','divergence'])) &
      call abort_run("nl_form must be 'rotational', 'advective' or 'divergence'")
    if(all(transpose_prec /= ['double','single'])) call abort_run("transpose_prec must be 'double' or 'single'")
    if(tr_batch < 1) call abort_run('tr_batch >= 1 required')
    if(all(scal_form /= ['advective ','divergence','skew      '])) &
      call abort_run("scal_form must be 'advective', 'divergence' or 'skew'")
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
    !
    ! buoyancy
    !
    is_buoyant = any(ri(1:nscal) /= 0._rp)
    if(is_buoyant) then
      if(norm2(gravity) == 0._rp) call abort_run('&buoyancy: gravity must be a non-zero vector')
      gvec(:) = gravity(:)/norm2(gravity)
    end if
    if(myid == 0) call print_setup
  contains
    subroutine print_setup
      integer :: n
      character(len=*), parameter :: f1 = '(1x,a,t36,a)', fi = '(1x,a,t36,3(i0,:,1x))', fr = '(1x,a,t36,4(es12.5,:,1x))'
      write(*,'(a)') ' ================================ HiCHAN run set-up ================================'
      write(*,fi) 'Fourier modes nx, nz / points ny',nx,nz,ny
      write(*,fr) 'box lx, lz (half-height h = 1)',lx,lz
      write(*,f1) 'wall-normal grid',trim(grid_type)
      if(trim(grid_type) == 'tanh') write(*,fr) '  stretching alpha_s',grid_alpha
      write(*,fi) 'compact scheme order',fd_order
      write(*,f1) '3/2 dealiasing',merge('yes','no ',dealias)
      write(*,fr) 'Reynolds number re',re
      if(forcing == 'cfr') then
        write(*,fr) 'constant flow rate, ubulk',ubulk
      else
        write(*,fr) 'constant pressure gradient -dp/dx',dpdx
      end if
      if(ns_method == 'vv') then
        write(*,f1) 'Navier-Stokes solver','velocity-vorticity (phi, omega_y), implicit viscous (SMR RK3)'
      else
        write(*,f1) 'Navier-Stokes solver','primitive variables + influence matrix'
      end if
      write(*,f1) 'nonlinear term',trim(nl_form)
      if(ns_method == 'vv') then
        write(*,f1) 'wall derivative in dv/dy = 0',trim(vv_neumann)
      else if(visc_theta > 0._rp) then
        write(*,fr) 'viscous term implicit, theta',visc_theta
      else
        write(*,f1) 'viscous term','explicit (RK3)'
      end if
      if(fixed_dt) then
        write(*,fr) 'fixed time step dt',dt
      else
        write(*,fr) 'adaptive time step, cfl, cfl_visc',cfl,cfl_visc
      end if
      write(*,fi) 'nstep',nstep
      if(tmax < huge(1._rp)) write(*,fr) 'tmax',tmax
      write(*,f1) 'initial condition',trim(init_type)
      if(trim(init_type) == 'largescale') then
        write(*,fi) '  wavelengths L/n, n from ... to',ls_nmin,min(ls_nmax,nx/2-1),min(ls_nmax,nz/2-1)
        write(*,fr) '  rms large scales, rms noise',init_amp,init_noise
      end if
      write(*,fi) 'number of scalars',nscal
      if(nscal > 0) write(*,f1) 'scalar advection form',trim(scal_form)
      do n=1,nscal
        write(*,'(1x,a,i0,a,es10.3,a,f5.2,a,a1,a,es10.3,a,a1,a,es10.3,a,es10.3)') &
          '  scalar ',n,': Pr = ',pr(n),', theta = ',scal_theta(n),', bottom ',sbc_bot(n),' = ',sval_bot(n), &
          ', top ',sbc_top(n),' = ',sval_top(n),', Ri = ',ri(n)
      end do
      if(is_buoyant) write(*,fr) 'gravity direction (unit vector)',gvec
      if(timing) write(*,f1) 'timers (timing_report.txt etc.)',merge('on, GPU-synchronised','on                  ', &
                                                                     timing_sync)
      write(*,'(a)') ' ===================================================================================='
    end subroutine print_setup
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
