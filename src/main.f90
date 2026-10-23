!
! HiCHAN: DNS of incompressible channel flow with passive scalars
!
!   Fourier series in x and z (3/2 dealiasing), collocated high-order compact finite
!   differences in y, influence-matrix method for the pressure, low-storage RK3:
!   P. Lenaers, P. Schlatter, G. Brethouwer, A.V. Johansson, "A new high-order method
!   for the simulation of incompressible wall-bounded turbulent flows",
!   J. Comput. Phys. 272 (2014) 108-126.
!
!   GPU port: OpenACC + cuFFT + cuDecomp (following CaNS, github.com/CaNS-World/CaNS)
!
program hichan
  use mpi
  use mod_types
  use mod_param
  use mod_decomp
  use mod_compact, only: init_grid,init_compact,yg,kl1,ku1,kl2,ku2,nc1,ncn,lam2max
  use mod_fields
  use mod_fft
  use mod_ysolve
  use mod_rk
  use mod_stats
  use mod_io
  use mod_initflow
  use mod_timer
  implicit none
  integer :: ierr,istep,istep0,ilog,n,nf
  real(rp) :: time,dtcur,divm,t0,t1,tstep,twall0
  logical :: is_done,is_stop
  character(len=8), allocatable :: names(:)
  character(len=4) :: cs
  !
  call MPI_INIT(ierr)
  twall0 = MPI_WTIME()
  call MPI_COMM_RANK(MPI_COMM_WORLD,myid,ierr)
  call read_input(myid)
  call init_decomp
  call init_grid(ny,grid_type,grid_alpha,grid_file)
  call init_compact(ny,fd_order)
  call alloc_fields
  call init_fft(up)
  call init_ysolve
  if(visc_theta > 0._rp) call init_implicit
  if(ns_method == 'vv') call init_vv
  call init_rk
  call init_stats
  if(myid == 0) then
    call execute_command_line('mkdir -p '//trim(data_dir))
    write(*,'(a)')        ' --------------------------------------------------------------'
    write(*,'(a,3i6)')    ' Fourier modes nx, nz / points ny : ',nx,nz,ny
    write(*,'(a,2i6)')    ' physical (dealiased) grid mx, mz : ',mx,mz
    write(*,'(a,es12.4)') ' min. wall-normal spacing         : ',yg(2)-yg(1)
    write(*,'(a,es12.4)') ' max. wall-normal spacing         : ',maxval(yg(2:ny)-yg(1:ny-1))
    write(*,'(a,i3,a,4i3)') ' compact scheme order ',fd_order,', bandwidths of B1, B2: ',kl1,ku1,kl2,ku2
    write(*,'(a,2i6)')    ' extent of the dense rows of D1   : ',nc1,ncn
    write(*,'(a,es12.4)') ' spectral radius of D2            : ',lam2max
    write(*,'(a,es12.4)') ' sum of quadrature weights - 2    : ',sum(wq)-2._rp
    write(*,'(a)')        ' --------------------------------------------------------------'
  end if
  call MPI_BARRIER(MPI_COMM_WORLD,ierr)
  !
  ! initial condition
  !
  if(trim(init_type) == 'restart') then
    call init_fields(u,v,w,s) ! sets scalars not present in the restart file
    call read_restart(trim(restart_file),time,istep,dtcur,u,v,w,s)
    if(reset_time) then
      time = 0._rp; istep = 0
    end if
  else
    call init_fields(u,v,w,s)
    time = 0._rp; istep = 0; dtcur = dt
  end if
  istep0 = istep
  if(ns_method == 'vv') call vv_state_from_velocity(u,v,w)
  !
  if(myid == 0) then
    open(newunit=ilog,file=trim(data_dir)//'/log.dat',position='append')
    write(ilog,'(a)',advance='no') '#    step          time            dt         ubulk  dpdx_applied'// &
                                  '         tau_w        Re_tau     div_max          Ekin        E_mode'
    do n=1,nscal
      write(ilog,'(a,i0,a,i0)',advance='no') '      dS',n,'dy_b      dS',n
      write(ilog,'(a)',advance='no') 'dy_t'
    end do
    write(ilog,*)
  end if
  divm = divergence_max(u,v,w,tmp1)
  call log_line(ilog,istep,time,dtcur,divm,u,v,w,s,dpdx_applied)
  if(myid == 0) call log_line_stdout()
  !
  ! time integration
  !
  is_done = .false.
  t0 = MPI_WTIME()
  call timing_start
  do while(.not.is_done)
    call tic(tm_rk)
    call rk_step(time,dtcur)
    call toc(tm_rk)
    istep = istep+1
    is_done = (istep >= nstep) .or. (time >= tmax)
    if(mod(istep,iout_log) == 0 .or. is_done) then
      call tic(tm_log)
      divm = divergence_max(u,v,w,tmp1)
      call log_line(ilog,istep,time,dtcur,divm,u,v,w,s,dpdx_applied)
      if(myid == 0) call log_line_stdout()
      if(divm /= divm .or. divm > 1.e3_rp) then
        if(myid == 0) write(*,'(a)') ' HiCHAN: the simulation diverged (NaN or huge divergence)'
        call MPI_ABORT(MPI_COMM_WORLD,2,ierr)
      end if
      is_stop = (MPI_WTIME()-twall0 > wall_time_max)
      call MPI_BCAST(is_stop,1,MPI_LOGICAL,0,MPI_COMM_WORLD,ierr)
      if(is_stop) then
        if(myid == 0) write(*,'(a)') ' HiCHAN: wall-clock limit reached'
        is_done = .true.
      end if
      call toc(tm_log)
    end if
    call tic(tm_stats)
    if(time >= tstats_start .and. iout_stats > 0) then
      if(mod(istep,iout_stats) == 0) call accumulate_stats(u,v,w,s,1._rp/(rk_bet(3)*dtcur),dtcur*iout_stats)
    end if
    if(iout_save_stats > 0) then
      if(mod(istep,iout_save_stats) == 0 .or. is_done) call write_stats(time)
    end if
    call toc(tm_stats)
    call tic(tm_io)
    if(iout_restart > 0) then
      if(mod(istep,iout_restart) == 0 .or. is_done) &
        call write_restart(trim(data_dir)//'/restart',time,istep,dtcur,u,v,w,s)
    end if
    if(iout_field > 0) then
      if(mod(istep,iout_field) == 0) call output_fields()
    end if
    call toc(tm_io)
    call timing_step(istep,time,dtcur,is_done)
  end do
  !$acc wait
  t1 = MPI_WTIME()
  call timing_report(istep-istep0,real(nx,rp)*ny*nz)
  tstep = (t1-t0)/max(istep-istep0,1)
  if(myid == 0) then
    write(*,'(a,es12.4,a)') ' wall-clock time per time step: ',tstep,' s'
    write(*,'(a,es12.4,a)') ' per time step and grid point : ',tstep/(real(nx,rp)*ny*nz)*nproc,' s x #processes'
    close(ilog)
  end if
  call end_fft
  call finalize_decomp
  call MPI_FINALIZE(ierr)
contains
  subroutine log_line_stdout
    write(*,'(a,i9,a,es12.5,a,es10.3,a,es12.5,a,es10.3)',advance='no') ' step ',istep,'  t = ',time, &
                                                         '  dt = ',dtcur,'  dpdx = ',dpdx_applied,'  div = ',divm
    if(visc_theta > 0._rp) write(*,'(a,i3,a,es9.2)',advance='no') '  impl. it. ',impl_nit,' dv ',impl_res
    write(*,*)
  end subroutine log_line_stdout
  !
  subroutine output_fields
    nf = 3+nscal
    if(allocated(names)) deallocate(names)
    allocate(names(nf))
    names(1:3) = ['u','v','w']
    call bwd_fft(u,ap); call write_field(istep,'u',ap)
    call bwd_fft(v,ap); call write_field(istep,'v',ap)
    call bwd_fft(w,ap); call write_field(istep,'w',ap)
    do n=1,nscal
      write(cs,'(i0)') n
      names(3+n) = 's'//trim(cs)
      call bwd_fft(s(:,:,:,n),ap); call write_field(istep,names(3+n),ap)
    end do
    call write_xdmf(istep,time,names,yg)
  end subroutine output_fields
end program hichan
