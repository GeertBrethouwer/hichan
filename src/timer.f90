!
! optional timers (timing = .true. in &output):
!   * accumulated wall-clock time of the main parts of the code, reported at the end of the run
!     (stdout and data_dir/timing_report.txt), with the min/mean/max over the MPI ranks
!   * wall-clock time of every time step (max/min/mean over the ranks) and of its main parts,
!     written to data_dir/timing_steps.dat, to look for spikes in the performance
!
! GPU kernels and transposes run asynchronously; with timing_sync = .true. every timer waits for the
! GPU (!$acc wait) when it starts and stops, so that the time is attributed to the right part.
! This adds a little synchronisation overhead. The time per step is exact in either case (each RK
! step ends with a wait).
!
module mod_timer
  use mpi
  use mod_types
  use mod_param , only: timing,timing_sync,data_dir
  use mod_decomp, only: myid
  implicit none
  private
  integer, parameter, public :: tm_loop = 1, tm_rk = 2, tm_nl = 3, tm_dt = 4, tm_vsolve = 5, &
                                tm_ssolve = 6, tm_fft = 7, tm_tr = 8, tm_log = 9, tm_stats = 10, &
                                tm_io = 11
  integer, parameter :: ntm = 11
  character(len=44), parameter :: tname(ntm) = [character(len=44) :: &
    'time loop (total)'                           , &
    '  RK3 steps'                                 , &
    '    nonlinear terms (incl. FFTs, buoyancy)'  , &
    '      time-step limit (CFL, MPI_Allreduce)'  , &
    '    velocity solve (pressure / vv step)'     , &
    '    scalar solves'                           , &
    'FFTs incl. transposes (all calls)'           , &
    '  transposes (communication, all calls)'     , &
    'log output (divergence, log.dat)'            , &
    'statistics (accumulate, write)'              , &
    'restart and field files'                     ]
  real(rp) :: ttot(ntm) = 0._rp, tbeg(ntm) = 0._rp
  integer  :: ncall(ntm) = 0
  !
  ! per-step records: step, t, dt, then (rank-local) step time and the time in fft, transposes,
  ! velocity solve, scalar solves and output during the step
  !
  integer, parameter :: ncol = 6, nbuf = 200
  integer, parameter :: rec_tm(ncol-1) = [tm_fft,tm_tr,tm_vsolve,tm_ssolve,tm_io]
  real(rp) :: buf(ncol,nbuf),bufh(3,nbuf),tprev(ntm),tmark
  integer  :: nb = 0, ipt = -1
  real(rp), allocatable :: hist(:,:)   ! rank 0: step, t_max, t_min, t_mean and parts (max) of all steps
  integer  :: nhist = 0
  public :: tic,toc,timing_start,timing_step,timing_report
contains
  subroutine tic(id)
    integer, intent(in) :: id
    if(.not.timing) return
    if(timing_sync) then
      !$acc wait
    end if
    tbeg(id) = MPI_WTIME()
  end subroutine tic
  !
  subroutine toc(id)
    integer, intent(in) :: id
    if(.not.timing) return
    if(timing_sync) then
      !$acc wait
    end if
    ttot(id) = ttot(id) + (MPI_WTIME()-tbeg(id))
    ncall(id) = ncall(id) + 1
  end subroutine toc
  !
  subroutine timing_start
    ! call just before the time loop
    if(.not.timing) return
    !$acc wait
    if(myid == 0) then
      open(newunit=ipt,file=trim(data_dir)//'/timing_steps.dat',status='replace')
      write(ipt,'(a)') '# wall-clock time per time step [s]: max/min/mean over the MPI ranks, and the time'// &
                       ' (max over the ranks) spent in parts of the step'
      write(ipt,'(a)') '#    step          time            dt    t_step_max    t_step_min   t_step_mean'// &
                       '      t_fft+tr   t_transpose      t_vsolve      t_ssolve      t_output'
      allocate(hist(ncol+3,1024))
    end if
    tprev(:) = ttot(:)
    tmark = MPI_WTIME()
    call tic(tm_loop)
  end subroutine timing_start
  !
  subroutine timing_step(istep,time,dt,is_last)
    ! call at the end of every time step (by all ranks)
    integer , intent(in) :: istep
    real(rp), intent(in) :: time,dt
    logical , intent(in) :: is_last
    real(rp) :: tnow
    integer  :: i
    if(.not.timing) return
    !$acc wait
    tnow = MPI_WTIME()
    nb = nb+1
    bufh(:,nb) = [real(istep,rp),time,dt]
    buf(1,nb) = tnow-tmark
    do i=1,ncol-1
      buf(1+i,nb) = ttot(rec_tm(i))-tprev(rec_tm(i))
    end do
    tprev(:) = ttot(:)
    if(nb == nbuf .or. is_last) call flush_steps()
    tmark = MPI_WTIME()   ! the flush is not counted in the next step
  end subroutine timing_step
  !
  subroutine flush_steps
    real(rp) :: bmax(ncol,nbuf),bmin(nbuf),bsum(nbuf)
    real(rp), allocatable :: tmp(:,:)
    integer :: ierr,nproc_,j
    call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc_,ierr)
    call MPI_REDUCE(buf,bmax,ncol*nb,MPI_DOUBLE_PRECISION,MPI_MAX,0,MPI_COMM_WORLD,ierr)
    call MPI_REDUCE(buf(1,1:nb),bmin,nb,MPI_DOUBLE_PRECISION,MPI_MIN,0,MPI_COMM_WORLD,ierr)
    call MPI_REDUCE(buf(1,1:nb),bsum,nb,MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,ierr)
    if(myid == 0) then
      do j=1,nb
        write(ipt,'(i9,2es14.6,8es14.5)') nint(bufh(1,j)),bufh(2,j),bufh(3,j),bmax(1,j),bmin(j), &
                                          bsum(j)/nproc_,bmax(2:ncol,j)
        if(nhist == size(hist,2)) then
          allocate(tmp(ncol+3,2*nhist)); tmp(:,1:nhist) = hist(:,1:nhist)
          call move_alloc(tmp,hist)
        end if
        nhist = nhist+1
        hist(:,nhist) = [bufh(1,j),bmax(1,j),bmin(j),bsum(j)/nproc_,bmax(2:ncol,j)]
      end do
      flush(ipt)
    end if
    nb = 0
  end subroutine flush_steps
  !
  subroutine timing_report(nsteps,npoints)
    ! call after the time loop (by all ranks)
    integer , intent(in) :: nsteps
    real(rp), intent(in) :: npoints
    real(rp) :: tmin(ntm),tmax(ntm),tsum(ntm),tloop,med,s
    real(rp), allocatable :: ts(:)
    integer :: ierr,nproc_,i,j,u(2),iu,nspike,ntop
    integer, allocatable :: idx(:)
    if(.not.timing) return
    call toc(tm_loop)
    call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc_,ierr)
    call MPI_REDUCE(ttot,tmin,ntm,MPI_DOUBLE_PRECISION,MPI_MIN,0,MPI_COMM_WORLD,ierr)
    call MPI_REDUCE(ttot,tmax,ntm,MPI_DOUBLE_PRECISION,MPI_MAX,0,MPI_COMM_WORLD,ierr)
    call MPI_REDUCE(ttot,tsum,ntm,MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,ierr)
    if(myid /= 0) return
    close(ipt)
    open(newunit=u(2),file=trim(data_dir)//'/timing_report.txt',status='replace')
    u(1) = 6
    tloop = max(tsum(tm_loop)/nproc_,tiny(1._rp))
    do iu=1,2
      write(u(iu),'(a)') ' ======================================================================================='
      write(u(iu),'(a,i0,a,i0,a,l1)') ' HiCHAN timing report: ',nsteps,' time steps, ',nproc_, &
                                       ' MPI ranks, timing_sync = ',timing_sync
      write(u(iu),'(a)') ' wall-clock times in s; min/mean/max over the ranks; % of the mean time-loop time'
      write(u(iu),'(a)') ' ---------------------------------------------------------------------------------------'
      write(u(iu),'(a44,a8,3a11,a8,a11)') 'part of the code','calls','min','mean','max','%','ms/step'
      do i=1,ntm
        write(u(iu),'(a44,i8,3f11.3,f8.1,f11.3)') tname(i),ncall(i),tmin(i),tsum(i)/nproc_,tmax(i), &
                                                  100*tsum(i)/nproc_/tloop,1000*tsum(i)/nproc_/max(nsteps,1)
        if(i == tm_ssolve) then
          s = (tsum(tm_rk)-tsum(tm_nl)-tsum(tm_vsolve)-tsum(tm_ssolve))/nproc_
          write(u(iu),'(a44,8x,11x,f11.3,11x,f8.1,f11.3)') '    rest (RK updates, right-hand sides)',s, &
                                                           100*s/tloop,1000*s/max(nsteps,1)
        end if
      end do
      s = (tsum(tm_loop)-tsum(tm_rk)-tsum(tm_log)-tsum(tm_stats)-tsum(tm_io))/nproc_
      write(u(iu),'(a44,8x,11x,f11.3,11x,f8.1,f11.3)') 'rest of the time loop',s,100*s/tloop,1000*s/max(nsteps,1)
      write(u(iu),'(a)') ' (FFTs and transposes include the calls in the nonlinear terms, statistics and output)'
      if(.not.timing_sync) write(u(iu),'(a)') ' (timing_sync = F: GPU work may be attributed to a later part)'
    end do
    !
    ! time per step: distribution and spikes
    !
    if(nhist > 0) then
      allocate(ts(nhist)); ts(:) = hist(2,1:nhist)
      med = median(ts)
      nspike = count(ts > 2*med)
      ntop = min(10,nhist)
      allocate(idx(ntop))
      do j=1,ntop
        idx(j) = maxloc(ts,1); ts(idx(j)) = -1._rp
      end do
      ts(:) = hist(2,1:nhist)
      do iu=1,2
        write(u(iu),'(a)') ' ---------------------------------------------------------------------------------------'
        write(u(iu),'(a)') ' time per step (slowest rank), see timing_steps.dat:'
        write(u(iu),'(a,4es11.3)') '   min / median / mean / max [s] : ',minval(ts),med,sum(ts)/nhist,maxval(ts)
        write(u(iu),'(a,es11.3,a)') '   per step and grid point       : ',sum(ts)/nhist/npoints*nproc_, &
                                    ' s x #processes'
        write(u(iu),'(a,i0,a,f6.2,a)') '   steps slower than 2 x median  : ',nspike,' (',100._rp*nspike/nhist,' %)'
        write(u(iu),'(a)') '   slowest steps [s]:     step     t_step   fft+tr  transp.   vsolve   ssolve   output'
        do j=1,ntop
          write(u(iu),'(17x,i12,6f9.4)') nint(hist(1,idx(j))),hist(2,idx(j)),hist(5:9,idx(j))
        end do
        write(u(iu),'(a)') ' ======================================================================================='
      end do
    end if
    close(u(2))
  end subroutine timing_report
  !
  real(rp) function median(a)
    real(rp), intent(in) :: a(:)
    real(rp) :: b(size(a))
    integer :: n
    n = size(a); b(:) = a(:)
    call qsort(b,1,n)
    median = 0.5_rp*(b((n+1)/2)+b(n/2+1))
  end function median
  !
  recursive subroutine qsort(b,lo,hi)
    real(rp), intent(inout) :: b(:)
    integer , intent(in)    :: lo,hi
    real(rp) :: p,x
    integer :: i,j
    if(lo >= hi) return
    p = b((lo+hi)/2); i = lo; j = hi
    do while(i <= j)
      do while(b(i) < p)
        i = i+1
      end do
      do while(b(j) > p)
        j = j-1
      end do
      if(i <= j) then
        x = b(i); b(i) = b(j); b(j) = x
        i = i+1; j = j-1
      end if
    end do
    call qsort(b,lo,j)
    call qsort(b,i,hi)
  end subroutine qsort
end module mod_timer
