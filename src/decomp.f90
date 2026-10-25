!
! 2D pencil decomposition and global transposes
!
! Two grids are used (all spectral data complex):
!   grid A, global (nxh, mz, ny): used between the x-FFT and the z-FFT
!     xa(nxh, nzA, nyA)   x-pencil (axis 1 local), natural memory order
!     ya(mz , nyA, nkx)   y-pencil (axis 2 local), memory order (2,3,1)
!   grid B, global (nxh, nz, ny): used between the z-FFT and the wall-normal solves
!     yb(nz , nyA, nkx)   y-pencil, memory order (2,3,1)
!     zb(nkx, nkz, ny )   z-pencil (axis 3 = wall-normal direction local), natural order
! Axis 1 = streamwise (kx), axis 2 = spanwise (z/kz), axis 3 = wall-normal (y).
! Process grid p1 x p2: in the x-pencil axis 2 is split over p1 and axis 3 over p2,
! in the y-pencil axis 1 over p1 and axis 3 over p2, in the z-pencil axis 1 over p1
! and axis 2 over p2 (same convention as 2DECOMP&FFT and cuDecomp).
!
! Transpose libraries, chosen at run time with transpose_lib in &parallel:
!   'cudecomp'   NVIDIA cuDecomp (as in CaNS; autotuning of the process grid and of the backend:
!                MPI, NCCL, NVSHMEM); needs -D_USE_CUDECOMP (make CUDECOMP=1)
!   'diezdecomp' diezDecomp (R. Diez Sanhueza, J. Peeters, P. Costa, Comput. Phys. Commun. 316
!                (2025) 109811; github.com/Rafael10Diez/diezDecomp): OpenACC packing and
!                autotuned MPI_Isend/Irecv, local copies without MPI; needs -D_USE_DIEZDECOMP
!                (make DIEZDECOMP=1)
!   'mpi'        portable MPI_Alltoallv (CUDA-aware MPI on GPUs); always available
! 'mpi' and 'diezdecomp' use the block distribution of init_mpi_layout; cuDecomp its own.
!
! transpose_prec = 'single': the data are rounded to single precision for the transposes only
! (half the communication volume); all arrays and all computations stay in double precision.
!
! Batched and overlapped transposes ('mpi' only; tr_batch, tr_overlap in &parallel, used by
! bwd_fft_n/fwd_fft_n in mod_fft): the fields of a group are sent in one MPI_Ialltoallv, the block of
! field f to peer p at nb*displ(p) + (f-1)*count(p) (nb fields in the group). With tr_overlap two
! buffer slots per transpose stage let the communication of one group run while the previous group
! is transformed.
!
module mod_decomp
  use mpi
  use mod_types
  use mod_param, only: nxh,mx,mz,nz,ny,pdims,abort_run,transpose_lib,transpose_prec,tr_batch,tr_overlap, &
                       cudecomp_t_backend,cudecomp_enable_nccl,cudecomp_enable_nvshmem
#if defined(_OPENACC)
  use openacc
#endif
#if defined(_USE_CUDECOMP)
  use cudecomp
#endif
#if defined(_USE_DIEZDECOMP)
  use diezdecomp_api_generic, only: diezdecomp_props_transp,diezdecomp_parsed_mpi_ranks, &
                                    diezdecomp_track_mpi_decomp,diezdecomp_generic_fill_tr_obj, &
                                    diezdecomp_transp_execute_generic_nobuf
#endif
  implicit none
  private
  public :: init_decomp,finalize_decomp,tr_xa2ya,tr_ya2xa,tr_yb2zb,tr_zb2yb
  public :: trb_xa,trb_ya,trb_yb,trb_zb,trb_post,trb_wait,trb_test
  integer, public :: myid,nproc,p1,p2,cx1,cx2
  integer, public :: zA_lo,nzA,yA_lo,nyA,kx_lo,nkx,kz_lo,nkz
  integer(i8), public :: tr_work_bytes = 0      ! memory of the transpose buffers / workspace (this rank)
  integer, parameter :: TL_MPI = 1, TL_CUDECOMP = 2, TL_DIEZ = 3
  integer :: tlib = TL_MPI
  logical, public :: tr_single = .false.        ! transposes in single precision
  logical, public :: trb_mpi = .false.          ! batched/overlapped MPI transposes set up
  integer, public :: trb_nslot = 1              ! buffer slots per stage (2 with tr_overlap)
#if defined(_OPENACC)
  integer(acc_handle_kind), public :: istream
#endif
#if defined(_USE_CUDECOMP)
  type(cudecompHandle)   :: ch
  type(cudecompGridDesc) :: gd_a,gd_b
  complex(rp), pointer, contiguous :: work(:)
  complex(rp), pointer, contiguous, device :: work_d(:)
  complex(sp), pointer, contiguous :: work4(:)              ! single-precision transposes:
  complex(sp), pointer, contiguous, device :: work4_d(:)    !   workspace and staging arrays
  complex(sp), allocatable :: cw4a(:),cw4b(:)
#endif
  integer :: comm_row,comm_col
  integer, allocatable :: kxlo_r(:),nkx_r(:),zlo_r(:),nz_r(:)   ! per peer in the row communicator
  integer, allocatable :: kzlo_c(:),nkz_c(:),ylo_c(:),ny_c(:)   ! per peer in the column communicator
  integer, allocatable :: own_kx(:),own_z(:),own_kz(:),own_y(:)
  integer, allocatable :: sc_xy(:),sd_xy(:),rc_xy(:),rd_xy(:)
  integer, allocatable :: sc_yz(:),sd_yz(:),rc_yz(:),rd_yz(:)
  complex(rp), allocatable :: sbuf(:),rbuf(:)
  complex(sp), allocatable :: sbuf4(:),rbuf4(:)
  ! batched transposes: send/receive buffers per slot, for grid A (x <-> y, comm_row) and grid B
  ! (y <-> z, comm_col), with the counts and displacements of the message in flight in each slot
  complex(rp), allocatable :: tbA_s(:,:),tbA_r(:,:),tbB_s(:,:),tbB_r(:,:)
  complex(sp), allocatable :: tbA_s4(:,:),tbA_r4(:,:),tbB_s4(:,:),tbB_r4(:,:)
  integer, allocatable :: csA(:,:),dsA(:,:),crA(:,:),drA(:,:),csB(:,:),dsB(:,:),crB(:,:),drB(:,:)
  integer :: reqA(2),reqB(2)
#if defined(_USE_DIEZDECOMP)
  ! diezDecomp works on real arrays: real and imaginary parts are stacked along the axis that a
  ! transpose does not redistribute (y for grid A, kx for grid B), so each complex transpose is one
  ! real transpose of the same data volume
  type(diezdecomp_props_transp), save :: dz_tr(4)        ! xa->ya, ya->xa, yb->zb, zb->yb
  ! (diezDecomp runs on its default OpenACC queue 1, the queue of all HiCHAN kernels)
  real(rp), allocatable, target :: dz_in(:),dz_out(:)
  ! single precision: one complex(sp) is moved as one 8-byte word (diezDecomp only copies data);
  ! dz_in8/dz_out8 are real(8) views of dz_in4/dz_out4 (same memory)
  complex(sp), allocatable, target :: dz_in4(:),dz_out4(:)
  real(rp), pointer, contiguous :: dz_in8(:),dz_out8(:)
#endif
contains
  subroutine init_decomp
    implicit none
    integer :: ierr
    call MPI_COMM_RANK(MPI_COMM_WORLD,myid ,ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD,nproc,ierr)
    if(myid == 0) write(*,'(a,i0)') ' number of MPI processes: ',nproc
    if(myid == 0 .and. nproc == 1 .and. len_trim(slurm_ntasks()) > 0) then
      if(trim(slurm_ntasks()) /= '1') write(*,'(a)') ' WARNING: SLURM_NTASKS = '//trim(slurm_ntasks())// &
        ' but MPI reports 1 process: the tasks were started as independent MPI singletons '// &
        '(launcher/MPI mismatch, try srun --mpi=pmix or the mpirun of the MPI used to compile HiCHAN)'
    end if
#if defined(_OPENACC)
    call init_device
#endif
    select case(trim(transpose_lib))
    case('auto')
#if defined(_USE_CUDECOMP)
      tlib = TL_CUDECOMP
#elif defined(_USE_DIEZDECOMP)
      tlib = TL_DIEZ
#else
      tlib = TL_MPI
#endif
    case('cudecomp')
#if !defined(_USE_CUDECOMP)
      call abort_run("transpose_lib = 'cudecomp', but HiCHAN was built without cuDecomp (make CUDECOMP=1)")
#endif
      tlib = TL_CUDECOMP
    case('diezdecomp')
#if !defined(_USE_DIEZDECOMP)
      call abort_run("transpose_lib = 'diezdecomp', but HiCHAN was built without diezDecomp (make DIEZDECOMP=1)")
#endif
      tlib = TL_DIEZ
    case('mpi')
      tlib = TL_MPI
    case default
      call abort_run("transpose_lib must be 'auto', 'cudecomp', 'diezdecomp' or 'mpi'")
    end select
    tr_single = (transpose_prec == 'single')
#if defined(_OPENACC)
    istream = acc_get_cuda_stream(1)
#endif
    if(tlib == TL_CUDECOMP) then
#if defined(_USE_CUDECOMP)
      call init_cudecomp
#endif
    else
      call init_mpi_layout
      if(tlib == TL_MPI) call init_mpi_transposes
      if(tlib == TL_MPI .and. (tr_batch > 1 .or. tr_overlap)) call init_batched_transposes
#if defined(_USE_DIEZDECOMP)
      if(tlib == TL_DIEZ) call init_diezdecomp
#endif
    end if
    if(myid == 0) then
      write(*,'(a,a)') ' transposes: ',trim(merge('cuDecomp  ',merge('diezDecomp','MPI       ',tlib == TL_DIEZ), &
                                                 tlib == TL_CUDECOMP))//', '//trim(transpose_prec)//' precision'
      write(*,'(a,i0,a,i0)') ' process grid p1 x p2 = ',p1,' x ',p2
      if(trb_mpi) then
        write(*,'(a,i0,a,a)') ' batched transforms: up to ',tr_batch,' fields per transpose, overlap ', &
                               trim(merge('on ','off',tr_overlap))
      else if(tr_batch > 1 .or. tr_overlap) then
        write(*,'(a)') ' NOTE: tr_batch/tr_overlap are implemented for transpose_lib = ''mpi'' only; '// &
                       'the transforms are done one field at a time'
      end if
    end if
  end subroutine init_decomp
  !
#if defined(_OPENACC)
  subroutine init_device
    !
    ! one GPU per rank: node-local rank -> device; the assignment is reported and checked,
    ! since two ranks on the same GPU make NCCL fail (ncclInvalidUsage, "Duplicate GPU")
    !
    use cudafor, only: cudaGetDeviceCount,cudaSetDevice,cudaGetDeviceProperties,cudaDeviceProp
    implicit none
    integer :: local_comm,lrank,mydev,ndev,istat,ierr,hlen,r,q
    integer(acc_device_kind) :: dev_type
    type(cudaDeviceProp) :: prop
    character(len=MPI_MAX_PROCESSOR_NAME) :: host
    character(len=128) :: cvd
    character(len=256) :: line
    character(len=256), allocatable :: lines(:)
    character(len=MPI_MAX_PROCESSOR_NAME+16) :: key
    character(len=MPI_MAX_PROCESSOR_NAME+16), allocatable :: keys(:)
    logical :: dup
    call MPI_COMM_SPLIT_TYPE(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,local_comm,ierr)
    call MPI_COMM_RANK(local_comm,lrank,ierr)
    istat = cudaGetDeviceCount(ndev)
    if(istat /= 0 .or. ndev < 1) call abort_run('no GPU visible to this MPI rank')
    mydev = mod(lrank,ndev)
    istat = cudaSetDevice(mydev)
    dev_type = acc_get_device_type()
    call acc_set_device_num(mydev,dev_type)
    call MPI_COMM_FREE(local_comm,ierr)
    !
    ! report and check
    !
    istat = cudaGetDeviceProperties(prop,mydev)
    call MPI_GET_PROCESSOR_NAME(host,hlen,ierr)
    call get_environment_variable('CUDA_VISIBLE_DEVICES',cvd,status=istat)
    if(istat /= 0) cvd = '(unset)'
    write(line,'(a,i0,a,a,a,i0,a,i0,a,i0,a,z4.4,a,z2.2,a,z2.2,a,a)') ' rank ',myid,' on ',trim(host), &
      ': local rank ',lrank,', sees ',ndev,' GPU(s), uses device ',mydev,' (PCI ',prop%pciDomainID,':', &
      prop%pciBusID,':',prop%pciDeviceID,'), CUDA_VISIBLE_DEVICES=',trim(cvd)
    write(key,'(a,1x,z4.4,z2.2,z2.2)') trim(host),prop%pciDomainID,prop%pciBusID,prop%pciDeviceID
    allocate(lines(0:nproc-1),keys(0:nproc-1))
    call MPI_GATHER(line,len(line),MPI_CHARACTER,lines,len(line),MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
    call MPI_GATHER(key ,len(key ),MPI_CHARACTER,keys ,len(key ),MPI_CHARACTER,0,MPI_COMM_WORLD,ierr)
    dup = .false.
    if(myid == 0) then
      write(*,'(a)') ' GPU assignment:'
      do r=0,nproc-1
        write(*,'(a)') trim(lines(r))
      end do
      do r=0,nproc-1
        do q=r+1,nproc-1
          if(keys(r) == keys(q)) dup = .true.
        end do
      end do
      flush(6)
    end if
    call MPI_BCAST(dup,1,MPI_LOGICAL,0,MPI_COMM_WORLD,ierr)
    if(dup) call abort_run('several MPI ranks use the same GPU (see the GPU assignment above). '// &
      'Each rank must see a different GPU: request one task per GPU and do not restrict all tasks to the '// &
      'same device (e.g. srun --gpus-per-task=1 --gpu-bind=single:1, or no GPU binding with --gpus-per-node=4)')
  end subroutine init_device
#endif
  !
  function slurm_ntasks() result(val)
    implicit none
    character(len=32) :: val
    integer :: stat
    call get_environment_variable('SLURM_NTASKS',val,status=stat)
    if(stat /= 0) val = ''
  end function slurm_ntasks
  !
  pure subroutine distribute(n,p,c,lo,cnt)
    ! block distribution of n points over p processes (remainder to the last ones)
    implicit none
    integer, intent(in)  :: n,p,c
    integer, intent(out) :: lo,cnt
    integer :: base,nrem
    base = n/p
    nrem = n - base*p
    if(c < p-nrem) then
      cnt = base
      lo  = 1 + c*base
    else
      cnt = base+1
      lo  = 1 + (p-nrem)*base + (c-(p-nrem))*(base+1)
    end if
  end subroutine distribute
  !
  subroutine init_mpi_layout
    ! block distribution on the process grid p1 x p2 (used by the 'mpi' and 'diezdecomp' transposes)
    implicit none
    integer :: dims(2),ierr,p
    if(all(pdims(:) > 0)) then
      if(pdims(1)*pdims(2) /= nproc) call abort_run('pdims(1)*pdims(2) /= number of MPI processes')
      dims(:) = pdims(:)
    else
      dims(:) = 0
      call MPI_DIMS_CREATE(nproc,2,dims,ierr)
      ! prefer a larger p1 (x<->y transposes stay within a node for p1 <= GPUs per node)
      if(dims(2) > dims(1)) dims(:) = [dims(2),dims(1)]
    end if
    p1 = dims(1); p2 = dims(2)
    if(p1 > min(nxh,mz) .or. p2 > min(nz,ny)) call abort_run('process grid too large for the problem size')
    cx1 = mod(myid,p1)
    cx2 = myid/p1
    call MPI_COMM_SPLIT(MPI_COMM_WORLD,cx2,cx1,comm_row,ierr) ! varying cx1
    call MPI_COMM_SPLIT(MPI_COMM_WORLD,cx1,cx2,comm_col,ierr) ! varying cx2
    allocate(kxlo_r(0:p1-1),nkx_r(0:p1-1),zlo_r(0:p1-1),nz_r(0:p1-1))
    allocate(kzlo_c(0:p2-1),nkz_c(0:p2-1),ylo_c(0:p2-1),ny_c(0:p2-1))
    do p=0,p1-1
      call distribute(nxh,p1,p,kxlo_r(p),nkx_r(p))
      call distribute(mz ,p1,p,zlo_r(p) ,nz_r(p) )
    end do
    do p=0,p2-1
      call distribute(nz,p2,p,kzlo_c(p),nkz_c(p))
      call distribute(ny,p2,p,ylo_c(p) ,ny_c(p) )
    end do
    kx_lo = kxlo_r(cx1); nkx = nkx_r(cx1)
    zA_lo = zlo_r(cx1) ; nzA = nz_r(cx1)
    kz_lo = kzlo_c(cx2); nkz = nkz_c(cx2)
    yA_lo = ylo_c(cx2) ; nyA = ny_c(cx2)
    allocate(own_kx(nxh),own_z(mz),own_kz(nz),own_y(ny))
    do p=0,p1-1
      own_kx(kxlo_r(p):kxlo_r(p)+nkx_r(p)-1) = p
      own_z(zlo_r(p):zlo_r(p)+nz_r(p)-1) = p
    end do
    do p=0,p2-1
      own_kz(kzlo_c(p):kzlo_c(p)+nkz_c(p)-1) = p
      own_y(ylo_c(p):ylo_c(p)+ny_c(p)-1) = p
    end do
  end subroutine init_mpi_layout
  !
  subroutine init_mpi_transposes
    implicit none
    integer :: p,i
    allocate(sc_xy(0:p1-1),sd_xy(0:p1-1),rc_xy(0:p1-1),rd_xy(0:p1-1))
    allocate(sc_yz(0:p2-1),sd_yz(0:p2-1),rc_yz(0:p2-1),rd_yz(0:p2-1))
    do p=0,p1-1
      sc_xy(p) = nkx_r(p)*nzA*nyA
      rc_xy(p) = nkx*nz_r(p)*nyA
    end do
    do p=0,p2-1
      sc_yz(p) = nkz_c(p)*nyA*nkx
      rc_yz(p) = nkz*ny_c(p)*nkx
    end do
    sd_xy(0) = 0; rd_xy(0) = 0; sd_yz(0) = 0; rd_yz(0) = 0
    do p=1,p1-1
      sd_xy(p) = sd_xy(p-1)+sc_xy(p-1)
      rd_xy(p) = rd_xy(p-1)+rc_xy(p-1)
    end do
    do p=1,p2-1
      sd_yz(p) = sd_yz(p-1)+sc_yz(p-1)
      rd_yz(p) = rd_yz(p-1)+rc_yz(p-1)
    end do
    i = max(nxh*nzA*nyA,mz*nyA*nkx,nz*nyA*nkx,nkx*nkz*ny)
    if(tr_single) then
      allocate(sbuf4(i),rbuf4(i),sbuf(1),rbuf(1))
      tr_work_bytes = 16_i8*i
    else
      allocate(sbuf(i),rbuf(i),sbuf4(1),rbuf4(1))
      tr_work_bytes = 32_i8*i
    end if
    !$acc enter data copyin(kxlo_r,nkx_r,zlo_r,nz_r,kzlo_c,nkz_c,ylo_c,ny_c, &
    !$acc                   own_kx,own_z,own_kz,own_y,sd_xy,rd_xy,sd_yz,rd_yz) create(sbuf,rbuf,sbuf4,rbuf4)
  end subroutine init_mpi_transposes
  !
  subroutine init_batched_transposes
    implicit none
    integer(i8) :: na,nb
    trb_mpi = .true.
    trb_nslot = merge(2,1,tr_overlap)
    na = int(tr_batch,i8)*max(nxh*nzA*nyA,mz*nyA*nkx)
    nb = int(tr_batch,i8)*max(nz*nyA*nkx,nkx*nkz*ny)
    if(tr_single) then
      allocate(tbA_s4(na,trb_nslot),tbA_r4(na,trb_nslot),tbB_s4(nb,trb_nslot),tbB_r4(nb,trb_nslot))
      allocate(tbA_s(1,trb_nslot),tbA_r(1,trb_nslot),tbB_s(1,trb_nslot),tbB_r(1,trb_nslot))
      tr_work_bytes = tr_work_bytes + 16_i8*trb_nslot*(na+nb)
    else
      allocate(tbA_s(na,trb_nslot),tbA_r(na,trb_nslot),tbB_s(nb,trb_nslot),tbB_r(nb,trb_nslot))
      allocate(tbA_s4(1,trb_nslot),tbA_r4(1,trb_nslot),tbB_s4(1,trb_nslot),tbB_r4(1,trb_nslot))
      tr_work_bytes = tr_work_bytes + 32_i8*trb_nslot*(na+nb)
    end if
    allocate(csA(0:p1-1,trb_nslot),dsA(0:p1-1,trb_nslot),crA(0:p1-1,trb_nslot),drA(0:p1-1,trb_nslot))
    allocate(csB(0:p2-1,trb_nslot),dsB(0:p2-1,trb_nslot),crB(0:p2-1,trb_nslot),drB(0:p2-1,trb_nslot))
    reqA(:) = MPI_REQUEST_NULL; reqB(:) = MPI_REQUEST_NULL
    !$acc enter data create(tbA_s,tbA_r,tbB_s,tbB_r,tbA_s4,tbA_r4,tbB_s4,tbB_r4) &
    !$acc            copyin(sc_xy,rc_xy,sc_yz,rc_yz)
  end subroutine init_batched_transposes
  !
#if defined(_USE_DIEZDECOMP)
  subroutine init_diezdecomp
    use, intrinsic :: iso_c_binding, only: c_f_pointer,c_loc
    !
    ! global axes 0 = kx, 1 = z (kz), 2 = y; memory orders: x-pencil (0,1,2), y-pencil (1,2,0),
    ! z-pencil (0,1,2). Rank coordinates (x,y,z order) in the three pencils: (0,cx1,cx2),
    ! (cx1,0,cx2), (cx1,cx2,0). The real arrays have the stacked axis doubled.
    !
    implicit none
    type(diezdecomp_parsed_mpi_ranks) :: rkx,rky,rkz
    integer :: ox(0:2),oy(0:2),oz(0:2),z6(0:2,0:1),sxa(0:2),sya(0:2),syb(0:2),szb(0:2)
    integer(i8) :: ws
    integer :: n
    ox = [0,1,2]; oy = [1,2,0]; oz = [0,1,2]; z6 = 0
    call diezdecomp_track_mpi_decomp([0,cx1,cx2],rkx,myid,nproc)
    call diezdecomp_track_mpi_decomp([cx1,0,cx2],rky,myid,nproc)
    call diezdecomp_track_mpi_decomp([cx1,cx2,0],rkz,myid,nproc)
    if(tr_single) then
      sxa = [nxh,nzA,nyA]      ! one 8-byte word per complex(sp) value
      sya = [mz,nyA,nkx]
      syb = [nz,nyA,nkx]
      szb = [nkx,nkz,ny]
    else
      sxa = [nxh,nzA,2*nyA]    ! xa: re/im stacked along y
      sya = [mz,2*nyA,nkx]     ! ya
      syb = [nz,nyA,2*nkx]     ! yb: re/im stacked along kx
      szb = [2*nkx,nkz,ny]     ! zb
    end if
    call diezdecomp_generic_fill_tr_obj(dz_tr(1),rkx,rky,sxa,z6,sya,z6,ox,oy,oy,.false.,-1,-1,ws,.false.)
    call diezdecomp_generic_fill_tr_obj(dz_tr(2),rky,rkx,sya,z6,sxa,z6,oy,ox,ox,.false.,-1,-1,ws,.false.)
    call diezdecomp_generic_fill_tr_obj(dz_tr(3),rky,rkz,syb,z6,szb,z6,oy,oz,oz,.false.,-1,-1,ws,.false.)
    call diezdecomp_generic_fill_tr_obj(dz_tr(4),rkz,rky,szb,z6,syb,z6,oz,oy,oy,.false.,-1,-1,ws,.false.)
    ! the 'nobuf' execution uses p_out to pack the input and p_in to receive: both need max size
    n = max(product(sxa),product(sya),product(syb),product(szb))
    if(tr_single) then
      allocate(dz_in4(0:n-1),dz_out4(0:n-1),dz_in(0:0),dz_out(0:0))
      dz_in4(:) = 0; dz_out4(:) = 0
      call c_f_pointer(c_loc(dz_in4) ,dz_in8 ,[n])
      call c_f_pointer(c_loc(dz_out4),dz_out8,[n])
    else
      allocate(dz_in(0:n-1),dz_out(0:n-1),dz_in4(0:0),dz_out4(0:0))
    end if
    tr_work_bytes = 16_i8*n
    dz_in(:) = 0._rp; dz_out(:) = 0._rp
    !$acc enter data copyin(dz_in,dz_out,dz_in4,dz_out4)
  end subroutine init_diezdecomp
#endif
  !
#if defined(_USE_CUDECOMP)
  subroutine init_cudecomp
    implicit none
    type(cudecompGridDescConfig) :: conf
    type(cudecompGridDescAutotuneOptions) :: atune
    type(cudecompPencilInfo) :: pxa,pya,pyb,pzb
    integer :: istat
    integer(i8) :: wsa,wsb,wsize,n
    istat = cudecompInit(ch,MPI_COMM_WORLD)
    if(istat /= CUDECOMP_RESULT_SUCCESS) call abort_run('cudecompInit failed (NCCL/NVSHMEM initialisation). '// &
      'Check that all ranks belong to one MPI job (e.g. srun --mpi=pmix) and that each rank sees its own GPU')
    !
    ! grid B (z-FFT <-> wall-normal solves): autotune process grid and backend
    !
    istat = cudecompGridDescConfigSetDefaults(conf)
    conf%gdims(:) = [nxh,nz,ny]
    conf%pdims(:) = pdims(:)
    conf%transpose_axis_contiguous(:) = [.true.,.true.,.false.]
    conf%transpose_comm_backend = cudecomp_t_backend
    istat = cudecompGridDescAutotuneOptionsSetDefaults(atune)
    atune%dtype = CUDECOMP_DOUBLE_COMPLEX
    if(tr_single) atune%dtype = CUDECOMP_FLOAT_COMPLEX
    atune%grid_mode = CUDECOMP_AUTOTUNE_GRID_TRANSPOSE
    atune%autotune_transpose_backend = (cudecomp_t_backend == 0)
    atune%disable_nccl_backends    = .not.cudecomp_enable_nccl
    atune%disable_nvshmem_backends = .not.cudecomp_enable_nvshmem
    atune%transpose_op_weights(:) = [1.,1.,1.,1.]
    if(cudecomp_t_backend == 0) conf%transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_MPI_A2A
    istat = cudecompGridDescCreate(ch,gd_b,conf,atune)
    if(istat /= CUDECOMP_RESULT_SUCCESS) call abort_run('cudecompGridDescCreate failed (grid B)')
    p1 = conf%pdims(1); p2 = conf%pdims(2)
    if(myid == 0) write(*,'(a,a)') ' cuDecomp transpose backend (grid B): ', &
                  cudecompTransposeCommBackendToString(conf%transpose_comm_backend)
    !
    ! grid A (x-FFT <-> z-FFT): same process grid, autotune the backend only
    !
    istat = cudecompGridDescConfigSetDefaults(conf)
    conf%gdims(:) = [nxh,mz,ny]
    conf%pdims(:) = [p1,p2]
    conf%transpose_axis_contiguous(:) = [.true.,.true.,.false.]
    conf%transpose_comm_backend = cudecomp_t_backend
    if(cudecomp_t_backend == 0) conf%transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_MPI_A2A
    atune%grid_mode = CUDECOMP_AUTOTUNE_GRID_TRANSPOSE
    istat = cudecompGridDescCreate(ch,gd_a,conf,atune)
    if(istat /= CUDECOMP_RESULT_SUCCESS) call abort_run('cudecompGridDescCreate failed (grid A)')
    if(myid == 0) write(*,'(a,a)') ' cuDecomp transpose backend (grid A): ', &
                  cudecompTransposeCommBackendToString(conf%transpose_comm_backend)
    !
    istat = cudecompGetPencilInfo(ch,gd_a,pxa,1)
    istat = cudecompGetPencilInfo(ch,gd_a,pya,2)
    istat = cudecompGetPencilInfo(ch,gd_b,pyb,2)
    istat = cudecompGetPencilInfo(ch,gd_b,pzb,3)
    ! x-pencil, order (1,2,3)
    zA_lo = pxa%lo(2); nzA = pxa%shape(2)
    yA_lo = pxa%lo(3); nyA = pxa%shape(3)
    ! y-pencil, order (2,3,1)
    if(any(pya%order(:) /= [2,3,1]) .or. any(pyb%order(:) /= [2,3,1])) &
      call abort_run('unexpected cuDecomp y-pencil memory order')
    kx_lo = pya%lo(3); nkx = pya%shape(3)
    if(pya%shape(2) /= nyA .or. pya%lo(2) /= yA_lo .or. pyb%shape(2) /= nyA .or. pyb%shape(3) /= nkx &
       .or. pyb%lo(3) /= kx_lo) call abort_run('inconsistent cuDecomp pencils (grids A and B)')
    ! z-pencil, order (1,2,3)
    if(any(pzb%order(:) /= [1,2,3])) call abort_run('unexpected cuDecomp z-pencil memory order')
    if(pzb%lo(1) /= kx_lo .or. pzb%shape(1) /= nkx) call abort_run('inconsistent cuDecomp z-pencil')
    kz_lo = pzb%lo(2); nkz = pzb%shape(2)
    cx1 = -1; cx2 = -1
    !
    istat = cudecompGetTransposeWorkspaceSize(ch,gd_a,wsa)
    istat = cudecompGetTransposeWorkspaceSize(ch,gd_b,wsb)
    wsize = max(wsa,wsb)
    if(tr_single) then
      ! single-precision workspace, plus staging arrays for the rounded pencils
      allocate(work4(wsize))
      istat = cudecompMalloc(ch,gd_a,work4_d,wsize)
      call acc_map_data(work4,work4_d,wsize*8_i8)
      n = max(int(nxh,i8)*nzA*nyA,int(mz,i8)*nyA*nkx,int(nz,i8)*nyA*nkx,int(nkx,i8)*nkz*ny)
      allocate(cw4a(n),cw4b(n))
      !$acc enter data create(cw4a,cw4b)
      tr_work_bytes = 8_i8*wsize + 16_i8*n
    else
      allocate(work(wsize))
      istat = cudecompMalloc(ch,gd_a,work_d,wsize)
      call acc_map_data(work,work_d,wsize*16_i8)
      tr_work_bytes = 16_i8*wsize
    end if
  end subroutine init_cudecomp
#endif
  !
  subroutine finalize_decomp
    implicit none
#if defined(_USE_CUDECOMP)
    integer :: istat
    if(tlib == TL_CUDECOMP) then
      if(tr_single) then
        call acc_unmap_data(work4)
        istat = cudecompFree(ch,gd_a,work4_d)
      else
        call acc_unmap_data(work)
        istat = cudecompFree(ch,gd_a,work_d)
      end if
      istat = cudecompGridDescDestroy(ch,gd_a)
      istat = cudecompGridDescDestroy(ch,gd_b)
      istat = cudecompFinalize(ch)
    end if
#endif
  end subroutine finalize_decomp
  !
  ! ---------------------------------------------------------------------------
  ! transposes
  ! ---------------------------------------------------------------------------
  !
  subroutine tr_xa2ya(xa,ya)
    implicit none
    complex(rp), intent(in ) :: xa(nxh,nzA,nyA)
    complex(rp), intent(out) :: ya(mz,nyA,nkx)
    integer :: i,j,k,p
#if defined(_USE_CUDECOMP)
    integer :: istat
#endif
    select case(tlib)
#if defined(_USE_CUDECOMP)
    case(TL_CUDECOMP)
      if(tr_single) then
        call to_sp(size(xa),xa,cw4a)
        !$acc host_data use_device(cw4a,cw4b,work4)
        istat = cudecompTransposeXToY(ch,gd_a,cw4a,cw4b,work4,CUDECOMP_FLOAT_COMPLEX,stream=istream)
        !$acc end host_data
        call to_dp(size(ya),cw4b,ya)
      else
        !$acc host_data use_device(xa,ya,work)
        istat = cudecompTransposeXToY(ch,gd_a,xa,ya,work,CUDECOMP_DOUBLE_COMPLEX,stream=istream)
        !$acc end host_data
      end if
#endif
#if defined(_USE_DIEZDECOMP)
    case(TL_DIEZ)
      if(tr_single) then
        !$acc parallel loop collapse(3) default(present) async(1)
        do k=1,nyA
          do j=1,nzA
            do i=1,nxh
              dz_in4(i-1+nxh*(j-1+nzA*(k-1))) = cmplx(xa(i,j,k),kind=sp)
            end do
          end do
        end do
        call diezdecomp_transp_execute_generic_nobuf(dz_tr(1),dz_in8,dz_out8)
        !$acc parallel loop collapse(3) default(present) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,mz
              ya(j,k,i) = cmplx(dz_out4(j-1+mz*(k-1+nyA*(i-1))),kind=rp)
            end do
          end do
        end do
      else
        !$acc parallel loop collapse(3) default(present) async(1)
        do k=1,nyA
          do j=1,nzA
            do i=1,nxh
              dz_in(i-1+nxh*(j-1+nzA*(k-1)))     = real(xa(i,j,k),rp)
              dz_in(i-1+nxh*(j-1+nzA*(k-1+nyA))) = aimag(xa(i,j,k))
            end do
          end do
        end do
        call diezdecomp_transp_execute_generic_nobuf(dz_tr(1),dz_in,dz_out)
        !$acc parallel loop collapse(3) default(present) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,mz
              ya(j,k,i) = cmplx(dz_out(j-1+mz*(k-1+2*nyA*(i-1))),dz_out(j-1+mz*(k-1+nyA+2*nyA*(i-1))),rp)
            end do
          end do
        end do
      end if
#endif
    case default
      if(tr_single) then
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do k=1,nyA
          do j=1,nzA
            do i=1,nxh
              p = own_kx(i)
              sbuf4(sd_xy(p)+1+(i-kxlo_r(p))+nkx_r(p)*((j-1)+nzA*(k-1))) = cmplx(xa(i,j,k),kind=sp)
            end do
          end do
        end do
        call alltoallv4(sbuf4,sc_xy,sd_xy,rbuf4,rc_xy,rd_xy,comm_row)
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,mz
              p = own_z(j)
              ya(j,k,i) = cmplx(rbuf4(rd_xy(p)+1+(i-1)+nkx*((j-zlo_r(p))+nz_r(p)*(k-1))),kind=rp)
            end do
          end do
        end do
      else
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do k=1,nyA
          do j=1,nzA
            do i=1,nxh
              p = own_kx(i)
              sbuf(sd_xy(p)+1+(i-kxlo_r(p))+nkx_r(p)*((j-1)+nzA*(k-1))) = xa(i,j,k)
            end do
          end do
        end do
        call alltoallv(sbuf,sc_xy,sd_xy,rbuf,rc_xy,rd_xy,comm_row)
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,mz
              p = own_z(j)
              ya(j,k,i) = rbuf(rd_xy(p)+1+(i-1)+nkx*((j-zlo_r(p))+nz_r(p)*(k-1)))
            end do
          end do
        end do
      end if
    end select
  end subroutine tr_xa2ya
  !
  subroutine tr_ya2xa(ya,xa)
    implicit none
    complex(rp), intent(in ) :: ya(mz,nyA,nkx)
    complex(rp), intent(out) :: xa(nxh,nzA,nyA)
    integer :: i,j,k,p
#if defined(_USE_CUDECOMP)
    integer :: istat
#endif
    select case(tlib)
#if defined(_USE_CUDECOMP)
    case(TL_CUDECOMP)
      if(tr_single) then
        call to_sp(size(ya),ya,cw4a)
        !$acc host_data use_device(cw4a,cw4b,work4)
        istat = cudecompTransposeYToX(ch,gd_a,cw4a,cw4b,work4,CUDECOMP_FLOAT_COMPLEX,stream=istream)
        !$acc end host_data
        call to_dp(size(xa),cw4b,xa)
      else
        !$acc host_data use_device(xa,ya,work)
        istat = cudecompTransposeYToX(ch,gd_a,ya,xa,work,CUDECOMP_DOUBLE_COMPLEX,stream=istream)
        !$acc end host_data
      end if
#endif
#if defined(_USE_DIEZDECOMP)
    case(TL_DIEZ)
      if(tr_single) then
        !$acc parallel loop collapse(3) default(present) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,mz
              dz_in4(j-1+mz*(k-1+nyA*(i-1))) = cmplx(ya(j,k,i),kind=sp)
            end do
          end do
        end do
        call diezdecomp_transp_execute_generic_nobuf(dz_tr(2),dz_in8,dz_out8)
        !$acc parallel loop collapse(3) default(present) async(1)
        do k=1,nyA
          do j=1,nzA
            do i=1,nxh
              xa(i,j,k) = cmplx(dz_out4(i-1+nxh*(j-1+nzA*(k-1))),kind=rp)
            end do
          end do
        end do
      else
        !$acc parallel loop collapse(3) default(present) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,mz
              dz_in(j-1+mz*(k-1+2*nyA*(i-1)))     = real(ya(j,k,i),rp)
              dz_in(j-1+mz*(k-1+nyA+2*nyA*(i-1))) = aimag(ya(j,k,i))
            end do
          end do
        end do
        call diezdecomp_transp_execute_generic_nobuf(dz_tr(2),dz_in,dz_out)
        !$acc parallel loop collapse(3) default(present) async(1)
        do k=1,nyA
          do j=1,nzA
            do i=1,nxh
              xa(i,j,k) = cmplx(dz_out(i-1+nxh*(j-1+nzA*(k-1))),dz_out(i-1+nxh*(j-1+nzA*(k-1+nyA))),rp)
            end do
          end do
        end do
      end if
#endif
    case default
      if(tr_single) then
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,mz
              p = own_z(j)
              rbuf4(rd_xy(p)+1+(i-1)+nkx*((j-zlo_r(p))+nz_r(p)*(k-1))) = cmplx(ya(j,k,i),kind=sp)
            end do
          end do
        end do
        call alltoallv4(rbuf4,rc_xy,rd_xy,sbuf4,sc_xy,sd_xy,comm_row)
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do k=1,nyA
          do j=1,nzA
            do i=1,nxh
              p = own_kx(i)
              xa(i,j,k) = cmplx(sbuf4(sd_xy(p)+1+(i-kxlo_r(p))+nkx_r(p)*((j-1)+nzA*(k-1))),kind=rp)
            end do
          end do
        end do
      else
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,mz
              p = own_z(j)
              rbuf(rd_xy(p)+1+(i-1)+nkx*((j-zlo_r(p))+nz_r(p)*(k-1))) = ya(j,k,i)
            end do
          end do
        end do
        call alltoallv(rbuf,rc_xy,rd_xy,sbuf,sc_xy,sd_xy,comm_row)
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do k=1,nyA
          do j=1,nzA
            do i=1,nxh
              p = own_kx(i)
              xa(i,j,k) = sbuf(sd_xy(p)+1+(i-kxlo_r(p))+nkx_r(p)*((j-1)+nzA*(k-1)))
            end do
          end do
        end do
      end if
    end select
  end subroutine tr_ya2xa
  !
  subroutine tr_yb2zb(yb,zb)
    implicit none
    complex(rp), intent(in ) :: yb(nz,nyA,nkx)
    complex(rp), intent(out) :: zb(nkx,nkz,ny)
    integer :: i,j,k,p
#if defined(_USE_CUDECOMP)
    integer :: istat
#endif
    select case(tlib)
#if defined(_USE_CUDECOMP)
    case(TL_CUDECOMP)
      if(tr_single) then
        call to_sp(size(yb),yb,cw4a)
        !$acc host_data use_device(cw4a,cw4b,work4)
        istat = cudecompTransposeYToZ(ch,gd_b,cw4a,cw4b,work4,CUDECOMP_FLOAT_COMPLEX,stream=istream)
        !$acc end host_data
        call to_dp(size(zb),cw4b,zb)
      else
        !$acc host_data use_device(yb,zb,work)
        istat = cudecompTransposeYToZ(ch,gd_b,yb,zb,work,CUDECOMP_DOUBLE_COMPLEX,stream=istream)
        !$acc end host_data
      end if
#endif
#if defined(_USE_DIEZDECOMP)
    case(TL_DIEZ)
      if(tr_single) then
        !$acc parallel loop collapse(3) default(present) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,nz
              dz_in4(j-1+nz*(k-1+nyA*(i-1))) = cmplx(yb(j,k,i),kind=sp)
            end do
          end do
        end do
        call diezdecomp_transp_execute_generic_nobuf(dz_tr(3),dz_in8,dz_out8)
        !$acc parallel loop collapse(3) default(present) async(1)
        do k=1,ny
          do j=1,nkz
            do i=1,nkx
              zb(i,j,k) = cmplx(dz_out4(i-1+nkx*(j-1+nkz*(k-1))),kind=rp)
            end do
          end do
        end do
      else
        !$acc parallel loop collapse(3) default(present) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,nz
              dz_in(j-1+nz*(k-1+nyA*(i-1)))     = real(yb(j,k,i),rp)
              dz_in(j-1+nz*(k-1+nyA*(i-1+nkx))) = aimag(yb(j,k,i))
            end do
          end do
        end do
        call diezdecomp_transp_execute_generic_nobuf(dz_tr(3),dz_in,dz_out)
        !$acc parallel loop collapse(3) default(present) async(1)
        do k=1,ny
          do j=1,nkz
            do i=1,nkx
              zb(i,j,k) = cmplx(dz_out(i-1+2*nkx*(j-1+nkz*(k-1))),dz_out(i-1+nkx+2*nkx*(j-1+nkz*(k-1))),rp)
            end do
          end do
        end do
      end if
#endif
    case default
      if(tr_single) then
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,nz
              p = own_kz(j)
              sbuf4(sd_yz(p)+1+(i-1)+nkx*((j-kzlo_c(p))+nkz_c(p)*(k-1))) = cmplx(yb(j,k,i),kind=sp)
            end do
          end do
        end do
        call alltoallv4(sbuf4,sc_yz,sd_yz,rbuf4,rc_yz,rd_yz,comm_col)
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do k=1,ny
          do j=1,nkz
            do i=1,nkx
              p = own_y(k)
              zb(i,j,k) = cmplx(rbuf4(rd_yz(p)+1+(i-1)+nkx*((j-1)+nkz*(k-ylo_c(p)))),kind=rp)
            end do
          end do
        end do
      else
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,nz
              p = own_kz(j)
              sbuf(sd_yz(p)+1+(i-1)+nkx*((j-kzlo_c(p))+nkz_c(p)*(k-1))) = yb(j,k,i)
            end do
          end do
        end do
        call alltoallv(sbuf,sc_yz,sd_yz,rbuf,rc_yz,rd_yz,comm_col)
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do k=1,ny
          do j=1,nkz
            do i=1,nkx
              p = own_y(k)
              zb(i,j,k) = rbuf(rd_yz(p)+1+(i-1)+nkx*((j-1)+nkz*(k-ylo_c(p))))
            end do
          end do
        end do
      end if
    end select
  end subroutine tr_yb2zb
  !
  subroutine tr_zb2yb(zb,yb)
    implicit none
    complex(rp), intent(in ) :: zb(nkx,nkz,ny)
    complex(rp), intent(out) :: yb(nz,nyA,nkx)
    integer :: i,j,k,p
#if defined(_USE_CUDECOMP)
    integer :: istat
#endif
    select case(tlib)
#if defined(_USE_CUDECOMP)
    case(TL_CUDECOMP)
      if(tr_single) then
        call to_sp(size(zb),zb,cw4a)
        !$acc host_data use_device(cw4a,cw4b,work4)
        istat = cudecompTransposeZToY(ch,gd_b,cw4a,cw4b,work4,CUDECOMP_FLOAT_COMPLEX,stream=istream)
        !$acc end host_data
        call to_dp(size(yb),cw4b,yb)
      else
        !$acc host_data use_device(yb,zb,work)
        istat = cudecompTransposeZToY(ch,gd_b,zb,yb,work,CUDECOMP_DOUBLE_COMPLEX,stream=istream)
        !$acc end host_data
      end if
#endif
#if defined(_USE_DIEZDECOMP)
    case(TL_DIEZ)
      if(tr_single) then
        !$acc parallel loop collapse(3) default(present) async(1)
        do k=1,ny
          do j=1,nkz
            do i=1,nkx
              dz_in4(i-1+nkx*(j-1+nkz*(k-1))) = cmplx(zb(i,j,k),kind=sp)
            end do
          end do
        end do
        call diezdecomp_transp_execute_generic_nobuf(dz_tr(4),dz_in8,dz_out8)
        !$acc parallel loop collapse(3) default(present) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,nz
              yb(j,k,i) = cmplx(dz_out4(j-1+nz*(k-1+nyA*(i-1))),kind=rp)
            end do
          end do
        end do
      else
        !$acc parallel loop collapse(3) default(present) async(1)
        do k=1,ny
          do j=1,nkz
            do i=1,nkx
              dz_in(i-1+2*nkx*(j-1+nkz*(k-1)))     = real(zb(i,j,k),rp)
              dz_in(i-1+nkx+2*nkx*(j-1+nkz*(k-1))) = aimag(zb(i,j,k))
            end do
          end do
        end do
        call diezdecomp_transp_execute_generic_nobuf(dz_tr(4),dz_in,dz_out)
        !$acc parallel loop collapse(3) default(present) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,nz
              yb(j,k,i) = cmplx(dz_out(j-1+nz*(k-1+nyA*(i-1))),dz_out(j-1+nz*(k-1+nyA*(i-1+nkx))),rp)
            end do
          end do
        end do
      end if
#endif
    case default
      if(tr_single) then
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do k=1,ny
          do j=1,nkz
            do i=1,nkx
              p = own_y(k)
              rbuf4(rd_yz(p)+1+(i-1)+nkx*((j-1)+nkz*(k-ylo_c(p)))) = cmplx(zb(i,j,k),kind=sp)
            end do
          end do
        end do
        call alltoallv4(rbuf4,rc_yz,rd_yz,sbuf4,sc_yz,sd_yz,comm_col)
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,nz
              p = own_kz(j)
              yb(j,k,i) = cmplx(sbuf4(sd_yz(p)+1+(i-1)+nkx*((j-kzlo_c(p))+nkz_c(p)*(k-1))),kind=rp)
            end do
          end do
        end do
      else
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do k=1,ny
          do j=1,nkz
            do i=1,nkx
              p = own_y(k)
              rbuf(rd_yz(p)+1+(i-1)+nkx*((j-1)+nkz*(k-ylo_c(p)))) = zb(i,j,k)
            end do
          end do
        end do
        call alltoallv(rbuf,rc_yz,rd_yz,sbuf,sc_yz,sd_yz,comm_col)
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,nz
              p = own_kz(j)
              yb(j,k,i) = sbuf(sd_yz(p)+1+(i-1)+nkx*((j-kzlo_c(p))+nkz_c(p)*(k-1)))
            end do
          end do
        end do
      end if
    end select
  end subroutine tr_zb2yb
  !
  subroutine alltoallv4(sb,sc,sd,rb,rc,rd,comm)
    implicit none
    complex(sp), intent(inout) :: sb(:),rb(:)
    integer, intent(in) :: sc(:),sd(:),rc(:),rd(:),comm
    integer :: ierr
    !$acc wait(1)
    !$acc host_data use_device(sb,rb)
    call MPI_ALLTOALLV(sb,sc,sd,MPI_COMPLEX,rb,rc,rd,MPI_COMPLEX,comm,ierr)
    !$acc end host_data
  end subroutine alltoallv4
  !
#if defined(_USE_CUDECOMP)
  subroutine to_sp(n,a,b)
    ! b = a rounded to single precision
    implicit none
    integer, intent(in) :: n
    complex(rp), intent(in) :: a(n)
    complex(sp), intent(out) :: b(n)
    integer :: i
    !$acc parallel loop default(present) async(1)
    do i=1,n
      b(i) = cmplx(a(i),kind=sp)
    end do
  end subroutine to_sp
  !
  subroutine to_dp(n,b,a)
    implicit none
    integer, intent(in) :: n
    complex(sp), intent(in) :: b(n)
    complex(rp), intent(out) :: a(n)
    integer :: i
    !$acc parallel loop default(present) async(1)
    do i=1,n
      a(i) = cmplx(b(i),kind=rp)
    end do
  end subroutine to_dp
#endif
  !
  subroutine alltoallv(sb,sc,sd,rb,rc,rd,comm)
    implicit none
    complex(rp), intent(inout) :: sb(:),rb(:)
    integer, intent(in) :: sc(:),sd(:),rc(:),rd(:),comm
    integer :: ierr
    !$acc wait(1)
    !$acc host_data use_device(sb,rb)
    call MPI_ALLTOALLV(sb,sc,sd,MPI_DOUBLE_COMPLEX,rb,rc,rd,MPI_DOUBLE_COMPLEX,comm,ierr)
    !$acc end host_data
  end subroutine alltoallv
  !
  ! ---------------------------------------------------------------------------
  ! batched transposes ('mpi' only): pack/unpack one field of a group, post, wait
  !
  !   trb_xa / trb_ya: grid A, x-pencil / y-pencil side (comm_row)
  !   trb_yb / trb_zb: grid B, y-pencil / z-pencil side (comm_col)
  !   f: field in the group, nb: fields in the group, slot: buffer slot,
  !   snd: this side sends (pack into / unpack from the send buffer), tobuf: pack (else unpack)
  ! ---------------------------------------------------------------------------
  !
  subroutine trb_xa(xa,f,nb,slot,snd,tobuf)
    implicit none
    complex(rp), intent(inout) :: xa(nxh,nzA,nyA)
    integer, intent(in) :: f,nb,slot
    logical, intent(in) :: snd,tobuf
    if(snd) then
      call cp_xa(xa,tbA_s(:,slot),tbA_s4(:,slot),f,nb,tobuf)
    else
      call cp_xa(xa,tbA_r(:,slot),tbA_r4(:,slot),f,nb,tobuf)
    end if
  end subroutine trb_xa
  !
  subroutine trb_ya(ya,f,nb,slot,snd,tobuf)
    implicit none
    complex(rp), intent(inout) :: ya(mz,nyA,nkx)
    integer, intent(in) :: f,nb,slot
    logical, intent(in) :: snd,tobuf
    if(snd) then
      call cp_ya(ya,tbA_s(:,slot),tbA_s4(:,slot),f,nb,tobuf)
    else
      call cp_ya(ya,tbA_r(:,slot),tbA_r4(:,slot),f,nb,tobuf)
    end if
  end subroutine trb_ya
  !
  subroutine trb_yb(yb,f,nb,slot,snd,tobuf)
    implicit none
    complex(rp), intent(inout) :: yb(nz,nyA,nkx)
    integer, intent(in) :: f,nb,slot
    logical, intent(in) :: snd,tobuf
    if(snd) then
      call cp_yb(yb,tbB_s(:,slot),tbB_s4(:,slot),f,nb,tobuf)
    else
      call cp_yb(yb,tbB_r(:,slot),tbB_r4(:,slot),f,nb,tobuf)
    end if
  end subroutine trb_yb
  !
  subroutine trb_zb(zb,f,nb,slot,snd,tobuf)
    implicit none
    complex(rp), intent(inout) :: zb(nkx,nkz,ny)
    integer, intent(in) :: f,nb,slot
    logical, intent(in) :: snd,tobuf
    if(snd) then
      call cp_zb(zb,tbB_s(:,slot),tbB_s4(:,slot),f,nb,tobuf)
    else
      call cp_zb(zb,tbB_r(:,slot),tbB_r4(:,slot),f,nb,tobuf)
    end if
  end subroutine trb_zb
  !
  subroutine cp_xa(xa,b,b4,f,nb,tobuf)
    ! (the branches are outside the loops, so that the copies vectorise)
    implicit none
    complex(rp), intent(inout) :: xa(nxh,nzA,nyA)
    complex(rp), intent(inout), contiguous :: b(:)
    complex(sp), intent(inout), contiguous :: b4(:)
    integer, intent(in) :: f,nb
    logical, intent(in) :: tobuf
    integer :: i,j,k,p
    if(tr_single) then
      if(tobuf) then
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do k=1,nyA
          do j=1,nzA
            do i=1,nxh
              p = own_kx(i)
              b4(nb*sd_xy(p)+(f-1)*sc_xy(p)+1+(i-kxlo_r(p))+nkx_r(p)*((j-1)+nzA*(k-1))) = cmplx(xa(i,j,k),kind=sp)
            end do
          end do
        end do
      else
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do k=1,nyA
          do j=1,nzA
            do i=1,nxh
              p = own_kx(i)
              xa(i,j,k) = cmplx(b4(nb*sd_xy(p)+(f-1)*sc_xy(p)+1+(i-kxlo_r(p))+nkx_r(p)*((j-1)+nzA*(k-1))),kind=rp)
            end do
          end do
        end do
      end if
    else
      if(tobuf) then
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do k=1,nyA
          do j=1,nzA
            do i=1,nxh
              p = own_kx(i)
              b(nb*sd_xy(p)+(f-1)*sc_xy(p)+1+(i-kxlo_r(p))+nkx_r(p)*((j-1)+nzA*(k-1))) = xa(i,j,k)
            end do
          end do
        end do
      else
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do k=1,nyA
          do j=1,nzA
            do i=1,nxh
              p = own_kx(i)
              xa(i,j,k) = b(nb*sd_xy(p)+(f-1)*sc_xy(p)+1+(i-kxlo_r(p))+nkx_r(p)*((j-1)+nzA*(k-1)))
            end do
          end do
        end do
      end if
    end if
  end subroutine cp_xa
  !
  subroutine cp_ya(ya,b,b4,f,nb,tobuf)
    ! (the branches are outside the loops, so that the copies vectorise)
    implicit none
    complex(rp), intent(inout) :: ya(mz,nyA,nkx)
    complex(rp), intent(inout), contiguous :: b(:)
    complex(sp), intent(inout), contiguous :: b4(:)
    integer, intent(in) :: f,nb
    logical, intent(in) :: tobuf
    integer :: i,j,k,p
    if(tr_single) then
      if(tobuf) then
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,mz
              p = own_z(j)
              b4(nb*rd_xy(p)+(f-1)*rc_xy(p)+1+(i-1)+nkx*((j-zlo_r(p))+nz_r(p)*(k-1))) = cmplx(ya(j,k,i),kind=sp)
            end do
          end do
        end do
      else
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,mz
              p = own_z(j)
              ya(j,k,i) = cmplx(b4(nb*rd_xy(p)+(f-1)*rc_xy(p)+1+(i-1)+nkx*((j-zlo_r(p))+nz_r(p)*(k-1))),kind=rp)
            end do
          end do
        end do
      end if
    else
      if(tobuf) then
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,mz
              p = own_z(j)
              b(nb*rd_xy(p)+(f-1)*rc_xy(p)+1+(i-1)+nkx*((j-zlo_r(p))+nz_r(p)*(k-1))) = ya(j,k,i)
            end do
          end do
        end do
      else
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,mz
              p = own_z(j)
              ya(j,k,i) = b(nb*rd_xy(p)+(f-1)*rc_xy(p)+1+(i-1)+nkx*((j-zlo_r(p))+nz_r(p)*(k-1)))
            end do
          end do
        end do
      end if
    end if
  end subroutine cp_ya
  !
  subroutine cp_yb(yb,b,b4,f,nb,tobuf)
    ! (the branches are outside the loops, so that the copies vectorise)
    implicit none
    complex(rp), intent(inout) :: yb(nz,nyA,nkx)
    complex(rp), intent(inout), contiguous :: b(:)
    complex(sp), intent(inout), contiguous :: b4(:)
    integer, intent(in) :: f,nb
    logical, intent(in) :: tobuf
    integer :: i,j,k,p
    if(tr_single) then
      if(tobuf) then
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,nz
              p = own_kz(j)
              b4(nb*sd_yz(p)+(f-1)*sc_yz(p)+1+(i-1)+nkx*((j-kzlo_c(p))+nkz_c(p)*(k-1))) = cmplx(yb(j,k,i),kind=sp)
            end do
          end do
        end do
      else
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,nz
              p = own_kz(j)
              yb(j,k,i) = cmplx(b4(nb*sd_yz(p)+(f-1)*sc_yz(p)+1+(i-1)+nkx*((j-kzlo_c(p))+nkz_c(p)*(k-1))),kind=rp)
            end do
          end do
        end do
      end if
    else
      if(tobuf) then
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,nz
              p = own_kz(j)
              b(nb*sd_yz(p)+(f-1)*sc_yz(p)+1+(i-1)+nkx*((j-kzlo_c(p))+nkz_c(p)*(k-1))) = yb(j,k,i)
            end do
          end do
        end do
      else
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do i=1,nkx
          do k=1,nyA
            do j=1,nz
              p = own_kz(j)
              yb(j,k,i) = b(nb*sd_yz(p)+(f-1)*sc_yz(p)+1+(i-1)+nkx*((j-kzlo_c(p))+nkz_c(p)*(k-1)))
            end do
          end do
        end do
      end if
    end if
  end subroutine cp_yb
  !
  subroutine cp_zb(zb,b,b4,f,nb,tobuf)
    ! (the branches are outside the loops, so that the copies vectorise)
    implicit none
    complex(rp), intent(inout) :: zb(nkx,nkz,ny)
    complex(rp), intent(inout), contiguous :: b(:)
    complex(sp), intent(inout), contiguous :: b4(:)
    integer, intent(in) :: f,nb
    logical, intent(in) :: tobuf
    integer :: i,j,k,p
    if(tr_single) then
      if(tobuf) then
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do k=1,ny
          do j=1,nkz
            do i=1,nkx
              p = own_y(k)
              b4(nb*rd_yz(p)+(f-1)*rc_yz(p)+1+(i-1)+nkx*((j-1)+nkz*(k-ylo_c(p)))) = cmplx(zb(i,j,k),kind=sp)
            end do
          end do
        end do
      else
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do k=1,ny
          do j=1,nkz
            do i=1,nkx
              p = own_y(k)
              zb(i,j,k) = cmplx(b4(nb*rd_yz(p)+(f-1)*rc_yz(p)+1+(i-1)+nkx*((j-1)+nkz*(k-ylo_c(p)))),kind=rp)
            end do
          end do
        end do
      end if
    else
      if(tobuf) then
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do k=1,ny
          do j=1,nkz
            do i=1,nkx
              p = own_y(k)
              b(nb*rd_yz(p)+(f-1)*rc_yz(p)+1+(i-1)+nkx*((j-1)+nkz*(k-ylo_c(p)))) = zb(i,j,k)
            end do
          end do
        end do
      else
        !$acc parallel loop collapse(3) default(present) private(p) async(1)
        do k=1,ny
          do j=1,nkz
            do i=1,nkx
              p = own_y(k)
              zb(i,j,k) = b(nb*rd_yz(p)+(f-1)*rc_yz(p)+1+(i-1)+nkx*((j-1)+nkz*(k-ylo_c(p))))
            end do
          end do
        end do
      end if
    end if
  end subroutine cp_zb
  !
  subroutine trb_post(stage,fwd,nb,slot)
    ! start the transpose of the nb packed fields in `slot`: stage 'A' (x <-> y) or 'B' (y <-> z),
    ! fwd: towards the spectral z-pencil (x -> y, y -> z), else backwards
    implicit none
    character(len=1), intent(in) :: stage
    logical, intent(in) :: fwd
    integer, intent(in) :: nb,slot
    !$acc wait(1)
    if(stage == 'A') then
      if(fwd) then
        csA(:,slot) = nb*sc_xy(:); dsA(:,slot) = nb*sd_xy(:); crA(:,slot) = nb*rc_xy(:); drA(:,slot) = nb*rd_xy(:)
      else
        csA(:,slot) = nb*rc_xy(:); dsA(:,slot) = nb*rd_xy(:); crA(:,slot) = nb*sc_xy(:); drA(:,slot) = nb*sd_xy(:)
      end if
      if(tr_single) then
        call ialltoallv4(tbA_s4(:,slot),csA(:,slot),dsA(:,slot),tbA_r4(:,slot),crA(:,slot),drA(:,slot),comm_row,reqA(slot))
      else
        call ialltoallv(tbA_s(:,slot),csA(:,slot),dsA(:,slot),tbA_r(:,slot),crA(:,slot),drA(:,slot),comm_row,reqA(slot))
      end if
    else
      if(fwd) then
        csB(:,slot) = nb*sc_yz(:); dsB(:,slot) = nb*sd_yz(:); crB(:,slot) = nb*rc_yz(:); drB(:,slot) = nb*rd_yz(:)
      else
        csB(:,slot) = nb*rc_yz(:); dsB(:,slot) = nb*rd_yz(:); crB(:,slot) = nb*sc_yz(:); drB(:,slot) = nb*sd_yz(:)
      end if
      if(tr_single) then
        call ialltoallv4(tbB_s4(:,slot),csB(:,slot),dsB(:,slot),tbB_r4(:,slot),crB(:,slot),drB(:,slot),comm_col,reqB(slot))
      else
        call ialltoallv(tbB_s(:,slot),csB(:,slot),dsB(:,slot),tbB_r(:,slot),crB(:,slot),drB(:,slot),comm_col,reqB(slot))
      end if
    end if
  end subroutine trb_post
  !
  subroutine ialltoallv(sb,sc,sd,rb,rc,rd,comm,req)
    ! non-blocking MPI_Alltoallv; the buffers are contiguous dummies, so MPI gets the arrays themselves
    ! (no temporary copies that would go out of scope before the transfer is complete)
    implicit none
    complex(rp), intent(inout), contiguous :: sb(:),rb(:)
    integer, intent(in), contiguous :: sc(:),sd(:),rc(:),rd(:)   ! must stay valid until the transfer is complete
    integer, intent(in) :: comm
    integer, intent(out) :: req
    integer :: ierr
    !$acc host_data use_device(sb,rb)
    call MPI_IALLTOALLV(sb,sc,sd,MPI_DOUBLE_COMPLEX,rb,rc,rd,MPI_DOUBLE_COMPLEX,comm,req,ierr)
    !$acc end host_data
  end subroutine ialltoallv
  !
  subroutine ialltoallv4(sb,sc,sd,rb,rc,rd,comm,req)
    implicit none
    complex(sp), intent(inout), contiguous :: sb(:),rb(:)
    integer, intent(in), contiguous :: sc(:),sd(:),rc(:),rd(:)   ! must stay valid until the transfer is complete
    integer, intent(in) :: comm
    integer, intent(out) :: req
    integer :: ierr
    !$acc host_data use_device(sb,rb)
    call MPI_IALLTOALLV(sb,sc,sd,MPI_COMPLEX,rb,rc,rd,MPI_COMPLEX,comm,req,ierr)
    !$acc end host_data
  end subroutine ialltoallv4
  !
  subroutine trb_wait(stage,slot)
    implicit none
    character(len=1), intent(in) :: stage
    integer, intent(in) :: slot
    integer :: ierr
    if(stage == 'A') then
      call MPI_WAIT(reqA(slot),MPI_STATUS_IGNORE,ierr)
    else
      call MPI_WAIT(reqB(slot),MPI_STATUS_IGNORE,ierr)
    end if
  end subroutine trb_wait
  !
  subroutine trb_test
    ! let MPI progress the transposes in flight (called between the FFTs of an overlapped group)
    implicit none
    integer :: ierr,n
    logical :: done
    do n=1,trb_nslot
      if(reqA(n) /= MPI_REQUEST_NULL) call MPI_TEST(reqA(n),done,MPI_STATUS_IGNORE,ierr)
      if(reqB(n) /= MPI_REQUEST_NULL) call MPI_TEST(reqB(n),done,MPI_STATUS_IGNORE,ierr)
    end do
  end subroutine trb_test
end module mod_decomp
