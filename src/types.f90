!
! HiCHAN - high-order Fourier/compact-FD solver for incompressible channel flow
!
! kinds and constants
!
module mod_types
  use, intrinsic :: iso_fortran_env, only: real32, real64, int64
  implicit none
  integer , parameter :: sp = real32
  integer , parameter :: dp = real64
  integer , parameter :: i8 = int64
  !
  ! the whole calculation is done in double precision: the influence-matrix
  ! systems are ill-conditioned (see Lenaers et al., JCP 272, 2014, Sec. 5.2)
  !
  integer , parameter :: rp = dp
  real(rp), parameter :: pi = acos(-1._rp)
end module mod_types
