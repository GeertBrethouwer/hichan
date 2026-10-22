!
! ygrid: HiCHAN wall-normal grid y_j and spacing dy_j for given ny, grid type and alpha
! (same formula as src/compact.f90, init_grid)
!
!   gfortran -O2 tools/ygrid.f90 -o ygrid
!   ./ygrid ny grid_type [grid_alpha] [re_tau]
!   ./ygrid 129 tanh 3.0           y, dy in units of the half-height h
!   ./ygrid 193 tanh 4.0 395       also y+ (from the bottom wall) and dy+
!   ./ygrid 65 uniform
!
! dy_j = y_{j+1} - y_j (spacing above point j). The summary gives the spacing at the wall and at
! the centre, and the largest ratio of neighbouring spacings.
!
program ygrid
  implicit none
  integer, parameter :: dp = kind(1.d0)
  integer :: ny,j,nargs
  character(len=32) :: arg,grid_type
  real(dp) :: alpha,retau,xi,rmax
  real(dp), allocatable :: y(:),dy(:)
  !
  nargs = command_argument_count()
  if(nargs < 2) then
    write(*,'(a)') 'usage: ygrid ny grid_type(tanh|uniform) [grid_alpha] [re_tau]'
    stop 1
  end if
  call get_command_argument(1,arg); read(arg,*) ny
  call get_command_argument(2,grid_type)
  alpha = 2._dp; retau = 0._dp
  if(nargs >= 3) then
    call get_command_argument(3,arg); read(arg,*) alpha
  end if
  if(nargs >= 4) then
    call get_command_argument(4,arg); read(arg,*) retau
  end if
  if(ny < 3) stop 'ny must be >= 3'
  !
  allocate(y(ny),dy(ny-1))
  do j=1,ny
    xi = (j-1._dp)/(ny-1._dp)
    select case(trim(grid_type))
    case('tanh')
      if(alpha <= 0._dp) stop 'grid_alpha must be > 0 for a tanh grid'
      y(j) = tanh(alpha*(xi-0.5_dp))/tanh(0.5_dp*alpha)
    case('uniform')
      y(j) = -1._dp + 2._dp*xi
    case default
      stop 'grid_type must be tanh or uniform'
    end select
  end do
  y(1) = -1._dp; y(ny) = 1._dp
  dy(:) = y(2:ny) - y(1:ny-1)
  !
  if(retau > 0._dp) then
    write(*,'(a5,4a16)') '#   j','y','dy','y+','dy+'
  else
    write(*,'(a5,2a16)') '#   j','y','dy'
  end if
  do j=1,ny
    if(retau > 0._dp) then
      if(j < ny) then
        write(*,'(i5,4es16.7)') j,y(j),dy(j),(1._dp+y(j))*retau,dy(j)*retau
      else
        write(*,'(i5,es16.7,16x,es16.7)') j,y(j),(1._dp+y(j))*retau
      end if
    else
      if(j < ny) then
        write(*,'(i5,2es16.7)') j,y(j),dy(j)
      else
        write(*,'(i5,es16.7)') j,y(j)
      end if
    end if
  end do
  rmax = maxval(max(dy(2:)/dy(:ny-2),dy(:ny-2)/dy(2:)))
  if(trim(grid_type) == 'tanh') then
    write(*,'(a,i0,a,f6.3)') '# ny = ',ny,', grid tanh, alpha = ',alpha
  else
    write(*,'(a,i0,a)') '# ny = ',ny,', grid uniform'
  end if
  write(*,'(a,es12.5,a,es12.5,a,f7.4)') '# dy at the wall = ',dy(1),', at the centre = ',dy(ny/2), &
                                        ', max. ratio of neighbouring dy = ',rmax
  if(retau > 0._dp) write(*,'(a,f8.3,a,f8.3,a,i0)') '# dy+ at the wall = ',dy(1)*retau, &
    ', at the centre = ',dy(ny/2)*retau,', points with y+ < 10 (bottom half): ',count((1._dp+y)*retau < 10._dp)
end program ygrid
