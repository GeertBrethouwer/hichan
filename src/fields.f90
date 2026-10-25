!
! field storage
!
module mod_fields
  use mod_types
  use mod_param , only: ny,mx,nscal,nl_form,scal_form
  use mod_decomp, only: nkx,nkz,nzA,nyA
  implicit none
  public
  ! spectral (z-pencils)
  complex(rp), allocatable, dimension(:,:,:)   :: u,v,w,nu,nv,nw,nuo,nvo,nwo,tmp1,tmp2
  complex(rp), allocatable, dimension(:,:,:,:) :: s,ns,nso
  ! physical (x-pencils on the dealiased grid)
  real(rp), allocatable, dimension(:,:,:)   :: up,vp,wp,ap
  real(rp), allocatable, dimension(:,:,:,:) :: gp
contains
  subroutine alloc_fields
    implicit none
    integer :: ng
    allocate(u(nkx,nkz,ny),v(nkx,nkz,ny),w(nkx,nkz,ny))
    allocate(nu(nkx,nkz,ny),nv(nkx,nkz,ny),nw(nkx,nkz,ny))
    allocate(nuo(nkx,nkz,ny),nvo(nkx,nkz,ny),nwo(nkx,nkz,ny))
    allocate(tmp1(nkx,nkz,ny),tmp2(nkx,nkz,ny))
    allocate(s(nkx,nkz,ny,max(nscal,1)),ns(nkx,nkz,ny,max(nscal,1)),nso(nkx,nkz,ny,max(nscal,1)))
    ! work arrays gp: u x omega needs 3; the divergence and skew scalar forms need 2 (add_flux_div)
    ng = 1
    if(nscal > 0 .and. trim(scal_form) /= 'advective') ng = 2
    if(trim(nl_form) == 'rotational') ng = 3
    allocate(up(mx,nzA,nyA),vp(mx,nzA,nyA),wp(mx,nzA,nyA),ap(mx,nzA,nyA),gp(mx,nzA,nyA,ng))
    u = 0; v = 0; w = 0; nu = 0; nv = 0; nw = 0; nuo = 0; nvo = 0; nwo = 0; tmp1 = 0; tmp2 = 0
    s = 0; ns = 0; nso = 0
    up = 0; vp = 0; wp = 0; ap = 0; gp = 0
    !$acc enter data copyin(u,v,w,nu,nv,nw,nuo,nvo,nwo,tmp1,tmp2,s,ns,nso,up,vp,wp,ap,gp)
  end subroutine alloc_fields
end module mod_fields
