#!/bin/bash
# usage: mms_run.sh TEST NY DT [key=value ...]   (run in an empty directory; MMS=path to build/mms, NP=ranks)
# writes input.nml for the manufactured-solution test and runs it; extra key=value pairs go into &physics
# (re, nl_form, visc_theta, ns_method), &grid (fd_order, grid_alpha, nx, nz), &scalars (nscal, scal_form,
# scal_theta, bc1/bc2 = wall conditions of scalars 1/2 as DD, DN, ND, NN) or &buoyancy (ri: Richardson number
# of scalar 1; scalar 2 always has ri = 0.5 when nscal = 2)
T=$1; NY=$2; DT=$3; shift 3
MMS=${MMS:-$(dirname $0)/../build/mms}; NP=${NP:-1}
RE=100; NL=rotational; TH=0.0; NS=im; FO=6; GA=3.0; TEND=0.5; PD="1, 1"; NX=16; NZ=16
NSC=0; SF=advective; STH=0.0; BC1=DD; BC2=NN; RI=0.0
for kv in "$@"; do case ${kv%%=*} in
  re) RE=${kv#*=};; nl_form) NL=${kv#*=};; visc_theta) TH=${kv#*=};; ns_method) NS=${kv#*=};;
  fd_order) FO=${kv#*=};; grid_alpha) GA=${kv#*=};; tend) TEND=${kv#*=};; pdims) PD=${kv#*=};;
  nx) NX=${kv#*=};; nz) NZ=${kv#*=};;
  nscal) NSC=${kv#*=};; scal_form) SF=${kv#*=};; scal_theta) STH=${kv#*=};; bc1) BC1=${kv#*=};; bc2) BC2=${kv#*=};;
  ri) RI=${kv#*=};; esac; done
cat > input.nml <<NML
&grid nx = $NX, nz = $NZ, ny = $NY, lx = 6.283185307179586, lz = 3.141592653589793,
  grid_type = 'tanh', grid_alpha = $GA, fd_order = $FO /
&physics re = $RE, forcing = 'cpg', dpdx = 0.0, nl_form = '$NL', visc_theta = $TH, ns_method = '$NS',
  impl_tol = 1.e-13, impl_maxit = 50 /
&time dt = $DT, fixed_dt = .true. /
&init /
&output /
&scalars nscal = $NSC, scal_form = '$SF',
  pr(1) = 0.71, scal_theta(1) = $STH, sbc_bot(1) = '${BC1:0:1}', sbc_top(1) = '${BC1:1:1}', sval_bot(1) = 1.0,
  sval_top(1) = 0.5, ssrc_const(1) = 0.2, ssrc_u(1) = 0.3,
  pr(2) = 2.0, scal_theta(2) = $STH, sbc_bot(2) = '${BC2:0:1}', sbc_top(2) = '${BC2:1:1}', sval_bot(2) = -0.4,
  sval_top(2) = 0.8 /
&buoyancy ri(1) = $RI, ri(2) = 0.5 /
&parallel pdims = $PD /
&mms test = '$T', tend = $TEND /
NML
mpirun -n $NP $MMS 2>&1 | grep -E "MMS|error|Error|abort"
