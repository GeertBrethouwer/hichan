#!/bin/bash
#
# download and build NVIDIA cuDecomp (Fortran bindings) into dependencies/cuDecomp/build
# requires the NVIDIA HPC SDK (nvc++, nvfortran, CUDA, NCCL) and cmake >= 3.18
#
#   NVSHMEM=1 ./tools/build_cudecomp.sh    to enable the NVSHMEM backends
#
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
mkdir -p dependencies
if [ ! -d dependencies/cuDecomp ]; then
  git clone https://github.com/NVIDIA/cuDecomp dependencies/cuDecomp
  # same revision as the CaNS v4.2.0 submodule
  git -C dependencies/cuDecomp checkout 1635dc37b8af38dbef187f2460685409d996ffdd
fi
cd dependencies/cuDecomp
mkdir -p build && cd build
cmake .. -DCMAKE_C_COMPILER=nvc -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_Fortran_COMPILER=nvfortran \
         -DCUDECOMP_CUDA_CC_LIST=90 \
         -DCUDECOMP_BUILD_FORTRAN=ON -DCUDECOMP_ENABLE_NVSHMEM=${NVSHMEM:-OFF}
make -j 16
echo "cuDecomp built in $PWD (Fortran module in include/, libraries in lib/)"
echo "at run time: export LD_LIBRARY_PATH=$PWD/lib:\$LD_LIBRARY_PATH"
