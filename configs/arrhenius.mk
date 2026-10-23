#
# NSC Arrhenius, GPU partition: 4 x NVIDIA GH200 (Grace Hopper, sm_90) per node
#
#   module load GPU/buildenv-nvhpc/25.9-cu13.0     # NVIDIA HPC SDK + MPI; check `module avail buildenv`
#   ./tools/build_cudecomp.sh                       # once
#   make CONFIG=configs/arrhenius.mk
#   optional diezDecomp transposes as well (transpose_lib in &parallel):
#   ./tools/get_diezdecomp.sh; make CONFIG=configs/arrhenius.mk DIEZDECOMP=1
#
FC       := mpifort
GPU      := 1
CUDECOMP := 1
GPU_CC   := cc90
# -tp=neoverse-v2 targets the Grace CPU (host code); the GPU code is generated for sm_90
FFLAGS   := -O3 -fast -tp=neoverse-v2
CUDECOMP_DIR := $(CURDIR)/dependencies/cuDecomp/build
