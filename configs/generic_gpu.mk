#
# any NVIDIA GPU system with the NVIDIA HPC SDK (nvfortran) and a CUDA-aware MPI
#
FC       := mpifort
GPU      := 1
CUDECOMP := 1
GPU_CC   := cc80,cc90
FFLAGS   := -O3 -fast
CUDECOMP_DIR := $(CURDIR)/dependencies/cuDecomp/build
