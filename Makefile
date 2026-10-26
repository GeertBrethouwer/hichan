#
# HiCHAN Makefile
#
#   make                       CPU build (gfortran/any MPI wrapper + FFTW3)
#   make GPU=1                 GPU build with nvfortran (OpenACC + cuFFT), MPI transposes
#                              (requires CUDA-aware MPI)
#   make GPU=1 CUDECOMP=1      GPU build with cuDecomp transposes (recommended)
#   make ... CUFFT_AUTO_WORK=1 only if the shared cuFFT work area causes problems: every cuFFT
#                              plan then allocates its own work area (more GPU memory)
#   make ... SBP=1             wider band storage for y_scheme = 'sbp48' (SBP 4-8 prototype);
#                              more memory and slower banded solves, also for the compact schemes
#   make ... DIEZDECOMP=1      add the diezDecomp transposes (tools/get_diezdecomp.sh first);
#                              CPU or GPU, can be combined with CUDECOMP=1. The library is chosen
#                              at run time with transpose_lib in &parallel
#
# Settings for a specific machine can be put in a file and passed with
#   make CONFIG=configs/arrhenius.mk
#
CONFIG ?=
ifneq ($(strip $(CONFIG)),)
include $(CONFIG)
endif

ifeq ($(origin FC),default)
FC := mpif90
endif
GPU       ?= 0
CUDECOMP  ?= 0
DIEZDECOMP ?= 0
SBP       ?= 0
DIEZDECOMP_DIR ?= $(CURDIR)/dependencies/diezDecomp
DEBUG     ?= 0
GPU_CC    ?= cc90
CUDECOMP_DIR ?= $(CURDIR)/dependencies/cuDecomp/build
FFTW_INC  ?= /usr/include
FFTW_LIB  ?= -lfftw3

SRCDIR   := src
BUILDDIR := build
EXE      := $(BUILDDIR)/hichan

DZOBJS :=
SRCS := types.f90 param.f90 compact.f90 decomp.f90 timer.f90 fft.f90 ysolve.f90 fields.f90 \
        rk.f90 stats.f90 io.f90 initflow.f90 main.f90
OBJS := $(addprefix $(BUILDDIR)/,$(SRCS:.f90=.o))

DEFINES :=
LIBS    :=
INCS    :=

ifeq ($(strip $(CUFFT_AUTO_WORK)),1)
  DEFINES += -D_CUFFT_AUTO_WORK
endif
ifeq ($(strip $(SBP)),1)
  DEFINES += -D_SBP48
endif
ifeq ($(strip $(DIEZDECOMP)),1)
  DEFINES += -D_USE_DIEZDECOMP
  DZOBJS  := $(BUILDDIR)/diezdecomp_core.o $(BUILDDIR)/diezdecomp_api_generic.o
endif

ifeq ($(strip $(GPU)),1)
  # NVIDIA HPC SDK (nvfortran)
  FFLAGS ?= -O3 -fast
  FFLAGS += -cpp -acc -cuda -gpu=$(GPU_CC) -Minfo=accel -module $(BUILDDIR)
  LIBS   += -cudalib=cufft
  ifeq ($(strip $(CUDECOMP)),1)
    DEFINES += -D_USE_CUDECOMP
    INCS    += -I$(CUDECOMP_DIR)/include
    LIBS    += -L$(CUDECOMP_DIR)/lib -lcudecomp -lcudecomp_fort
  endif
else
  ifeq ($(DEBUG),1)
    FFLAGS ?= -O0 -g -fbacktrace -fcheck=all -ffpe-trap=invalid,zero,overflow -Wall -Wno-unused-dummy-argument
  else
    FFLAGS ?= -O3 -march=native
  endif
  FFLAGS += -cpp -ffree-line-length-none -J$(BUILDDIR)
  INCS   += -I$(FFTW_INC)
  LIBS   += $(FFTW_LIB)
endif

.PHONY: all clean ygrid
all: $(EXE)

# stand-alone tool: wall-normal grid y_j, dy_j (and y+, dy+) for given ny, grid type, alpha
ygrid: $(BUILDDIR)/ygrid
$(BUILDDIR)/ygrid: tools/ygrid.f90 | $(BUILDDIR)
	gfortran -O2 $< -o $@ || $(FC) -O2 $< -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(EXE): $(DZOBJS) $(OBJS)
	$(FC) $(FFLAGS) $(DZOBJS) $(OBJS) $(LIBS) -o $@

# diezDecomp (external, MIT licence): compiled with the same flags
$(BUILDDIR)/diezdecomp_core.o: $(DIEZDECOMP_DIR)/src/diezdecomp_core.f90 | $(BUILDDIR)
	$(FC) $(FFLAGS) -c $< -o $@
$(BUILDDIR)/diezdecomp_api_generic.o: $(DIEZDECOMP_DIR)/src/diezdecomp_api_generic.f90 $(BUILDDIR)/diezdecomp_core.o
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.f90 | $(BUILDDIR)
	$(FC) $(FFLAGS) $(DEFINES) $(INCS) -c $< -o $@

# module dependencies
$(BUILDDIR)/param.o:    $(BUILDDIR)/types.o
$(BUILDDIR)/compact.o:  $(BUILDDIR)/param.o
$(BUILDDIR)/decomp.o:   $(BUILDDIR)/param.o $(DZOBJS)
$(BUILDDIR)/timer.o:    $(BUILDDIR)/decomp.o
$(BUILDDIR)/fft.o:      $(BUILDDIR)/decomp.o $(BUILDDIR)/timer.o
$(BUILDDIR)/ysolve.o:   $(BUILDDIR)/decomp.o $(BUILDDIR)/compact.o
$(BUILDDIR)/fields.o:   $(BUILDDIR)/decomp.o
$(BUILDDIR)/rk.o:       $(BUILDDIR)/fields.o $(BUILDDIR)/fft.o $(BUILDDIR)/ysolve.o $(BUILDDIR)/timer.o
$(BUILDDIR)/stats.o:    $(BUILDDIR)/ysolve.o
$(BUILDDIR)/io.o:       $(BUILDDIR)/decomp.o
$(BUILDDIR)/initflow.o: $(BUILDDIR)/ysolve.o $(BUILDDIR)/fields.o
$(BUILDDIR)/main.o:     $(BUILDDIR)/timer.o $(BUILDDIR)/rk.o $(BUILDDIR)/stats.o $(BUILDDIR)/io.o $(BUILDDIR)/initflow.o

clean:
	rm -rf $(BUILDDIR)
