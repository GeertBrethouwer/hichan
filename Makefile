#
# HiCHAN Makefile
#
#   make                       CPU build (gfortran/any MPI wrapper + FFTW3)
#   make GPU=1                 GPU build with nvfortran (OpenACC + cuFFT), MPI transposes
#                              (requires CUDA-aware MPI)
#   make GPU=1 CUDECOMP=1      GPU build with cuDecomp transposes (recommended)
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
DEBUG     ?= 0
GPU_CC    ?= cc90
CUDECOMP_DIR ?= $(CURDIR)/dependencies/cuDecomp/build
FFTW_INC  ?= /usr/include
FFTW_LIB  ?= -lfftw3

SRCDIR   := src
BUILDDIR := build
EXE      := $(BUILDDIR)/hichan

SRCS := types.f90 param.f90 compact.f90 decomp.f90 fft.f90 ysolve.f90 fields.f90 \
        rk.f90 stats.f90 io.f90 initflow.f90 main.f90
OBJS := $(addprefix $(BUILDDIR)/,$(SRCS:.f90=.o))

DEFINES :=
LIBS    :=
INCS    :=

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

$(EXE): $(OBJS)
	$(FC) $(FFLAGS) $(OBJS) $(LIBS) -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.f90 | $(BUILDDIR)
	$(FC) $(FFLAGS) $(DEFINES) $(INCS) -c $< -o $@

# module dependencies
$(BUILDDIR)/param.o:    $(BUILDDIR)/types.o
$(BUILDDIR)/compact.o:  $(BUILDDIR)/param.o
$(BUILDDIR)/decomp.o:   $(BUILDDIR)/param.o
$(BUILDDIR)/fft.o:      $(BUILDDIR)/decomp.o
$(BUILDDIR)/ysolve.o:   $(BUILDDIR)/decomp.o $(BUILDDIR)/compact.o
$(BUILDDIR)/fields.o:   $(BUILDDIR)/decomp.o
$(BUILDDIR)/rk.o:       $(BUILDDIR)/fields.o $(BUILDDIR)/fft.o $(BUILDDIR)/ysolve.o
$(BUILDDIR)/stats.o:    $(BUILDDIR)/ysolve.o
$(BUILDDIR)/io.o:       $(BUILDDIR)/decomp.o
$(BUILDDIR)/initflow.o: $(BUILDDIR)/ysolve.o
$(BUILDDIR)/main.o:     $(BUILDDIR)/rk.o $(BUILDDIR)/stats.o $(BUILDDIR)/io.o $(BUILDDIR)/initflow.o

clean:
	rm -rf $(BUILDDIR)
