#!/bin/bash
#
# HiCHAN on NSC Arrhenius (GPU nodes: 4 x NVIDIA GH200 each), one MPI rank per GPU
#
# Check the account, partition and module names with the NSC/NAISS Arrhenius documentation
# (`module avail buildenv`, `sinfo`) -- the values below are placeholders/assumptions.
#
#SBATCH -J hichan
#SBATCH -A naiss20XX-X-XX          # your project
#SBATCH -N 2                       # nodes
#SBATCH --ntasks-per-node=4        # one rank per GH200
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72         # one Grace CPU (72 cores) per rank
#SBATCH -t 24:00:00
##SBATCH -p gpu                    # GPU partition name on Arrhenius (check with sinfo)

module load GPU/buildenv-nvhpc/25.9-cu13.0

HICHAN=$HOME/hichan                                  # where HiCHAN was built
export LD_LIBRARY_PATH=$HICHAN/dependencies/cuDecomp/build/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
# GPU-aware MPI (needed by cuDecomp's MPI backends and by the non-cuDecomp build):
export MPICH_GPU_SUPPORT_ENABLED=1                   # Cray MPICH
export UCX_MEMTYPE_CACHE=n                           # Open MPI / UCX
export NCCL_DEBUG=WARN                               # NCCL prints the reason if it fails
# NCCL's libfabric (OFI) network plugin fails on Arrhenius with "NET/OFI Couldn't create domain ...
# Function not implemented" -> "Failed to initialize any NET plugin" -> ncclCommInitRank error 5.
# Single node: NCCL uses NVLink between the GPUs and does not need the plugin, so disable it:
export NCCL_NET_PLUGIN=none
export NCCL_NET=Socket
# Multi-node: NCCL needs a working OFI plugin for Slingshot (ask NSC which one; typically
# FI_PROVIDER=cxi, NCCL_NET="AWS Libfabric", NCCL_CROSS_NIC=1), or set cudecomp_enable_nccl = .false.
# in &parallel to use cuDecomp's (CUDA-aware) MPI backends only.
env | grep -E '^(NCCL|FI)_' | sort

cd $SLURM_SUBMIT_DIR
# The MPI of the NVIDIA HPC SDK (Open MPI/HPC-X) needs PMIx. With a plain srun/mpprun the tasks
# start as independent MPI "singletons" ("No PMIx server was reachable ... singletons will be
# started"), all use GPU 0 and cuDecomp fails in ncclCommInitRank. Check `srun --mpi=list` and use:
# HiCHAN prints the GPU used by each rank; every rank must use a different GPU.
srun --mpi=pmix $HICHAN/build/hichan
# if Slurm hands each task its own GPU (or the ranks end up on the same GPU), bind explicitly:
#   srun --mpi=pmix --gpus-per-task=1 --gpu-bind=single:1 $HICHAN/build/hichan
# if NCCL still fails, set  cudecomp_enable_nccl = .false.  in &parallel (MPI backends only)
# alternatives:
#   mpirun -np $SLURM_NTASKS --map-by ppr:4:node $HICHAN/build/hichan   # mpirun of the MPI used to build
#   mpprun $HICHAN/build/hichan                                       # only if HiCHAN was built with an
#                                                                     # MPI that mpprun recognises
# A correct start prints the input namelist ONCE and "number of MPI processes: <SLURM_NTASKS>".
