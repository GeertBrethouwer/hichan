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
# NCCL over the node interconnect (Slingshot): uncomment if NCCL backends are selected by the autotuner
#export NCCL_NET="AWS Libfabric"

cd $SLURM_SUBMIT_DIR
# NSC recommends mpprun; srun with GPU binding also works
mpprun $HICHAN/build/hichan
# srun --gpus-per-task=1 --gpu-bind=closest --cpu-bind=cores $HICHAN/build/hichan
