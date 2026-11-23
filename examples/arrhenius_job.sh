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
# NCCL network, chosen from the number of nodes of the job. The site sets NCCL_NET_PLUGIN=.../libnccl-net-ofi.so
# and NCCL_NET="AWS Libfabric" in every task, after srun has copied this environment, so `export` here does not
# reach NCCL: a different choice is given on the srun command line with env (NCCL_TASK_ENV).
# - one node: NCCL uses NVLink between the GPUs and needs no network. The site's libfabric (OFI) plugin failed in
#   single-node runs ("NET/OFI Couldn't create domain ... Function not implemented" -> "Failed to initialize any
#   NET plugin" -> ncclCommInitRank error 5), so it is switched off: NCCL_NET_PLUGIN=none NCCL_NET=Socket
#   (Socket is only NCCL's transport between nodes).
# - several nodes: the site's settings (AWS OFI plugin, libfabric provider cxi, one Slingshot NIC per GPU).
#   Tested on two nodes with FDCHAN's examples/arrhenius_nccl_test.sh: about 21 GB/s per GPU between the
#   nodes; NCCL_NET=Socket gave 1.7 GB/s, so never use it on several nodes. Keep cudecomp_enable_nccl = .true.
#   The same test: Open MPI works between the nodes with the site's settings (do not set OMPI_MCA_btl=^ofi).
NNODES=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}
if [ "$NNODES" -eq 1 ]; then
  NCCL_TASK_ENV="NCCL_NET_PLUGIN=none NCCL_NET=Socket"
else
  NCCL_TASK_ENV=""
fi
env | grep -E '^(NCCL|FI)_' | sort
echo "$NNODES node(s); set in every task: ${NCCL_TASK_ENV:-(nothing: the site's NCCL settings)}"

cd $SLURM_SUBMIT_DIR
# The MPI of the NVIDIA HPC SDK (Open MPI/HPC-X) needs PMIx. With a plain srun/mpprun the tasks
# start as independent MPI "singletons" ("No PMIx server was reachable ... singletons will be
# started"), all use GPU 0 and cuDecomp fails in ncclCommInitRank. Check `srun --mpi=list` and use:
# HiCHAN prints the GPU used by each rank; every rank must use a different GPU.
srun --mpi=pmix env $NCCL_TASK_ENV $HICHAN/build/hichan
# check what the tasks see:  srun -n 1 env $NCCL_TASK_ENV bash -c 'env | grep ^NCCL_NET'
# if Slurm hands each task its own GPU (or the ranks end up on the same GPU), bind explicitly:
#   srun --mpi=pmix --gpus-per-task=1 --gpu-bind=single:1 env $NCCL_TASK_ENV $HICHAN/build/hichan
# if NCCL still fails, set  cudecomp_enable_nccl = .false.  in &parallel (MPI backends only)
# alternatives:
#   mpirun -np $SLURM_NTASKS --map-by ppr:4:node env $NCCL_TASK_ENV $HICHAN/build/hichan   # MPI used to build
#   mpprun env $NCCL_TASK_ENV $HICHAN/build/hichan                    # only if HiCHAN was built with an
#                                                                     # MPI that mpprun recognises
# A correct start prints the input namelist ONCE and "number of MPI processes: <SLURM_NTASKS>".
