#!/bin/bash
#
# Four independent simulations at the same time on ONE Arrhenius GPU node (4 x NVIDIA GH200):
# one simulation per GPU, each with one MPI rank, in the directories run1, run2, run3 and run4.
# Written for CaNS; works the same for HiCHAN (set EXE below). Every run directory needs its
# own input.nml (and restart files, if the run restarts).
#
# Check the account, partition and module names with the NSC/NAISS Arrhenius documentation
# (`module avail buildenv`, `sinfo`) -- the values below are placeholders/assumptions.
#
#SBATCH -J cans_4x1gpu
#SBATCH -A naiss20XX-X-XX          # your project
#SBATCH -N 1                       # one node
#SBATCH --ntasks=4                 # four simulations of one rank each
#SBATCH --gpus-per-node=4          # all four GH200 of the node
#SBATCH --cpus-per-task=72         # one Grace CPU (72 cores) per simulation
#SBATCH -t 24:00:00
##SBATCH -p gpu                    # GPU partition name on Arrhenius (check with sinfo)

module load GPU/buildenv-nvhpc/25.9-cu13.0      # the same module that was used to build the code

# executable (CaNS: the path of the 'cans' executable you built; HiCHAN: $HOME/hichan/build/hichan)
EXE=$HOME/CaNS/run/cans
# cuDecomp library used by the build (adjust the path to where your cuDecomp was built)
export LD_LIBRARY_PATH=$HOME/CaNS/dependencies/cuDecomp/build/lib:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
# NCCL is not needed between GPUs here (one rank per run); avoid the broken OFI network plugin.
# The site sets NCCL_NET_PLUGIN=.../libnccl-net-ofi.so and NCCL_NET="AWS Libfabric" again in every task, after
# srun has copied this environment, so they are set inside each task (in the bash -c below), not here.
export NCCL_DEBUG=WARN

cd $SLURM_SUBMIT_DIR
if [ ! -x "$EXE" ]; then echo "executable $EXE not found"; exit 1; fi

for i in 1 2 3 4; do
  dir=run$i
  if [ ! -f $dir/input.nml ]; then echo "$dir/input.nml missing, $dir not started"; continue; fi
  mkdir -p $dir/data                              # CaNS (and HiCHAN) write into data/
  # one job step per run: 1 task, its own GPU and its own 72 cores (--exact: do not share
  # resources between the steps, so the four steps run at the same time on different GPUs)
  srun --exact --ntasks=1 --gpus=1 --cpus-per-task=72 --mpi=pmix \
       --chdir=$dir --output=log_slurm.out --error=log_slurm.err \
       bash -c 'export NCCL_NET_PLUGIN=none NCCL_NET=Socket
                echo "$(hostname): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; exec '"$EXE" &
  sleep 2                                         # stagger the start-ups a little
done
wait                                              # until all four simulations have finished
echo "all runs finished: $(date)"

# If your Slurm version rejects --exact, replace it by --exclusive (older Slurm).
#
# Alternative (if the steps above wait for each other or end up on the same GPU; check the
# CUDA_VISIBLE_DEVICES line at the top of runN/log_slurm.out): one srun with four tasks, each
# task picks its directory and GPU from its local task id and runs as a one-rank MPI job
# (--mpi=none: each task is its own MPI "singleton", which is fine for one rank). Replace the
# for loop and the wait by:
#   srun --ntasks=4 --cpus-per-task=72 --mpi=none bash -c '
#     i=$((SLURM_LOCALID+1)); export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID NCCL_NET_PLUGIN=none NCCL_NET=Socket
#     cd run$i && mkdir -p data && exec '"$EXE"' > log_slurm.out 2> log_slurm.err'
