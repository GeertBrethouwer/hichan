#!/bin/bash
#
# Quick checks of a HiCHAN build on NSC Arrhenius: examples/ts_wave and examples/scalar_check on one GPU.
# Both compare with known values (see TEST.md).
#
# Submit from the HiCHAN directory:  sbatch examples/arrhenius_tests.sh
# The runs are done in tests_<jobid>/ (the example directories stay clean).
#
# Check the account, partition and module names with the NSC/NAISS Arrhenius documentation.
#
#SBATCH -J hichan-tests
#SBATCH -A naiss20XX-X-XX          # your project
#SBATCH -N 1
#SBATCH --ntasks=1                 # the test grids are tiny: one rank, one GPU
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 01:00:00                # ts_wave 15000 steps, scalar_check 158000 steps (CPU: 7 s, 6.5 min)
##SBATCH -p gpu                    # GPU partition name on Arrhenius (check with sinfo)

module load GPU/buildenv-nvhpc/25.9-cu13.0
# ts_wave needs Python 3 with numpy and scipy (tools/os_mode.py), scalar_check numpy only.
# Load a Python module here if python3 has no numpy/scipy (check with `module avail python`).
PYTHON=${PYTHON:-python3}

HICHAN=${SLURM_SUBMIT_DIR:-$PWD}                     # the HiCHAN directory (submit from there)
export LD_LIBRARY_PATH="$HICHAN/dependencies/cuDecomp/build/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1
# GPU-aware MPI and NCCL on one node, as in examples/arrhenius_job.sh. Without NCCL_NET=Socket NCCL fails
# with "cudecomp.cc:69 NCCL error (error code 5)" (its libfabric plugin does not work in the job, see README)
export MPICH_GPU_SUPPORT_ENABLED=1
export UCX_MEMTYPE_CACHE=n
export NCCL_DEBUG=WARN                               # NCCL prints the reason if it fails
export NCCL_NET_PLUGIN=none
export NCCL_NET=Socket
# if NCCL still fails: sbatch --export=ALL,NO_NCCL=1 examples/arrhenius_tests.sh (cuDecomp's MPI backends only)
NO_NCCL=${NO_NCCL:-0}
LAUNCH=${LAUNCH:-"srun --mpi=pmix"}                  # e.g. LAUNCH="mpirun -np 1" for a CPU build

RUN="$HICHAN/tests_${SLURM_JOB_ID:-local}"
mkdir -p "$RUN"

echo "=== ts_wave: decay rate of a TS wave (Re = 5000, alpha = 1.12) vs. Orr-Sommerfeld"
cp -r "$HICHAN/examples/ts_wave" "$RUN/" && cd "$RUN/ts_wave" || exit 1
[ "$NO_NCCL" = 1 ] && printf '&parallel\n  cudecomp_enable_nccl = .false.\n/\n' >> input.nml
"$PYTHON" "$HICHAN/tools/os_mode.py" --re 5000 --alpha 1.12 --ny 65 --grid tanh --galpha 4.0 -o mode.in
if $LAUNCH "$HICHAN/build/hichan" > out.txt 2>&1; then
  "$PYTHON" check.py
else
  echo "hichan failed, see $PWD/out.txt"; tail -20 out.txt
fi
echo "    expected: rel. error 3.9e-5 (CPU build); the GPU build should agree to several digits"

echo "=== scalar_check: wall gradients of five scalars in laminar Poiseuille flow vs. exact solutions"
cp -r "$HICHAN/examples/scalar_check" "$RUN/" && cd "$RUN/scalar_check" || exit 1
[ "$NO_NCCL" = 1 ] && printf '&parallel\n  cudecomp_enable_nccl = .false.\n/\n' >> input.nml
if $LAUNCH "$HICHAN/build/hichan" > out.txt 2>&1; then
  "$PYTHON" check.py
else
  echo "hichan failed, see $PWD/out.txt"; tail -20 out.txt
fi
echo "    expected: max rel. error 0 (exact to the 7 digits of log.dat)"

echo "=== run directories and full output: $RUN"
