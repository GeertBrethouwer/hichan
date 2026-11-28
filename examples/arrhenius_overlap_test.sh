#!/bin/bash
#
# Test and timing of the overlapped transposes (tr_overlap) and of single-precision transposes (transpose_prec) on
# Arrhenius, with a production case: four short runs from the same restart file,
#   A: as in the input   B: tr_overlap = .true.   C: transpose_prec = 'single'   D: both
# Checks: B must give a log.dat identical to A, and D identical to C (the overlap changes only when the work is
# done, not what is computed; a difference is a bug, e.g. a race between the queues). C differs from A by the
# single-precision rounding of the transposed data (expect ~1e-7 relative). The summary gives the time per step.
#
# Usage (from any directory; the runs go into ./overlap_test_<jobid>):
#   sbatch -A <your-account> --export=ALL,RUN=/scratch/geert/hichan_run $HOME/hichan/examples/arrhenius_overlap_test.sh
# RUN: a run directory with its input.nml and data/restart(.bin, .hdr). NSTEP (default 300) steps per run; the
# restart file and statistics of RUN are only read, nothing in RUN is changed. Use the node count of the
# production run (here 3 nodes, 12 GPUs; adjust -N).
#
#SBATCH -J overlap_test
#SBATCH -N 3
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH -t 02:00:00
#SBATCH -p gpu

module load GPU/buildenv-nvhpc/25.9-cu13.0
HICHAN=${HICHAN:-$HOME/hichan}
RUN=${RUN:-/scratch/geert/hichan_run}
NSTEP=${NSTEP:-300}
export LD_LIBRARY_PATH=$HICHAN/dependencies/cuDecomp/build/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
NNODES=${SLURM_JOB_NUM_NODES:-1}
if [ "$NNODES" -eq 1 ]; then  # as examples/arrhenius_job.sh
  export OMPI_MCA_btl=^ofi OMPI_MCA_mtl=^ofi
  NCCL_TASK_ENV="NCCL_NET_PLUGIN=none NCCL_NET=Socket"
else
  unset OMPI_MCA_btl OMPI_MCA_mtl
  NCCL_TASK_ENV=""
fi
[ -x $HICHAN/build/hichan ] || { echo "$HICHAN/build/hichan not found"; exit 1; }
[ -f $RUN/input.nml ] && [ -f $RUN/data/restart.hdr ] || { echo "need $RUN/input.nml and $RUN/data/restart.*"; exit 1; }
T=$SLURM_SUBMIT_DIR/overlap_test_${SLURM_JOB_ID:-local}
mkdir -p $T
echo "HiCHAN $(cd $HICHAN && git log -1 --format='%h %s' 2>/dev/null)"

# input of a test run: the production input with these items replaced or added (at the end of their group, where
# they override earlier values; groups that are missing are appended)
setnml() {  # setnml <in> <out> "group:item = value" ...
  local in=$1 out=$2; shift 2
  awk -v sets="$(printf '%s\n' "$@")" '
    BEGIN { n = split(sets, a, "\n"); for (i = 1; i <= n; i++) if (a[i] != "") { g = substr(a[i], 1, index(a[i], ":") - 1);
            add[tolower(g)] = add[tolower(g)] "  " substr(a[i], index(a[i], ":") + 1) "\n" } }
    /^[ \t]*&[A-Za-z]+/ { grp = tolower($1); sub(/^&/, "", grp); sub(/[ \t].*/, "", grp) }
    /^[ \t]*\/[ \t]*$/ { if (grp in add) { printf "%s", add[grp]; done[grp] = 1 } grp = "" }
    { print }
    END { for (g in add) if (!(g in done)) printf "&%s\n%s/\n", g, add[g] }' $in > $out
}
COMMON=("time:nstep = $NSTEP" "init:init_type = 'restart'" "init:restart_file = '$RUN/data/restart'"
        "output:data_dir = 'data'" "output:iout_restart = 0" "output:iout_log = 50" "output:timing = .true."
        "output:stats_restart = .false." "output:iout_field = 0" "output:tout_samples = 0." "output:iout_samples = 0")
declare -A PAR=( [A]="" [B]="parallel:tr_overlap = .true." [C]="parallel:transpose_prec = 'single'"
                 [D]="parallel:tr_overlap = .true.|parallel:transpose_prec = 'single'" )
for r in A B C D; do
  d=$T/$r; mkdir -p $d
  IFS='|' read -ra extra <<< "${PAR[$r]}"
  setnml $RUN/input.nml $d/input.nml "${COMMON[@]}" "${extra[@]}"
  echo "== run $r: ${PAR[$r]:-(as in the input)}"
  (cd $d && srun --mpi=pmix env $NCCL_TASK_ENV $HICHAN/build/hichan > out.txt 2>&1)
  echo "   exit code $?"
done

echo
echo "== summary ($NSTEP steps from t = $(awk 'NR==4{print $1}' $RUN/data/restart.hdr))"
for r in A B C D; do
  tps=$(grep -h "wall-clock time per time step" $T/$r/out.txt | awk '{print $(NF-1)}')
  med=$(grep -h "min / median / mean / max" $T/$r/out.txt | awk '{print $(NF-2)}')
  printf "  %s  %-55s time per step %s s (median %s s)\n" $r "${PAR[$r]:-(as in the input)}" "${tps:-?}" "${med:-?}"
  grep -h "transposes overlapped\|NOTE: tr_overlap\|error\|Error" $T/$r/out.txt | head -2 | sed 's/^/      /'
done
same() { cmp -s <(grep -v '^#' $T/$1/data/log.dat) <(grep -v '^#' $T/$2/data/log.dat); }
same A B && echo "  B = A: log.dat identical (overlap correct)" || echo "  B and A: log.dat DIFFERENT -- please send me $T/A and $T/B"
same C D && echo "  D = C: log.dat identical (overlap correct)" || echo "  D and C: log.dat DIFFERENT -- please send me $T/C and $T/D"
echo "  A vs C (single-precision transposes), last log lines:"
tail -1 $T/A/data/log.dat | cut -c1-110 | sed 's/^/     A /'
tail -1 $T/C/data/log.dat | cut -c1-110 | sed 's/^/     C /'
