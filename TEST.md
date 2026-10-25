# Testing batched and overlapped transforms on a cluster

The options `tr_batch` and `tr_overlap` (see PERF.md, section 4b) are bit-identical to the default and as fast
as the default on a single node, where there is no network latency to hide. Their gain has to be measured on
several nodes.

## First: check the build

Before the timing runs, check the GPU build with two small cases that compare with known values, on one GPU:

```bash
sbatch examples/arrhenius_tests.sh
```

Submit it from the HiCHAN directory after setting your project in `#SBATCH -A`. The runs are done in
`tests_<jobid>/`. `tools/os_mode.py` needs Python 3 with numpy and scipy; load a Python module in the script if
the default `python3` does not have them. Expected output (CPU build):

| Test | Checks | Expected |
|------|--------|----------|
| `ts_wave` | decay rate of a TS wave (Re = 5000, α = 1.12, Ny = 65) vs. Orr–Sommerfeld | rel. error 3.9e-5 |
| `scalar_check` | wall gradients of five scalars in laminar Poiseuille flow vs. exact solutions | 5, −5 \| 5, −5 \| 0.5, 0.5 \| 0.5, 0.5 \| 0.2, 0.2, error 0 to the printed digits |

`ts_wave` takes 15 000 steps and `scalar_check` 158 000 (7 s and 6.5 min on one CPU core). On a GPU these tiny
grids are slower per step than the large ones, so the job asks for one hour.

## Runs

Run the same case three times, about 200 steps each (`nstep = 200` in `&time`), with `timing = .true.` in
`&output` and `transpose_lib = 'mpi'` in `&parallel`:

| Run | `&parallel` settings                   |
|-----|----------------------------------------|
| A   | `tr_batch = 1` (current behaviour)     |
| B   | `tr_batch = 3`                         |
| C   | `tr_batch = 3, tr_overlap = .true.`    |

## Checks

- **Gain:** compare the time per step of `FFTs incl. transposes (all calls)` in `timing_report.txt`. This is
  where any gain appears.
- **Transpose library:** the options work only with `transpose_lib = 'mpi'`. With cuDecomp or diezDecomp they
  are ignored, and hichan prints a note saying so.
- **Overlap (run C):** set `MPICH_ASYNC_PROGRESS=1`, or the equivalent of your MPI. Without background progress
  the communication cannot overlap with the FFTs, and C will time like B.
- **Memory:** check the memory report at startup before running on full GPUs. Run C adds about 240 B per grid
  point (40 B x `tr_batch`, doubled with overlap), against roughly 330 B per point for the rest of the code.
- **Correctness:** A, B and C must give identical `log.dat` files. A difference on the cluster is a bug.
