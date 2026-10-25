# Testing batched and overlapped transforms on a cluster

The options `tr_batch` and `tr_overlap` (see PERF.md, section 4b) are bit-identical to the default and as fast
as the default on a single node, where there is no network latency to hide. Their gain has to be measured on
several nodes.

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
