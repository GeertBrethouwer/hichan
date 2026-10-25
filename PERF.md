# Performance tuning through the input file

Most of the cost of a time step is in the FFTs and the transposes between processes (and GPUs). After that come
the wall-normal banded solves, and finally statistics and I/O. The settings below are in rough order of impact.
The GPU-related advice is reasoning from the code's structure; it has not been measured on a GPU.

Every run prints `wall-clock time per time step` at the end, so you can compare settings with short runs
(a few hundred steps).

## 1. Fewer transforms per step

- **`nl_form = 'rotational'`** (default): 9 FFT+transpose pairs per RK stage for the velocity, against 15 for
  `'advective'`. That's about 40% fewer transforms, and it is also the more robust form.
  `'divergence'` also needs 9 (3 backward, 6 forward for the products uu…ww).
- **Number of scalars:** each scalar adds 4 transforms per stage. With 2 scalars the transform count rises from
  9 to 17. With `scal_form = 'skew'` each scalar costs 8 transforms (2 scalars: 9 → 25). It is
  worth it when the scalars limit the stability; otherwise keep `'advective'`.
- **`dealias = .true.`:** keep it for DNS. Switching it off makes the physical grid 2.25× smaller, but brings back
  aliasing errors.

## 2. A larger time step

- **`cfl`:** RK3 is stable up to √3 ≈ 1.73. Going from 1.0 to 1.5 means a third fewer steps; around 1.5 still
  leaves some margin. `cfl_visc` (limit 2.51) can go up to about 2.
- **Wall resolution:** with explicit viscous terms the time-step limit scales with dy_wall². Clustering points
  more than needed at the wall costs a lot. Use `build/ygrid ny tanh alpha Re_tau` (`make ygrid`) to choose `ny`
  and `grid_alpha` for dy⁺_wall ≈ 0.5–1 and dy⁺_centre ≲ 5–9. For example, 193 points with α = 4 at Re_τ = 395
  gives 0.62 / 8.5.
- **`visc_theta`:** keep 0 (explicit) unless the viscous limit really sets dt.
  - On the coarse CPU test an implicit step cost about 1.6× an explicit one (6 iterations), so implicit only pays
    off if it allows a dt well over 1.6× larger.
  - That happens with very fine wall grids, low Re, or when the log shows dt well below what `cfl` would allow.
  - If you do use it, `impl_tol = 1e-10` instead of 1e-12 saves 1–2 of the ~6 iterations per substep, at the cost
    of a slightly larger divergence.
- **Scalars:** use `scal_theta = 0.5` for low Prandtl numbers (e.g. 0.025, liquid metals). Otherwise the scalar
  diffusion limit dominates dt. For Pr ≳ 0.7, `scal_theta = 0` (explicit) is cheaper and doesn't reduce dt.
  Dirichlet walls are cheaper than Neumann walls (fewer solves per step).
- **`fixed_dt = .false.`**, with a sensible `dt_max`.

## 3. Grid sizes that suit the FFTs and the parallel split

- The FFT lengths are 3·nx/2 and 3·nz/2. Choose `nx` and `nz` so these contain only factors 2, 3 and 5:
  - good: 128 → 192, 192 → 288, 256 → 384
  - poor: 130 → 195 = 3·5·13
- `nx/2`, `nz` and `ny` should divide evenly by the process grid, so every GPU gets the same amount of work.

## 4. Parallel settings (`&parallel`) and the number of GPUs

- **Use fewer GPUs for small cases.** Memory is about 330 bytes per grid point (see *Memory* below), so even the
  Re_τ = 395 case (256×193×192, about 4 GB) fits on one 96 GB GH200.
  - The wall-normal solves use one GPU thread per Fourier mode pair, (nx/2)·nz in total. With 128×128 modes that
    is only 8192 threads; split over 4 GPUs, each GPU is far from fully used, and the transposes over NVLink add
    cost.
  - Try 1, 2 and 4 GPUs and keep the setting with the best time per step, or the best cost per simulated time.
- **Process grid.** On one node, `pdims = 4, 1` keeps all transposes on the node's GPU links. On several nodes,
  keep the first dimension (`p1`) at 4 or less, so the larger transposes on the dealiased grid stay within a node.
  Use `pdims = 0, 0` first to see what cuDecomp's autotuner chooses.
- **Communication backend.**
  - `cudecomp_enable_nccl = .true.` once NCCL works in the job environment (see the NCCL note in the README);
    NCCL is usually fastest over NVLink.
  - NVSHMEM is only an option if cuDecomp was built with `NVSHMEM=1`.
  - After the first run, set `cudecomp_t_backend` to the backend the autotuner reported (e.g. 4 for NCCL) to skip
    the autotuning at start-up.

- **Transpose library** (`transpose_lib`, see the README). With `timing = .true.` the line
  `transposes (communication)` of the timing report compares cuDecomp, diezDecomp and MPI directly; build with
  `CUDECOMP=1 DIEZDECOMP=1` to have all three in one executable.
- **`transpose_prec = 'single'`** halves the data sent in every transpose; on several nodes the transposes take
  most of the time per step, so this can be the largest single gain. Accuracy: see the README. On one CPU node
  (where the transposes are mostly local memory copies) the gain was only 8%; on GPUs connected by the network,
  measure it with `timing = .true.`.
- **Slab decompositions** (one of `p1`, `p2` equal to 1) halve the communication: one of the two transposes of
  every FFT becomes a local copy. That copy is done without MPI by cuDecomp and diezDecomp, and through
  MPI_Alltoallv on a one-rank communicator by `'mpi'`.
  - `pdims = 1, P` communicates only the z-FFT ↔ wall-normal transposes. These carry the non-dealiased data
    (nz instead of 3nz/2), so they move ⅓ less data than `pdims = P, 1`. It needs P ≤ min(nz, ny).
  - `pdims = P, 1` needs P ≤ nx/2.
  - Diez Sanhueza et al. (CPC 2025) find slabs the fastest choice up to the GPU counts at which they still fit,
    and 2D grids with few partitions in one direction (e.g. 2 × 512) the best beyond that. Measure on your case:
    the inter-node links decide.

## 4b. Batched and overlapped transforms (`tr_batch`, `tr_overlap`; `transpose_lib = 'mpi'`)

On many nodes the transposes are dominated by the number of messages, not only by their size. Each RK stage
does 9 velocity transforms in 18 all-to-alls, and the message per peer shrinks as 1/(number of ranks). Two
options act on the velocity transforms of the nonlinear term, for all three `nl_form`s:

- **`tr_batch = n`**: up to n fields share one transpose message, giving n× fewer collectives with n× larger
  messages.
  - The rotational form transforms u, v, w and ω together (6 fields), then the three products (3 fields).
  - The advective and divergence forms batch u, v, w (3) and, for divergence, the six products.
  - `tr_batch = 3` is the natural choice; 6 batches the whole backward step of the rotational form.
- **`tr_overlap = .true.`**: the groups of fields are pipelined with non-blocking `MPI_Ialltoallv` and two buffer
  sets per transpose stage. Group g+1 is packed and sent while group g is unpacked and FFT'd, and the y→x transpose
  of group g runs while group g−1 gets its x-FFT.
  - Combine it with a `tr_batch` smaller than the number of fields (e.g. 1 or 3), so there are several groups to
    pipeline.
  - Real overlap needs an MPI that progresses non-blocking collectives in the background. Many need a setting
    for that, e.g. `MPICH_ASYNC_PROGRESS=1`, Open MPI with a progress thread, or a hardware offload. Without it
    the transfer advances only inside the `MPI_Test` calls HiCHAN makes between FFTs.
  - On GPUs every pack kernel is followed by `acc wait` before the message is posted, so it still needs
    CUDA-aware MPI.
- **Results** are bit-identical to the default.
  - Checked with bounds checking for the rotational, advective, divergence and `'vv'` paths, double and single
    precision, on 1–4 ranks, `tr_batch` 1–6 with and without overlap: all `log.dat` identical. See the README
    validation table.
- **Memory:** the batch buffers are 40 · `tr_batch` · (2 with overlap, else 1) bytes per grid point (half with
  `transpose_prec = 'single'`), for example:

  | setting | extra memory per grid point |
  |---|---|
  | `tr_batch = 3`, no overlap | 120 B |
  | `tr_batch = 1`, overlap | 80 B |
  | `tr_batch = 3`, overlap | 240 B |

  The rotational form also needs one more physical work array (18 B per point). Compare this with about 330 B
  per point for the whole code: check the memory report at start-up before using it on a full GPU.
- **Measured so far: only on one CPU node** (4 ranks, 96×97×96, laptop, shared-memory MPI). The variants are as
  fast as the default within the noise: 711–724 ms/step against 719–724 for the 4×1 grid, and 701–718 against
  697–709 for the divergence form. There is no network latency to hide there. The gain has to be measured on
  several nodes: run 200 steps each of `tr_batch = 1`, `3` and `3` + `tr_overlap` with `timing = .true.`, and
  compare the `FFTs incl. transposes` line.
- With cuDecomp or diezDecomp the options are accepted, with a note at start-up. The transforms are then done one
  field at a time (the same results; cuDecomp's own NCCL/NVSHMEM backends are often faster on NVLink anyway).

## Further steps for many GPUs/nodes (not implemented)

From the analysis of the dealiasing and communication path, after Diez Sanhueza et al. (CPC 2025):

- **Dealiasing order: already optimal.**
  - The z-padding is applied after the first transpose, and the x-padding after the second. So zeros are never
    sent, and the only padded data that are communicated are the z-physical values in the x↔y transposes
    (nx/2 × 3nz/2 × ny).
  - Per transform the communicated volume is 2.5 units of nx/2·nz·ny complex values, against 4.5 when both
    directions are padded before the first transpose.
  - The order z first, x last is the cheaper one, because the last transform is complex-to-real. With x first
    the padded transpose would carry 3nx/2 × nz complex values, twice as much.
- **Rank placement:** cx1 = mod(rank, p1), so the padded x↔y transposes (comm_row) stay within a node when p1 ≤
  GPUs per node and the ranks are placed node by node (`--ntasks-per-node`, `--map-by`). See section 4.
- **Scalars:** their transforms are not batched yet (4–8 per scalar and stage). With several scalars, batching
  their gradients is the next step (it needs three more physical work arrays per batch).
- **Phase-shift dealiasing instead of the 3/2 rule:** 20% less transpose volume and 2.25× less FFT work. With RK3
  the double-fold (corner) aliases are not cancelled, so this is an accuracy trade-off.
- **P-TDMA** (see the table below).

## Memory

At start-up HiCHAN prints the memory of its large arrays per rank (max over the ranks), before allocating them,
and with OpenACC also the GPU memory actually used after the initialisation. Per Fourier grid point
(N = nx·ny·nz, 3/2 dealiasing), the memory in bytes is:

| arrays | bytes per point |
|---|---|
| spectral fields u, v, w, N, N_old, 2 work arrays (11 × 8 B) | 88 |
| per scalar: s, N, N_old | 24 |
| physical fields on the dealiased grid, 7 × 18 B (`rotational`); 5 × 18 B with `nl_form = 'advective'` | 126 (90) |
| wall-normal solver work arrays | 40 |
| FFT buffers (shared between stages; also the cuFFT work area) | 30 |
| transpose buffers (MPI or diezDecomp; cuDecomp similar, printed at start-up) | 24 (12 with `transpose_prec = 'single'`; cuDecomp single: about the same as double, because of staging arrays) |
| implicit viscous terms (`visc_theta > 0`) / `'vv'` | +24 / +32 |

Explicit, rotational form, one scalar: about 330 bytes per point, plus the CUDA/NCCL/MPI context (1–2 GB per GPU).
Example: 2560 × 385 × 1920 with one scalar: 1.89·10⁹ points × 332 B ≈ 630 GB, i.e. ≈ 79 GB per GPU on 8 GH200s
(96 GB each).

Up to version 7d129bc the four FFT buffers were separate (50 instead of 30 B per point) and every cuFFT plan
allocated its own work area (up to ~6 B per point per plan). That made this example ≈ 84 GB of arrays per GPU
plus cuFFT work areas and context: more than the ≈ 95 GB available.

If a case does not fit:
- use more GPUs (the memory per GPU falls as 1/number of GPUs);
- `nl_form = 'advective'` saves 36 B per point, but needs 15 instead of 9 velocity FFTs per RK stage;
- every extra scalar costs 24 B per point.

## 5. Output and diagnostics

- **Measure first:** `timing = .true.` in `&output` prints where the time goes (FFTs, transposes, solves,
  output; min/max over the ranks shows load imbalance) and writes the time of every step to
  `timing_steps.dat`. `tools/plot_timing.m` plots it and lists spikes, e.g. from restart writes, slow
  nodes or network contention. For production runs switch it off, or set `timing_sync = .false.`.

- **`iout_stats`:** every sample computes 11 + 5·nscal statistics profiles, each with a host synchronisation and
  an MPI reduction. Successive steps are strongly correlated anyway, so 50–100 is as useful as 20 and cheaper.
- **`iout_log`:** every log line computes the divergence (an extra derivative pass and global reductions).
  100–500 is enough.
- **`iout_field`:** 3D field output is expensive (one inverse transform plus a large write per field). Use it
  rarely, or use `tools/restart_slices.py` on restart files afterwards.
- **`iout_restart`:** every few hours of wall time. Set `wall_time_max` so the final restart is written before
  the job ends.
- **`mode_log_kx = -1`** unless needed.

## Options from Diez Sanhueza, Peeters & Costa (CPC 316, 2025) and their status in HiCHAN

| idea in the paper | status in HiCHAN |
|---|---|
| diezDecomp transposes, with local copies instead of MPI | available: `transpose_lib = 'diezdecomp'` |
| keep the number of partitions of the wall-normal direction small (p2 here), or use slabs | `pdims`, see section 4 |
| fewer, larger messages and overlap of communication and computation (the paper's analysis of what dominates at scale) | `tr_batch`, `tr_overlap` (section 4b), `'mpi'` transposes |
| P-TDMA: distributed wall-normal tridiagonal solves (cyclic reduction plus a reduced system), which remove the transposes to wall-normal pencils | not implemented. In HiCHAN the wall-normal operators are compact (tridiagonal A and banded B up to ±3) with influence-matrix wall conditions, which couple both walls and use dense first and last rows of D1. A distributed version needs a banded (SPIKE/block cyclic reduction) solver and global wall couplings. The paper gains 1.5× per step at 1024 GPUs, but for slab grids, the natural choice at moderate GPU counts, these transposes are already local. Worth it only for very large runs |
| splitting the implicit solves into a set-up part and a run-time part | already done in this form: the influence matrices and the implicit 6×6/2×2 corrections are cached per wavenumber (and RK substep), and the banded factorisations are rebuilt on the fly from a few shared coefficient rows. Storing per-column factors would increase memory traffic on the GPU |
| direct x↔z transposes for the implicit 1D diffusion | not applicable: HiCHAN always needs the intermediate z-FFT pencil |

## Not in the input file

Compiler flags (`-O3 -fast -gpu=cc90` in `configs/arrhenius.mk`) and binding each rank to its own GPU and Grace
CPU in the job script also matter, and are already set up.
