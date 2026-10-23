# Performance tuning through the input file

Most of the cost of a time step is in the FFTs and the transposes between processes (and GPUs). After that come
the wall-normal banded solves, and finally statistics and I/O. The settings below are in rough order of impact.
The GPU-related advice is reasoning from the code's structure; it has not been measured on a GPU.

Every run prints `wall-clock time per time step` at the end, so you can compare settings with short runs
(a few hundred steps).

## 1. Fewer transforms per step

- **`nl_form = 'rotational'`** (default): 9 FFT+transpose pairs per RK stage for the velocity, against 15 for
  `'advective'`. That's about 40% fewer transforms, and it is also the more robust form.
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
- **Slab decompositions** (one of `p1`, `p2` equal to 1) halve the communication: one of the two transposes of
  every FFT becomes a local copy. That copy is done without MPI by cuDecomp and diezDecomp, and through
  MPI_Alltoallv on a one-rank communicator by `'mpi'`.
  - `pdims = 1, P` communicates only the z-FFT ↔ wall-normal transposes. These carry the non-dealiased data
    (nz instead of 3nz/2), so they move ⅓ less data than `pdims = P, 1`. It needs P ≤ min(nz, ny).
  - `pdims = P, 1` needs P ≤ nx/2.
  - Diez Sanhueza et al. (CPC 2025) find slabs the fastest choice up to the GPU counts at which they still fit,
    and 2D grids with few partitions in one direction (e.g. 2 × 512) the best beyond that. Measure on your case:
    the inter-node links decide.

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
| transpose buffers (MPI or diezDecomp; cuDecomp similar, printed at start-up) | 24 |
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
| P-TDMA: distributed wall-normal tridiagonal solves (cyclic reduction plus a reduced system), which remove the transposes to wall-normal pencils | not implemented. In HiCHAN the wall-normal operators are compact (tridiagonal A and banded B up to ±3) with influence-matrix wall conditions, which couple both walls and use dense first and last rows of D1. A distributed version needs a banded (SPIKE/block cyclic reduction) solver and global wall couplings. The paper gains 1.5× per step at 1024 GPUs, but for slab grids, the natural choice at moderate GPU counts, these transposes are already local. Worth it only for very large runs |
| splitting the implicit solves into a set-up part and a run-time part | already done in this form: the influence matrices and the implicit 6×6/2×2 corrections are cached per wavenumber (and RK substep), and the banded factorisations are rebuilt on the fly from a few shared coefficient rows. Storing per-column factors would increase memory traffic on the GPU |
| direct x↔z transposes for the implicit 1D diffusion | not applicable: HiCHAN always needs the intermediate z-FFT pencil |

## Not in the input file

Compiler flags (`-O3 -fast -gpu=cc90` in `configs/arrhenius.mk`) and binding each rank to its own GPU and Grace
CPU in the job script also matter, and are already set up.
