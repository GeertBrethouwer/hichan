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
  9 to 17.
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

- **Use fewer GPUs for small cases.** Memory is roughly 300 bytes per grid point plus about 24 bytes per scalar,
  so even the Re_τ = 395 case (256×193×192, about 3 GB) fits on one 96 GB GH200.
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

## 5. Output and diagnostics

- **`iout_stats`:** every sample computes 11 + 5·nscal statistics profiles, each with a host synchronisation and
  an MPI reduction. Successive steps are strongly correlated anyway, so 50–100 is as useful as 20 and cheaper.
- **`iout_log`:** every log line computes the divergence (an extra derivative pass and global reductions).
  100–500 is enough.
- **`iout_field`:** 3D field output is expensive (one inverse transform plus a large write per field). Use it
  rarely, or use `tools/restart_slices.py` on restart files afterwards.
- **`iout_restart`:** every few hours of wall time. Set `wall_time_max` so the final restart is written before
  the job ends.
- **`mode_log_kx = -1`** unless needed.

## Not in the input file

Compiler flags (`-O3 -fast -gpu=cc90` in `configs/arrhenius.mk`) and binding each rank to its own GPU and Grace
CPU in the job script also matter, and are already set up.
