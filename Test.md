# Testing the new HiCHAN code on Arrhenius

This covers the two open pull requests on `GeertBrethouwer/hichan`:

- **PR 1** (`fix/gpu-race-restart-stats`): GPU race in the flow-rate correction, restartable statistics, safe restart writes.
- **PR 2** (`perf/scalar-solve-fft2d`): fewer banded solves per scalar, optional batched 2D FFT (`fft_2d`), one FFT less for `scal_form = 'divergence'`. This branch contains PR 1.

None of this has been compiled or run on a GPU. Everything below marked "expected" is what the code should do, not something measured on Arrhenius.

The module, account and partition names are the ones already used in `examples/arrhenius_job.sh`; adjust them if yours differ.

## 1. Get and build the code

Keep the old and the new code in separate directories so both executables exist side by side.

```bash
cd $HOME
git clone -b main https://github.com/GeertBrethouwer/hichan.git hichan_main
git clone -b perf/scalar-solve-fft2d https://github.com/GeertBrethouwer/hichan.git hichan_new
```

If you already have a checkout with cuDecomp built, reuse that library instead of building it twice. Below, `CUDECOMP_DIR` points at an existing build; if you have none, run `./tools/build_cudecomp.sh` once in `hichan_main`.

```bash
module load GPU/buildenv-nvhpc/25.9-cu13.0
export CUDECOMP_DIR=$HOME/hichan_main/dependencies/cuDecomp/build

cd $HOME/hichan_main && make clean && make CONFIG=configs/arrhenius.mk CUDECOMP_DIR=$CUDECOMP_DIR
cd $HOME/hichan_new  && make clean && make CONFIG=configs/arrhenius.mk CUDECOMP_DIR=$CUDECOMP_DIR
```

`configs/arrhenius.mk` gives these settings; use the same for both builds:

| setting | value |
|---|---|
| compiler | `mpifort` (nvfortran of the NVIDIA HPC SDK) |
| compile flags | `-O3 -fast -tp=neoverse-v2 -cpp -acc -cuda -gpu=cc90 -Minfo=accel` |
| defines | `-D_USE_CUDECOMP` |
| libraries | `-cudalib=cufft -lcudecomp -lcudecomp_fort` |

Optional variants:

- `DIEZDECOMP=1` (after `./tools/get_diezdecomp.sh`) adds the diezDecomp transposes to the same executable.
- `CUFFT_AUTO_WORK=1` lets every cuFFT plan allocate its own work area. Use it only if the 2D plan fails with the shared work area (see section 2.2).

In every job script set

```bash
export LD_LIBRARY_PATH=$CUDECOMP_DIR/lib:$LD_LIBRARY_PATH
```

Things to look for in the compiler output of `hichan_new`:

- `ysolve.f90`: the two `serial` regions in `cfr_correct_implicit` and `vv_cfr` compile without error (they now carry `async(1)`).
- `fft.f90`: the rank-2 `cufftMakePlanMany` calls compile. These lines have never been through nvfortran.

## 2. Validate

Run these on one node first. Each test says which executable to use and what to compare.

### 2.1 Tollmien–Schlichting wave (seconds, 1 GPU)

```bash
cd $HOME/hichan_new/examples/ts_wave
python3 ../../tools/os_mode.py --re 5000 --alpha 1.12 --ny 65 --grid tanh --galpha 4.0 -o mode.in
srun --mpi=pmix -n 1 $HOME/hichan_new/build/hichan
python3 check.py
```

Expected: `rel. error` of about 1e-4 (the README value for `'im'`, 6th order, Ny = 65), and the same digits as `hichan_main` on the same input.

### 2.2 The 2D FFT path (minutes, 1 GPU, then 4 GPUs)

One rank always has `p1 = 1`, so `fft_2d` can be tested on a single GPU. Use `examples/retau180` with a short run and compare the two paths.

Changes to `examples/retau180/input.nml` for both runs:

```
&time
  nstep = 200
/
&output
  iout_log = 20, iout_stats = 0, iout_save_stats = 0, iout_restart = 0
/
```

Run A with `fft_2d = .false.` and run B with `fft_2d = .true.` in `&parallel`, each in its own directory.

Expected:

- Run B prints `FFTs: batched 2D transforms in x and z (fft_2d, slabs in y)` at start-up.
- `log.dat` of A and B agree to all or nearly all printed digits over 200 steps, and `div_max` stays at round-off (about 1e-14) in both.
- If the 2D plan fails, the run stops with `cuFFT plan (2D, forward)` or `(2D, backward)`. Rebuild with `CUFFT_AUTO_WORK=1` and try again; tell me the result either way.

Then repeat A and B on one node with 4 GPUs and `pdims = 1, 4`. The same agreement is expected. With `pdims = 4, 1` or `2, 2` and `fft_2d = .true.` the code prints `fft_2d needs p1 = 1` and uses the 1D transforms.

### 2.3 Scalar solves (10–30 minutes, 1 GPU)

```bash
cd $HOME/hichan_new/examples/scalar_check
srun --mpi=pmix -n 1 $HOME/hichan_new/build/hichan
tail -1 data/log.dat
```

Expected: the last ten columns of `log.dat` are the wall gradients

```
5 -5 | 5 -5 | 0.5 0.5 | 0.5 0.5 | 0.2 0.2
```

to about 1e-6 or better. (The comment in the input file says `-5, 5`; the sign there is wrong, the code is right: S = 2.5 (1 − y²) has dS/dy = +5 at y = −1.) This covers Dirichlet/Dirichlet, Neumann/Dirichlet and Neumann/Neumann walls with θ = 0, 0.5 and 1.

### 2.4 The race fix (minutes, 1 GPU)

The race only occurs on a GPU with `forcing = 'cfr'` and either `ns_method = 'vv'` or `visc_theta > 0`. Take the short `retau180` input of 2.2 and set

```
&physics
  ns_method = 'vv'
/
```

Run it twice with `hichan_new`, in two directories.

Expected with `hichan_new`: the two `log.dat` files are identical, and the `ubulk` column is `1.000000E+00` on every line.

For comparison run it twice with `hichan_main`. If the race shows, `ubulk` deviates from 1 or the two runs differ. It may also not show: a race does not have to trigger. Identical, correct runs with `hichan_new` are the pass criterion.

Repeat once with `ns_method = 'im'`, `visc_theta = 0.5`.

### 2.5 Restart and statistics (minutes, 1 GPU)

Use the short `retau180` input with

```
&output
  iout_log = 20, iout_stats = 1, iout_save_stats = 100, iout_restart = 100, tstats_start = 0.
/
```

1. Run 200 steps from `init_type = 'noise'` in directory `straight`.
2. In directory `chain`: run with `nstep = 100`; then set `init_type = 'restart'`, `restart_file = 'data/restart'`, `nstep = 200` and run again.

Expected:

- `chain/data` contains `restart.bin`, `restart.hdr` and `restart.stats`, and no `.tmp` files.
- The second `chain` run prints `statistics: continuing the average of data/restart.stats, samples = 100`.
- The first line of `chain/data/stats.dat` says `samples = 200`, and the profiles agree with `straight/data/stats.dat`.

## 3. Performance test: 2560 × 385 × 1920, one scalar, 16 GPUs, 200 steps

Use your existing test case and job script (4 nodes, 4 ranks per node, one GPU per rank) and change only the items below. Start every run from the same restart file or the same initial condition.

### 3.1 Settings common to all runs

```
&time
  nstep = 200          ! as in your test; if you restart, nstep counts from the step in the restart file
/
&output
  timing = .true.
  timing_sync = .true.
  iout_restart = 0     ! no restart write inside the timed loop
/
```

Keep `iout_log`, `iout_stats` and all physics, grid and scalar settings as they are in your test, and identical in all runs.

`timing_sync = .true.` makes the split between FFTs, transposes and solves exact, at a small cost. For the final number of the best configuration, run it once more with `timing_sync = .false.`.

After the first run, set `cudecomp_t_backend` in `&parallel` to the backend that cuDecomp reports at start-up, so every later run uses the same backend and skips the autotuning.

### 3.2 The runs

| run | executable | `&parallel` | what it measures |
|---|---|---|---|
| P0 | `hichan_main` | as in your test | baseline |
| P1 | `hichan_new` | as in your test | scalar-solve change alone |
| P2 | `hichan_new` | `pdims = 1, 16`, `fft_2d = .false.` | slab decomposition, 1D FFTs |
| P3 | `hichan_new` | `pdims = 1, 16`, `fft_2d = .true.` | slab decomposition, 2D FFT |
| P4 | `hichan_new` | P3 plus `transpose_prec = 'single'` | half the communication volume |

If your test already uses `pdims = 1, 16`, P1 and P2 are the same run.

Optional: P5 = the faster of P2/P3 with `scal_theta(1) = 0.0`. This removes the scalar banded solves entirely, but it changes the time integration of the scalar and adds the explicit diffusion limit to the time step, so it is only a fair comparison if `dt` stays the same (check the `dt` column of `log.dat`).

### 3.3 What to read

At the end of each run:

- stdout: `wall-clock time per time step` and `per time step and grid point`.
- `data/timing_report.txt`: the lines `FFTs incl. transposes`, `transposes (communication)`, `velocity solve`, `scalar solves`, each with ms/step.
- `data/timing_steps.dat`: the first steps can be slower; `tools/plot_timing.m` shows that.

"FFT compute" below means `FFTs incl. transposes` minus `transposes`.

## 4. Expected performance

No timing of HiCHAN on Arrhenius exists in the repository, so there are two kinds of estimate here: changes relative to your own baseline P0, which are fairly safe, and absolute times, which are rough.

### 4.1 Relative to your baseline

| change | expected effect |
|---|---|
| P1 vs P0 | `scalar solves` line about 45–50 % lower; everything else unchanged. Total time per step probably 0–2 % lower, because this line is a small part of the step. |
| P2 vs P1 (if P1 is a 4 × 4 grid) | `transposes` lower: the x↔z transposes (60 % of the data volume) no longer leave the GPU. Total perhaps 10–25 % lower, depending on how fast the links inside a node are compared with those between nodes. |
| P3 vs P2 | FFT compute about 20–35 % lower and the local transposes disappear from the `transposes` line. Total perhaps 5–10 % lower. This is the least certain number: it has not been measured anywhere. |
| P4 vs P3 | `transposes` about 40–50 % lower. Total perhaps 20–35 % lower. |

The code changes by themselves (P1, and P3 against P2) are worth a few percent to about 10 %. The larger gains come from the input settings (slabs, single-precision transposes), which already existed.

If P3 is not faster than P2, keep `fft_2d = .false.`; the option costs nothing when it is off.

### 4.2 Absolute estimate

Assumptions:

- 16 GH200, one rank each; rotational form (9 transforms per RK stage) plus one scalar in advective form (4): 13 per stage, 39 per step.
- Data per transform: 22.7 GB (x↔z, dealiased grid) and 15.1 GB (z↔wall-normal) in double precision. Per step and per GPU that is 55 GB and 37 GB.
- Effective transpose throughput between nodes, including packing and unpacking: 5–15 GB/s per GPU. This is the main unknown; it depends on whether NCCL or CUDA-aware MPI is used and on how they perform on the Arrhenius network.
- FFT kernels and copies: 1.0–2.5 s per step per GPU. Wall-normal solves and the rest: 0.6–1.3 s.

| run | communication | computation | time per step | 200 steps |
|---|---|---|---|---|
| P0/P1, 4 × 4, double | 4–10 s | 1.6–3.8 s | 6–14 s | 20–45 min |
| P2, 1 × 16, double, 1D FFT | 3–8 s | 1.6–3.8 s | 5–11 s | 15–40 min |
| P3, 1 × 16, double, 2D FFT | 2.5–7.5 s | 1.3–3.2 s | 4.5–10 s | 15–35 min |
| P4, P3 with single precision | 1.2–3.7 s | 1.4–3.3 s | 3–7 s | 10–25 min |

In the code's own unit (`per time step and grid point`, s × processes, with 1.89·10⁹ points) these correspond to roughly 5·10⁻⁸ – 1.2·10⁻⁷ for P0 and 2.5·10⁻⁸ – 6·10⁻⁸ for P4.

Treat the absolute numbers as correct to within a factor of two at best. If your measured P0 is outside the range, trust the measurement and apply the relative changes of 4.1 to it. If the `transposes` line is more than about 70 % of the step, the network is the limit and only P2 and P4 will make a visible difference.

Memory: about 630 GB of arrays in total, so roughly 40 GB per GPU on 16 GPUs; all runs above fit in 96 GB.

## 5. What to send back

For each of P0–P4: the `&parallel` namelist, the cuDecomp backend line from stdout, `timing_report.txt`, and the last line of `log.dat`. With those the estimates above can be replaced by measurements and the default of `fft_2d` decided.
