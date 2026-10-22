# HiCHAN — high-order DNS of channel flow with passive scalars (CPU and Grace Hopper GPUs)

HiCHAN solves the incompressible Navier–Stokes equations in a plane channel, together with any number
(up to 16) of passive scalars, with the method of

> P. Lenaers, P. Schlatter, G. Brethouwer, A. V. Johansson, *A new high-order method for the simulation of
> incompressible wall-bounded turbulent flows*, J. Comput. Phys. 272 (2014) 108–126.

* **x, z (streamwise, spanwise):** Fourier series, pseudo-spectral nonlinear terms, 3/2-rule dealiasing.
* **y (wall-normal):** collocated compact finite differences on an arbitrary (e.g. tanh-stretched) grid, with
  the (3-4-3) fourth-order or (3-5-6-5-3) sixth-order schemes of the paper. Coefficients are computed directly
  on the non-uniform grid. First and second derivatives are both compact (`A1 f' = B1 f`, `A2 f'' = B2 f`).
* **Pressure:** influence-matrix method. Continuity is satisfied exactly at **every** grid point, including the
  walls, so the discrete divergence is at round-off level (1e-15 to 1e-10, depending on grid and resolution).
* **Time:** low-storage 3-stage RK3 (Table 1 of the paper). The viscous terms are either explicit (Appendix B.1,
  `visc_theta = 0`, the default and the variant the paper recommends for channel flow, Sec. 6) or implicit with
  the θ-method (Appendix B.2, `visc_theta > 0`, e.g. 0.5 = Crank–Nicolson). Scalar diffusion uses the same
  θ-method (`scal_theta`, default 0.5), or explicit RK3 when θ = 0.
* **Scalars:** up to 16 passive or active scalars. Active scalars act on the flow through a Boussinesq
  buoyancy term, with gravity in an arbitrary direction.
* **Parallelisation:** 2D pencil decomposition with MPI. On GPUs it uses OpenACC, cuFFT and NVIDIA **cuDecomp**
  (autotuned process grid and NCCL/NVSHMEM/MPI transposes). That design is taken from
  [CaNS](https://github.com/CaNS-World/CaNS) v4.2.0 (see *What was taken from CaNS* below).

## Contents

```
src/types.f90      kinds
src/param.f90      input (namelists in input.nml)
src/compact.f90    grid, compact-scheme coefficients on non-uniform grids, quadrature
src/decomp.f90     pencil decomposition and transposes (cuDecomp or MPI_Alltoallv)
src/fft.f90        physical x-pencil <-> spectral z-pencil transforms (cuFFT / FFTW), dealiasing
src/ysolve.f90     wall-normal column solvers: derivatives, influence-matrix pressure step, scalar Helmholtz
src/rk.f90         nonlinear terms (advective or rotational form), RK3 step, CFL
src/fields.f90     field storage
src/stats.f90      statistics (Parseval, spectral space), log
src/io.f90         restart files (MPI-IO), 3D fields (MPI-IO + XDMF)
src/initflow.f90   initial conditions
src/main.f90       main program
configs/           machine settings (arrhenius.mk, generic_gpu.mk)
examples/          input_reference.nml (every input item, commented), ts_wave (paper Sec. 5.1, check.py),
                   scalar_check (exact laminar scalar solutions), rayleigh_benard and vertical_channel
                   (active scalars, check.py), retau180, retau395 (paper Sec. 5.3), arrhenius_job.sh
tools/             build_cudecomp.sh, os_mode.py (Orr–Sommerfeld modes), plot_stats.py (wall units),
                   restart_slices.py (2D slices from a restart file -> .mat), plot_slices.m (MATLAB/Octave),
                   mean_tauw.m (time-averaged tau_w, u_tau, Re_tau from log.dat, MATLAB/Octave),
                   ygrid.f90 (y and dy, optionally y+ and dy+, for given ny/grid type/alpha: make ygrid;
                   build/ygrid 193 tanh 4.0 395)
```

## The numerical method in brief

All wall-normal work is done in spectral space on z-pencils `f(kx, kz, y)`. One GPU thread handles one
wavenumber pair and marches sequentially in `y`. Consecutive threads access consecutive `kx`, so memory access
is coalesced. Only *banded* matrices are factorised (LU without pivoting, on the fly). The dense operators
`D1 = A1⁻¹B1` and `D2 = A2⁻¹B2` are never formed, which keeps the cost at O(Ny) per wavenumber pair (Sec. 4).

**Pressure step** (explicit viscous terms, Appendix B.1). With `φ = β_q Δt p` and the RK right-hand side `r`:

```
u = r_x − i kx φ,   w = r_z − i kz φ,   v = r_y − D1 φ        (interior points)
u = v = w = 0 at the walls,     i kx u + i kz w + D1 v = 0     at all points
```

The Poisson operator is factorised as `D1² − k² = (D1 + k)(D1 − k)` (Sec. 4.1, Appendix A). With `ψ = (D1 − k)φ`,
the problem becomes two first-order banded problems:

```
(B1 + k A1) ψ = A1 g      (first row replaced by ψ(−1) = μ)
(B1 − k A1) φ = A1 ψ      (last row replaced by φ(+1) = b)
```

Here `g` is the Poisson right-hand side, including the boundary-error terms σ (Eqs. 21, B.2). Six quantities
cannot be imposed by the banded solves:

* μ and b,
* the two wall values of `g`,
* the two wall corrections σ_b, σ_t of `v` (Eqs. 27–30).

They are found by superposition with six homogeneous solutions. Their 6×6 influence matrix depends only on
`k = |(kx, kz)|`, not on Δt, so it is computed and inverted **once** at start-up (Eqs. 23–25 and B.9–B.22,
merged into a single superposition). Each RK substep then costs two pairs of banded solves per wavenumber pair.
The result solves the discrete system of the paper exactly (up to round-off): `u, w` from the momentum
equations, and continuity at every point.

**Implicit viscous terms** (`visc_theta > 0`, Appendix B.2). With `c = θ β_q Δt/Re` and
`L = (1 + c k²) I − c D2`, the discrete problem is

```
L u + i kx φ = r_x,   L v + D1 φ = r_y,   L w + i kz φ = r_z      (interior points)
u = v = w = 0 at the walls,     i kx u + i kz w + D1 v = 0          at all points
```

Taking the divergence gives the pressure equation (cf. Eq. B.32):

```
(D1² − k²) φ = R + C v + δ_b d1(:,1) + δ_t d1(:,N),     C = c (D1 D2 − D2 D1)
```

Because `D1` and `D2` do not commute, `v` appears on the right-hand side. As in Appendix B.2 this term is
**lagged and iterated** (initial guess `v^q`). Everything else in each iteration is solved exactly by
superposition:

* the factorised Poisson solve (as in the explicit case),
* the A2-premultiplied Helmholtz solves `[(1+ck²)A2 − cB2] v = A2 g` (Eqs. B.29–B.31), with their boundary
  errors,
* the wall corrections δ_b, δ_t (the σ of the paper),
* continuity at the walls.

The 6×6 influence matrix (Eq. B.41) and the 2×2 Helmholtz wall corrections depend only on `k` and `c`. They are
cached for each of the three RK substeps and recomputed only when Δt changes; with adaptive Δt the step is kept
piecewise constant. The iteration stops when `max|v^{m+1} − v^m| < impl_tol·max|v|` or after `impl_maxit`
iterations. Each iteration reduces the error by 1–3 orders of magnitude (cf. Fig. 9 of the paper), and 5–10
iterations reach round-off. At convergence the result is the exact discrete θ-method solution, with the
divergence at round-off. Differences from B.2:

* the Poisson operator is factorised exactly as `(D1+k)(D1−k)`, so the term `k(D1A2 − A2D1)p` of Eq. (B.46)
  does not have to be lagged;
* only `C v` is iterated.

The constant-flow-rate correction uses the implicit response `L⁻¹·1` of the mean flow, so it stays exact.

**Scalars** (θ-method, the same formulation as B.2). HiCHAN solves `[(1 + c k²) A2 − c B2] s = A2 r` with
Dirichlet (`s = g`) or Neumann (`D1 s = g`) walls, where `c = θ β_q Δt/(Re Pr)`. Premultiplying by `A2` couples
the unknown wall values of the operator into the first interior rows. This is the boundary error that the paper
corrects with σ, and it is corrected here by superposition in the same way. No iteration is needed, because there
is no pressure coupling. The explicit part `(1−θ)` and the right-hand side follow Eq. (13).

**Nonlinear terms.** Two forms are available:

* `nl_form = 'rotational'` (default): `u × ω`, with the pressure becoming the total pressure. It needs 9 FFTs per
  RK stage for the velocity. Because `u·(u×ω) = 0` holds exactly at every collocation point, the nonlinear term
  cannot create kinetic energy, so this form is robust.
* `'advective'`: the form used in the paper, 15 FFTs per stage. It is equally accurate on a resolved grid: one
  test showed a truncation-size difference to the rotational form of 3e-5 → 7e-6 for ny = 33 → 129. But with
  collocated compact differences on a stretched grid it is not discretely energy conserving. In a deliberately
  under-resolved test (32×65×32 at Re_τ ≈ 180), it blew up during the transition from strong random noise, while
  the rotational form stayed stable. Restarted from the developed turbulent field, it ran stably. Use it at DNS
  resolution, or start transition with the rotational form.

Each scalar adds 4 FFTs and uses the advective form. The time step comes from the RK3 stability limits: CFL
`cfl` (< √3) and explicit diffusion `cfl_visc` (< 2.51). The spectral radius of `D2` is computed numerically on
the actual grid.

**Active scalars (Boussinesq).** Each scalar can act on the flow through the buoyancy term

```
du/dt + ... = ... - sum_n ri(n) * s_n * e_g ,       e_g = gravity/|gravity|
```

`gravity` is the direction in which gravity acts and may point in any direction: (0,−1,0) for a horizontal
channel, (−1,0,0) for a vertical channel with upward flow, or oblique. `ri(n) = g β ΔT_ref h/U_ref²` is the
Richardson number of scalar n; it is positive for a scalar that makes the fluid lighter, such as temperature, and
negative for one that makes it heavier, such as salinity. `ri(n) = 0` keeps the scalar passive. Between the walls
(gap 2h, scalar difference Δs), the Rayleigh number is `Ra = 8 ri Δs re² pr`.

The term is linear in the scalars, so it is added exactly in spectral space, explicitly with the RK3 weights and
using the scalar at the current stage. Its mean part is taken up by the hydrostatic pressure (wall-normal
component) or drives the mean flow (streamwise and spanwise components; the CFR correction acts on the
streamwise part). The time step also respects the internal-gravity-wave limit `Δt ≤ cfl/N`, with
`N² = |ri| max|dS/dy|` from the mean stratification.

**Forcing.** Constant flow rate (`cfr`, exact correction of the mean flow, which is valid for explicit viscous
terms) or constant pressure gradient (`cpg`).

### Differences from the paper (all deliberate)

* The paper's nested superposition (B.9 and Eq. 23) is merged into one 6×6 influence matrix per wavenumber.
  The banded solves (Sec. 4.1, App. A) are used unchanged. The discrete solution is the same.
* In the explicit scheme the viscous term is advanced with the RK3 weights `γ_q, ζ_q` (3rd order), not with
  `β_q` alone (Eq. B.1). With `β_q` alone the composition of the three sub-steps is only first-order accurate for
  the viscous term.
* With explicit viscous terms, the time step obeys both RK3 limits (`cfl`, `cfl_visc`). With θ ≥ 1/2 (velocity
  or scalars) the diffusive limit is dropped; for 0 < θ < 1/2 it is applied with the effective diffusivity
  (1−2θ)ν.

## Validation (run on CPU in this repository; see `examples/`)

| test | result |
|---|---|
| Laminar Poiseuille, CFR | `dp/dx = 3/Re` exactly, divergence ~1e-15 |
| Decomposition independence | the same results on 1, 2, 3 and 4 ranks with process grids 1×1, 2×1, 1×2, 2×2, 3×1, 1×4 (identical to the printed 7 digits; the divergence differs at the 1e-16 level) |
| TS wave, Re = 5000, α = 1.12 (Sec. 5.1), grid α_s = 4 | the decay rate matches the eigenvalue of the semi-discrete system to 7e-10 (relative 3e-7). Against the Orr–Sommerfeld value (c_i = −2.4847e-3, as in the paper), the error at Ny = 65 is 2.6e-5 (4th order) and 3e-7 (6th order) |
| Scalars, steady laminar solutions | source with Dirichlet walls, Neumann/Dirichlet, Neumann/Neumann, θ = 0 and θ = 0.5/1: errors < 2e-8 (the output precision); wall fluxes exact |
| Constant pressure gradient | laminar U = Re dpdx (1−y²)/2 is kept exactly (Re_τ = 180, U_b = 60) |
| Turbulent channel, Re_b = 2800, 32×65×32 (very coarse), rotational form, 2 scalars | transition from noise, then sustained turbulence for t > 140 h/U_b. Short average (11 h/U_b): Re_τ = 181, centreline U⁺ = 18.2, u_rms⁺ peak 2.4, −uv⁺ peak 0.76. The divergence stays ~1e-14 throughout. This is only a stability check, not a converged DNS |
| Implicit viscous terms (B.2): TS wave, θ = 0.5, Δt = 0.002 | decay rate agrees with the semi-discrete eigenvalue to 6.6e-10, divergence 1e-20, 5 iterations to 2e-15 |
| Implicit, Δt = 0.02 and 0.05 (explicit viscous limit ≈ 0.006) | stable. Error 2.1e-8 → 1.4e-7 for θ = 0.5 (2nd order); 6.4e-5 → 1.7e-4 for θ = 1 (1st order); 8–13 iterations to 1e-13 |
| Implicit, laminar CFR / 1–4 ranks / OpenACC build | −dp/dx = 3/Re exactly; results independent of the decomposition; OpenACC host-fallback build identical |
| Implicit, turbulent channel (restart from the coarse Re_τ = 180 field) | stable, 6 iterations per substep, divergence ~5e-14. On this coarse CPU case an implicit step costs ≈ 1.6× an explicit one; there Δt is set by the advective CFL, so implicit pays off only when the viscous limit binds (fine near-wall spacing, low Re, low Pr) |
| Buoyancy, natural convection in a vertical channel (`examples/vertical_channel`) | gravity along −x, −z and oblique (−0.6, 0, −0.8): steady profile u = ri·re·(y³−y)/6 (projected on gravity) reproduced to 1.6e-6 (the residual start-up transient), v = 0 exactly |
| Buoyancy, onset of Rayleigh–Bénard convection (`examples/rayleigh_benard`) | growth rate changes sign at **Ra_c = 1707.78** (linear theory 1707.76), at k d = 3.117 |
| Buoyancy, 3D with oblique gravity, two active scalars | same results on 1, 3 and 4 ranks and with the OpenACC build |
| Restart | 20 steps straight vs. 10 + restart + 10 (restarting on 3 instead of 2 ranks): bit-identical |
| OpenACC code path | built with `gfortran -fopenacc` (host fallback), with cuFFT emulated by FFTW using the same layout parameters: results identical to the CPU build |

## Building

### CPU (any MPI + FFTW3)

```
make                     # or: make FC=mpiifx FFTW_INC=... FFTW_LIB="-L... -lfftw3"
```

### NVIDIA GPUs, e.g. NSC Arrhenius (GH200)

```
module load GPU/buildenv-nvhpc/25.9-cu13.0   # NVIDIA HPC SDK + CUDA-aware MPI; see `module avail buildenv`
./tools/build_cudecomp.sh                     # clones and builds cuDecomp (same revision as CaNS v4.2.0)
make CONFIG=configs/arrhenius.mk              # -acc -cuda -gpu=cc90, cuFFT, cuDecomp
```

`make GPU=1` without `CUDECOMP=1` uses the portable MPI_Alltoallv transposes on device buffers. This requires a
CUDA-aware MPI. Use it as a fallback if cuDecomp is not available.

**Launching:** a correct start prints `number of MPI processes: N` with N equal to the number of Slurm tasks,
and the input namelist only once. If the log says `No PMIx server was reachable ... singletons will be started`,
the launcher does not match the MPI library: use `srun --mpi=pmix` or the `mpirun` of the MPI that HiCHAN was
built with (HiCHAN then also prints a warning).

**NCCL:** if the log shows `NET/OFI Couldn't create domain ... Function not implemented` and `Failed to
initialize any NET plugin` (then `cudecomp.cc:69 NCCL error (error code 5)`), NCCL's libfabric network plugin
does not work in the job environment. On one node set `NCCL_NET_PLUGIN=none` and `NCCL_NET=Socket`
(the GPUs communicate over NVLink); otherwise set `cudecomp_enable_nccl = .false.` or fix the plugin
configuration (see `examples/arrhenius_job.sh`).

Run time: `export LD_LIBRARY_PATH=$PWD/dependencies/cuDecomp/build/lib:$LD_LIBRARY_PATH`, one MPI rank per GPU
(4 per Arrhenius node). The process grid and the communication backend are autotuned by cuDecomp when
`pdims = 0,0` and `cudecomp_t_backend = 0`. See `examples/arrhenius_job.sh`.

**Note:** the GPU build could not be compiled or run where this code was written (no NVIDIA compiler or GPU was
available). The OpenACC directives were checked with `gfortran -fopenacc`, and the cuFFT branch was checked by
emulation. Build on Arrhenius and run `examples/ts_wave` and `examples/scalar_check` first: they finish in
seconds and check the GPU results against known values.

## Input (`input.nml`)

Performance tuning through the input file is described in [PERF.md](PERF.md).


**`examples/input_reference.nml`** lists every input item with its meaning, units and default value. Copy it as
the starting point for a new case. The input consists of Fortran namelists, one group per topic. `!` starts a
comment, items that are not given keep their defaults, and per-scalar items are written per index
(`pr(2) = 0.71`). At start-up HiCHAN prints a summary of the set-up.

| namelist | contents |
|---|---|
| `&grid` | `nx, nz` (Fourier modes, even), `ny` (points incl. walls), `lx, lz`, `grid_type` (`tanh`/`uniform`/`file`), `grid_alpha` (α_s of Eq. 41), `grid_file`, `fd_order` (4 or 6), `dealias` |
| `&physics` | `re` (U h/ν), `forcing` (`cfr`/`cpg`), `ubulk`, `dpdx`, `nl_form` (`rotational`/`advective`), `visc_theta` (0 = explicit, B.1; > 0 = implicit, B.2), `impl_maxit`, `impl_tol` |
| `&time` | `nstep`, `tmax`, `wall_time_max` (s), `fixed_dt`, `dt`, `dt_max`, `cfl`, `cfl_visc` |
| `&init` | `init_type` (`laminar`/`noise`/`mode`/`restart`), `init_amp`, `init_seed`, `mode_file`, `mode_kx`, `mode_kz`, `restart_file`, `reset_time` |
| `&output` | `data_dir`, `iout_log`, `iout_stats`, `iout_save_stats`, `iout_restart`, `iout_field`, `tstats_start`, `mode_log_kx`, `mode_log_kz` |
| `&scalars` | `nscal`; per scalar `pr(n)`, `scal_theta(n)`, `sbc_bot(n)`/`sbc_top(n)` (`D`/`N`), `sval_bot(n)`/`sval_top(n)`, `ssrc_const(n)`, `ssrc_u(n)`, `sinit(n)` |
| `&buoyancy` | `gravity` (direction, any orientation), `ri(n)` (Richardson number per scalar, 0 = passive) |
| `&parallel` | `pdims`, `cudecomp_t_backend`, `cudecomp_enable_nccl`, `cudecomp_enable_nvshmem` |

## Output (in `data_dir`)

* `log.dat`: step, t, Δt, bulk velocity, applied −dp/dx, τ_w, Re_τ, max |div u|, fluctuation energy, energy of
  one Fourier mode (`mode_log_kx, mode_log_kz`), and the wall gradients of each scalar.
  Time average of τ_w from a given time on (MATLAB/Octave):
  `r = mean_tauw('data/log.dat', 300, 'plot', true)` (time-weighted; handles restarts; error by batch means).
* `stats.dat`: time-averaged profiles, computed exactly in spectral space: U, V, W, ⟨u′u′⟩ … ⟨v′w′⟩, P, ⟨p′p′⟩,
  and per scalar S, ⟨s′s′⟩, ⟨u′s′⟩, ⟨v′s′⟩, ⟨w′s′⟩. `tools/plot_stats.py` converts them to wall units.
* `restart.bin` / `restart.hdr`: spectral fields (global `(nx/2, nz, ny)` complex arrays), independent of the
  number of ranks.
* `fld_NNNNNNN_*.bin` + `.xmf`: physical fields on the dealiased grid, readable with ParaView/VisIt.

## 2D slices for MATLAB

`tools/restart_slices.py` reads `restart.hdr`/`restart.bin` and evaluates the Fourier series exactly, optionally
on a finer x–z grid. It writes xz-, xy- and yz-slices of u, v, w and all scalars to a MATLAB `.mat` file. It
reads the restart file through a memory map, one y-plane at a time, so large restart files are not loaded into
memory at once. `tools/plot_slices.m` plots them; it runs in MATLAB and GNU Octave.

```
python3 tools/restart_slices.py data/restart --yplus 15 --re 2800 --utau 0.0647 --z 1.0 --x 3.0 \
        --upsample 2 --fluct -o slices.mat          # or --y -0.9 0 (outer units); needs numpy, scipy
```
```matlab
addpath tools
plot_slices('slices.mat')                         % all fields
plot_slices('slices.mat', {'u','s1'}, 'save', true, 'clim', [-0.3 0.3])
S = load('slices.mat');   % x, y, z, xz_u (Nz x Nx x n_y), xy_u (ny x Nx), yz_u (ny x Nz), ...
```

With `--upsample 1.5` the slices coincide with the code's own 3D field output (dealiased grid); the two agree to
round-off (≤ 1e-15).

## What was taken from CaNS (and CaNS-EIGEN)

* GPU device binding per node-local rank, OpenACC data management with persistent device arrays, and all kernels
  on one async queue whose CUDA stream is shared with cuFFT and cuDecomp, so no host synchronisation is needed
  inside a time step.
* The cuDecomp set-up: autotuning of the process grid and of the transpose backend, axis-contiguous pencils
  (`transpose_axis_contiguous = [T,T,F]`), workspace from `cudecompMalloc` mapped with `acc_map_data`, and two
  grid descriptors with a shared process grid (here: the dealiased and non-dealiased grids).
* Batched cuFFT plans through the advanced data layout. The Makefile flags and the cuDecomp build follow CaNS.
* One thread per Fourier column with sequential wall-normal sweeps. This is CaNS' `gaussel_gpu` pattern,
  extended here to banded compact-scheme systems.
* CaNS-EIGEN's GEMM-based eigen-projection solvers target second-order FD Poisson problems and were not adopted.
  The compact collocated D1² operator with influence-matrix wall conditions is strongly non-normal, so its
  numerical diagonalisation is less accurate than the O(Ny) banded factorisation of Sec. 4.1, which is cheap on
  GPUs. Only the CaNS infrastructure it shares (cuDecomp/cuFFT/OpenACC) is used.
