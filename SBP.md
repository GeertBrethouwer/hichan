# The SBP 4-8 prototype on Arrhenius

The branch `sbp48-prototype` adds a summation-by-parts (SBP) scheme for the wall-normal derivatives. It is the
diagonal-norm SBP 4-8 operator: 8th order in the interior, 4th order at the walls. It is used for both the
velocity and the scalars. The method, its cost and the CPU tests are in README.md, section *SBP 4-8 option*.
The GPU build compiles with `gfortran -fopenacc`, but it has not yet been run on a GPU.

## 1. Get the code

In a new directory:

```bash
git clone -b sbp48-prototype https://github.com/GeertBrethouwer/hichan.git hichan-sbp
cd hichan-sbp
```

Or in an existing clone:

```bash
git fetch origin
git checkout sbp48-prototype
```

## 2. Compile

```bash
module load GPU/buildenv-nvhpc/25.9-cu13.0
./tools/build_cudecomp.sh                 # only once per clone (skip if dependencies/cuDecomp exists)
make clean
make CONFIG=configs/arrhenius.mk SBP=1
```

- **`SBP=1`** widens the band storage of the wall-normal solver from 3 to 15 diagonals. Without it, a run with
  `y_scheme = 'sbp48'` stops with an error message.
- **`make clean`** is needed whenever you switch between `SBP=1` and a normal build. `make` does not see the
  changed flag and would otherwise mix old and new object files.
- **To keep both executables:** copy `build/hichan` to e.g. `hichan_sbp` after the SBP build, and
  `hichan_default` after a normal build.

The SBP build also runs the normal compact scheme (`y_scheme = 'compact'`), with results bit-identical to the
default build. It is then about 1.3× slower and needs about 30% more memory, so use the normal build for that.

## 3. Use

In `&grid` of `input.nml`:

```
&grid
  ...
  y_scheme = 'sbp48'
/
```

Restrictions:

- `ny >= 25`;
- not with `ns_method = 'vv'`;
- `fd_order` is ignored.

All other options work as before: `nl_form`, `scal_form`, implicit viscous terms, buoyancy, the transpose
libraries and restarts. A restart file can be read by either scheme.

Recommended settings:

- **Momentum:** `nl_form = 'rotational'` (the default). With SBP it conserves the total kinetic energy without
  viscosity. The advective form has no such property and diverged in the tests with both schemes.
- **Scalars:** `scal_form = 'skew'`. With SBP it conserves ∫s² without diffusion. In the vertical-convection test
  (Re 400, Pr 0.7, Ri 1) the advective scalar form diverged with both schemes on a strongly stretched grid; the
  skew form was stable with both.
- **Resolution:** the wall rows are 4th order, and on a stretched grid the scheme is exact for polynomials in the
  grid coordinate, not in y. Use at least as many points in y as with the compact scheme.

The memory report at startup includes the wider band, about +96 B per grid point. Check it before a large run.

## 4. First test on the GPU

Run the two quick tests with the SBP scheme, as with `examples/arrhenius_tests.sh` (see TEST.md), after adding
`y_scheme = 'sbp48'` to both inputs. For example, by hand on one GPU:

```bash
cp -r examples/ts_wave examples/scalar_check $RUN/
sed -i "s/fd_order = 6/fd_order = 6, y_scheme = 'sbp48'/" $RUN/ts_wave/input.nml $RUN/scalar_check/input.nml
```

Then generate `mode.in` and run both cases as in `arrhenius_tests.sh`, with the SBP executable. Expected values
(CPU):

| test | expected with `sbp48` | default scheme, for comparison |
|---|---|---|
| `ts_wave` (`check.py`) | rel. error 9.3e-6 | 3.9e-5 |
| `scalar_check` (`check.py`) | 5, −5 \| 5, −5 \| 0.5, 0.5 \| 0.5, 0.5 \| 0.2, 0.2, error 0 | the same |

A clearly different value on the GPU is a bug in the GPU path. Please send the output.

## 5. Cost

These figures are from one CPU core; GPU timings are still to be measured:

- **Time:** a step costs about 2.5× the default (velocity solve 3×, scalar solves 5×). The banded solves run over
  the maximum band width (15) in every row, although the interior needs only ±4 (D1) and ±8 (D2). A
  row-dependent band would recover most of this; it is not yet implemented.
- **Memory:** about +30% (+96 B per grid point).

Compare `timing_report.txt` of the same case with the default build, `y_scheme = 'sbp48'` and
`timing = .true.`.
