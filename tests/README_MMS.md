# Verification with manufactured solutions (MMS)

`tests/mms.f90` checks HiCHAN against an exact, three-dimensional, time-dependent solution of the forced equations.
The header of `tests/mms.f90` gives the solution in full. In short:

- **velocity:** two oblique waves, wall-normal velocity (1, 1) and wall-normal vorticity (1, −2) in (kx, kz) mode
  numbers, plus a mean flow in x and z. It is zero at the walls, divergence free and not polynomial in y.
- **pressure:** an oblique mode plus a mean part in y.
- **scalars:** three non-polynomial parts plus a wall profile that meets each scalar's `sbc_bot`/`sbc_top`
  conditions (Dirichlet or Neumann).

The body force F = ∂u/∂t − u×ω + ∇P − ν∇²u (+ Ri s g) and the scalar source Q are evaluated analytically at
every RK substage. HiCHAN adds them through two hooks in `mod_rk`, `extra_forcing` and `extra_scalar_forcing`.
Both are null pointers in production runs, so they cost nothing there.

The 3/2 rule resolves every product of the solution exactly, so all errors come from y and t. Errors are
max |f − f_exact| / max |f_exact| over the spectral coefficients, for all rows, the interior rows or the core
|y| ≤ 0.5.

```bash
make FC=... mms                                     # build/mms, same flags as hichan
cd <empty dir>; tests/mms_run.sh operators 65 0.01   # operator tests at t = 0
tests/mms_run.sh run 129 0.002 re=1000 nscal=2 scal_form=skew scal_theta=0.5 bc1=DN bc2=ND ri=0.3
```

`mms_run.sh TEST NY DT key=value...` writes `input.nml` and runs `build/mms` (`NP=` ranks, `pdims=`). Keys: `re`,
`nl_form`, `visc_theta`, `ns_method`, `fd_order`, `grid_alpha`, `nx`, `nz`, `tend`, `nscal`, `scal_form`,
`scal_theta`, `bc1`/`bc2` (wall conditions of scalars 1 and 2: DD, DN, ND, NN), `ri` (scalar 1; scalar 2 has 0.5).
All runs use `forcing = 'cpg'` with `dpdx = 0` (the forcing contains the mean pressure gradient).

## Results (laptop CPU, tanh grid α = 3, nx = nz = 16)

### Operators at t = 0 (fd_order = 6)

| ny | diffusion (all / core) | nonlinear rot. / adv. | pressure: velocity, φ | divergence |
|---|---|---|---|---|
| 33 | 1.6e-5 / 3.3e-7 | 7.3e-8 / 1.2e-7 | 6.7e-5, 1.6e-5 | 8e-12 |
| 65 | 1.9e-6 / 5.1e-9 | 3.0e-9 / 1.9e-9 | 4.2e-6, 9.5e-7 | 3e-13 |
| 129 | 2.6e-7 / 7.9e-11 | 1.6e-10 / 2.9e-11 | 2.6e-7, 5.7e-8 | 7e-13 |
| 257 | 4.0e-8 / 1.7e-12 | | 1.6e-8 | |
| order | 3 / 6 | 6 / 6 | 4, 4 | round-off |

(Largest of the three components.)

- **Diffusion** (`add_lap`): 6th order in the core, 3rd order at the wall rows (the wall closure of D2).
- **Nonlinear terms:** 6th order in both momentum forms.
- **Pressure step:** 4th order everywhere, the core included. The wall closures of D1 set the accuracy of the
  elliptic problem. The discrete divergence is zero to round-off.
- **fd_order = 4:** diffusion and pressure are 4th order.

### Time-dependent runs, order in space (Re = 1000, dt = 2e-3, t = 0.5; error of v, the largest)

| ny | 6th, rotational | 6th, advective | 4th, rotational | vv | implicit, θ = 0.5 |
|---|---|---|---|---|---|
| 33 | 6.3e-5 | 6.9e-5 | 4.9e-5 | 4.3e-5 | 6.3e-5 |
| 65 | 4.2e-6 | 4.4e-6 | 3.1e-6 | 2.1e-6 (w) | 4.2e-6 |
| 129 | 2.7e-7 | 2.7e-7 | 2.0e-7 | 1.9e-7 (w) | |
| 257 | 1.7e-8 | 1.7e-8 | 1.2e-8 | 2.1e-8 (w) | |

- All are 4th order, set by the pressure (influence-matrix) step and its wall closures. The 6th-order scheme gains
  little over the 4th-order one for the velocity at these resolutions.
- `ns_method = 'vv'`: in the core it is far more accurate (v 1.1e-10, w 1.3e-10 at ny = 257). Near the walls it is
  3rd order (u, w), and its divergence is not round-off: 6e-8 at ny = 257, converging at 3rd order. This is the
  documented effect of `vv_neumann = 'explicit'`.

### Order in time

- **Explicit, Re = 10⁴, ny = 257:** halving dt from 0.02 to 0.01 reduces the error by 7.9 / 7.7 / 8.0 (u / v / w),
  so RK3 is 3rd order.
- **Implicit θ = 0.5 (influence matrix), Re = 100, ny = 129:** the error falls by 4.7, 4.3, 4.2 per halving, so
  2nd order, as expected for the θ-method.
- **vv** (SMR coefficients): about 3rd order down to its spatial floor.

### Scalars

**Operators at t = 0.** The advection term of scalar 1 (ssrc_const = 0.2, ssrc_u = 0.3, so the sources are
included), as the error over all rows / interior rows / core:

| ny | advective | divergence | skew |
|---|---|---|---|
| 33 | 9.5e-8 | 8.9e-5 / 2.4e-5 / 4.8e-6 | 3.9e-4 / 7.0e-5 / 7.5e-6 |
| 65 | 1.5e-9 | 9.5e-6 / 2.2e-6 / 7.7e-8 | 4.9e-5 / 9.2e-6 / 1.1e-7 |
| 129 | 2.4e-11 | 1.1e-6 / 2.4e-7 / 1.2e-9 | 6.0e-6 / 1.2e-6 / 1.7e-9 |
| 257 | | 1.3e-7 / 2.8e-8 / 1.9e-11 | 7.5e-7 / 1.5e-7 / 2.6e-11 |

- All three forms are 6th order in the core.
- The divergence and skew forms are 3rd order near the walls. There the 3rd-order D1 wall closure acts on v s,
  whose y-derivative is not small. In the advective form the factor v ≈ 0 at the walls hides the closure error.
- The skew form (half of it with the mapped derivative `deriv1m`) has about 5× the near-wall error of the
  divergence form.

**Time-dependent runs** (two scalars, Re = 1000, dt = 2e-3, t = 0.5; Pr = 0.71 and 2; buoyancy Ri = 0.3 and 0.5
in the last two cases):

| ny | adv., θ = 0, DD / NN | div., θ = 0.5, DN / ND | skew, θ = 0.5, DD / NN |
|---|---|---|---|
| 33 | 3.8e-6 / 1.2e-5 | 8.8e-6 / 2.1e-5 | 1.1e-5 / 4.8e-5 |
| 65 | 1.3e-7 / 5.8e-7 | 3.4e-7 / 1.2e-6 | 4.0e-7 / 3.5e-6 |
| 129 | 2.6e-9 / 2.7e-8 | 1.4e-8 / 6.4e-8 | 1.3e-8 / 2.4e-7 |
| order | 5 / 4.4 | 4.6 / 4.2 | 5 / 3.9 |

These runs cover all wall conditions (Dirichlet and Neumann, the wall values and the Neumann gradients of the mean
mode), explicit and implicit diffusion, the sources, and buoyancy coupling back into the momentum equations.

**Order in time** (scalar 1, advective, DD, Re = 10⁴, ny = 257, t = 1):

- **θ = 0:** halving dt from 0.01 to 0.005 reduces the error by 7.8, so 3rd order. At dt = 0.02 the run is
  unstable: explicit scalar diffusion with κ = ν/Pr > ν is past its limit there, while the velocity is still
  stable.
- **θ = 0.5:** the error falls by 7.9 and 7.6. At this Péclet number the 3rd-order advection error dominates the
  2nd-order error of the θ-method.

### Parallel

1 rank, `pdims = 1, 2` and `pdims = 2, 1` give the same errors to all printed digits (operators and time-dependent
runs with two scalars).

### Summary

No errors were found:

- every term converges at the order its stencils and wall closures allow;
- the discrete divergence is at round-off with `ns_method = 'im'`;
- the time schemes have their design orders.

The accuracy of the velocity is limited to 4th order by the pressure step, i.e. by the 3rd-order wall closures of
D1. A higher-order wall closure for D1 is the place to improve the 6th-order scheme.

### fd_order = 8

**Operators at t = 0:**

| ny | diffusion all / core | pressure velocity | nonlinear | scalar div. form (all / core) |
|---|---|---|---|---|
| 33 | 1.3e-6 / 1.7e-9 | 9.2e-6 | 1.2e-8 | 3.4e-5 / 2.4e-7 |
| 65 | 5.3e-8 / 6.9e-12 | 2.8e-7 | 2.2e-10 | 1.4e-6 / 1.0e-9 |
| 129 | 2.6e-9 / 1.4e-13 | 8.6e-9 | 5.1e-12 | 6.9e-8 / 3.9e-12 |
| 257 | 1.4e-10 | 2.7e-10 | 1.4e-13 | 3.9e-9 / 2.4e-14 |
| order | 4.3 / 8 | 5 | 8 | 4.3 / 8 |

The divergence is at round-off.

**Time-dependent runs** (Re = 1000, dt = 2e-3, t = 0.5; largest velocity error):

| ny | skew scalars, θ = 0.5, Ri = 0.3 | vv | implicit θ = 0.5, advective |
|---|---|---|---|
| 33 | 1.4e-5 | 8.4e-6 | 1.3e-5 |
| 65 | 5.2e-7 | 1.7e-7 | 4.9e-7 |
| 129 | 1.7e-8 | 2.2e-9 | 1.7e-8 |

- Scalars at ny = 129: DD 1.0e-9 (6th order), NN 4.7e-8 (5th order).
- vv divergence at ny = 129: 7.6e-9 (order 6: 5e-7).
- Orders 4 and 6 give results bitwise identical to before the change.
