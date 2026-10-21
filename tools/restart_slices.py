#!/usr/bin/env python3
"""
Extract 2D slices of u, v, w and the scalars from a HiCHAN restart file and save them in a
MATLAB .mat file (read with load(), or with tools/plot_slices.m).

The restart file holds the Fourier coefficients f^(kx,kz,y) on the z-pencil layout
(global complex arrays (nx/2, nz, ny), Fortran order, fields u, v, w, s1, ..., s_nscal),
    f(x,z,y) = sum_{kx,kz} f^(kx,kz,y) exp(i (kx x + kz z)),  kx >= 0 (Hermitian symmetry).
The physical fields are evaluated exactly (inverse FFT, optionally on a finer x-z grid by
zero padding) at the collocation points y_j of the wall-normal grid.

Slices:
    xz-planes at given wall-normal positions y (nearest grid points) -> arrays (Nz, Nx, n_y)
    xy-plane  at spanwise position z                                  -> array  (ny, Nx)
    yz-plane  at streamwise position x                                -> array  (ny, Nz)
The x (z) positions are rounded to the nearest point of the output grid.

Examples:
    python3 restart_slices.py data/restart                      # defaults: y = -0.9, 0; z = 0; x = 0
    python3 restart_slices.py data/restart --y -0.98 -0.8 0 --z 1.0 --x 2.0 --upsample 2 --fluct
    python3 restart_slices.py data/restart --yplus 15 --re 2800 --utau 0.0647 -o slices.mat

Requires numpy and scipy (scipy.io.savemat writes MATLAB v5 .mat files).
"""
import argparse
import os
import sys
import numpy as np


def read_header(base):
    with open(base + '.hdr') as f:
        lines = [l for l in f.read().splitlines() if l.strip() and not l.lstrip().startswith('#')]
    nx, ny, nz, nscal = (int(t) for t in lines[0].split()[:4])
    lx, lz = (float(t.replace('D', 'E')) for t in lines[1].split()[:2])
    t = lines[2].split()
    time, step, dt = float(t[0].replace('D', 'E')), int(t[1]), float(t[2].replace('D', 'E'))
    g = lines[3].split()
    grid_type, grid_alpha, fd_order = g[0], float(g[1].replace('D', 'E')), int(g[2])
    return dict(nx=nx, ny=ny, nz=nz, nscal=nscal, lx=lx, lz=lz, time=time, step=step, dt=dt,
                grid_type=grid_type, grid_alpha=grid_alpha, fd_order=fd_order)


def wall_normal_grid(h, grid_file=None):
    """same formula as src/compact.f90 (init_grid)"""
    ny = h['ny']
    if h['grid_type'] == 'tanh':
        xi = np.arange(ny) / (ny - 1.0)
        y = np.tanh(h['grid_alpha'] * (xi - 0.5)) / np.tanh(0.5 * h['grid_alpha'])
    elif h['grid_type'] == 'uniform':
        y = -1.0 + 2.0 * np.arange(ny) / (ny - 1.0)
    else:
        if grid_file is None:
            sys.exit("grid_type = 'file': give the grid file with --grid-file")
        y = np.loadtxt(grid_file).ravel()[:ny]
    y[0], y[-1] = -1.0, 1.0
    return y


def even(n):
    n = int(round(n))
    return n + (n % 2)


class Field:
    """memory-mapped access to one field of the restart file"""

    def __init__(self, base, h, ifield):
        nxh, nz, ny = h['nx'] // 2, h['nz'], h['ny']
        size = nxh * nz * ny
        nbytes = os.path.getsize(base + '.bin')
        nfields = 3 + h['nscal']
        if nbytes < nfields * size * 16:
            sys.exit('restart.bin is smaller than expected from restart.hdr (%d < %d bytes)'
                     % (nbytes, nfields * size * 16))
        self.data = np.memmap(base + '.bin', dtype='<c16', mode='r', offset=ifield * size * 16,
                              shape=(nxh, nz, ny), order='F')

    def plane(self, j):
        return np.asarray(self.data[:, :, j])          # (nx/2, nz) complex


def to_physical(fh, nx, nz, Nx, Nz):
    """f^(kx, kz) (kx = 0..nx/2-1, kz in FFT order with Nyquist = 0) -> f(x, z), shape (Nz, Nx)"""
    nxh = nx // 2
    # spanwise: place kz = 0..nz/2-1 and -nz/2+1..-1 into an Nz-point FFT array
    g = np.zeros((nxh, Nz), dtype=complex)
    g[:, :nz // 2] = fh[:, :nz // 2]
    g[:, Nz - (nz // 2 - 1):] = fh[:, nz // 2 + 1:]
    g = np.fft.ifft(g, axis=1) * Nz                       # sum over kz -> f^(kx, z)
    # streamwise: Hermitian half spectrum (kx = 0..Nx/2) -> real
    G = np.zeros((Nx // 2 + 1, Nz), dtype=complex)
    G[:nxh, :] = g
    f = np.fft.irfft(G, n=Nx, axis=0) * Nx              # (Nx, Nz)
    return f.T                                          # (Nz, Nx): rows z, columns x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('restart', help='restart file name without extension (e.g. data/restart)')
    ap.add_argument('-o', '--output', default=None, help='output .mat file (default: <restart>_slices.mat)')
    ap.add_argument('--y', type=float, nargs='+', default=None, help='y of the xz-planes (default -0.9 0)')
    ap.add_argument('--yplus', type=float, nargs='+', default=None,
                    help='xz-planes at y+ from the bottom wall (needs --re and --utau)')
    ap.add_argument('--re', type=float, default=None, help='Reynolds number re of input.nml (for --yplus)')
    ap.add_argument('--utau', type=float, default=None, help='friction velocity (for --yplus)')
    ap.add_argument('--z', type=float, default=0.0, help='z of the xy-plane (default 0)')
    ap.add_argument('--x', type=float, default=0.0, help='x of the yz-plane (default 0)')
    ap.add_argument('--upsample', type=float, default=1.0,
                    help='output grid Nx = upsample*nx, Nz = upsample*nz (exact Fourier interpolation)')
    ap.add_argument('--fields', nargs='+', default=None, help='subset, e.g. u v s1 (default: all)')
    ap.add_argument('--fluct', action='store_true',
                    help='subtract the x-z mean (kx = kz = 0 mode): fluctuations u\', v\', ..., s\'')
    ap.add_argument('--grid-file', default=None, help="y grid file if grid_type = 'file'")
    a = ap.parse_args()

    base = a.restart[:-4] if a.restart.endswith(('.bin', '.hdr')) else a.restart
    h = read_header(base)
    nx, ny, nz = h['nx'], h['ny'], h['nz']
    y = wall_normal_grid(h, a.grid_file)
    Nx, Nz = even(a.upsample * nx), even(a.upsample * nz)
    x = h['lx'] * np.arange(Nx) / Nx
    z = h['lz'] * np.arange(Nz) / Nz

    names = ['u', 'v', 'w'] + ['s%d' % (n + 1) for n in range(h['nscal'])]
    sel = names if a.fields is None else a.fields
    for f in sel:
        if f not in names:
            sys.exit('unknown field %s (available: %s)' % (f, ' '.join(names)))

    if a.yplus is not None:
        if a.re is None or a.utau is None:
            sys.exit('--yplus needs --re and --utau')
        ytarget = [-1.0 + yp / (a.utau * a.re) for yp in a.yplus]
    else:
        ytarget = a.y if a.y is not None else [-0.9, 0.0]
    jxz = [int(np.argmin(np.abs(y - yt))) for yt in ytarget]
    iz = int(round(a.z / h['lz'] * Nz)) % Nz
    ix = int(round(a.x / h['lx'] * Nx)) % Nx

    out = dict(x=x, y=y, z=z, time=h['time'], step=h['step'], lx=h['lx'], lz=h['lz'],
               nx=nx, ny=ny, nz=nz, Nx=Nx, Nz=Nz,
               y_xz=y[jxz], j_xz=np.array(jxz) + 1, z_xy=z[iz], x_yz=x[ix],
               fluct=int(a.fluct), fields=np.array(sel, dtype=object))
    print('restart: nx ny nz = %d %d %d, %d scalar(s), t = %.6g, step %d' % (nx, ny, nz, h['nscal'],
                                                                             h['time'], h['step']))
    print('output grid Nx x Nz = %d x %d; xz-planes at y =' % (Nx, Nz), ' '.join('%.5f' % y[j] for j in jxz),
          '; xy-plane at z = %.5f; yz-plane at x = %.5f' % (z[iz], x[ix]))
    for name in sel:
        fld = Field(base, h, names.index(name))
        xz = np.zeros((Nz, Nx, len(jxz)))
        xy = np.zeros((ny, Nx))
        yz = np.zeros((ny, Nz))
        for j in range(ny):
            fh = fld.plane(j).copy()
            if a.fluct:
                fh[0, 0] = 0.0
            p = to_physical(fh, nx, nz, Nx, Nz)
            xy[j, :] = p[iz, :]
            yz[j, :] = p[:, ix]
            for m, jj in enumerate(jxz):
                if jj == j:
                    xz[:, :, m] = p
        out['xz_' + name] = xz
        out['xy_' + name] = xy
        out['yz_' + name] = yz
        print('  %-3s  min %12.5e  max %12.5e' % (name, min(xy.min(), yz.min(), xz.min()),
                                                  max(xy.max(), yz.max(), xz.max())))
    try:
        from scipy.io import savemat
    except ImportError:
        sys.exit('scipy is required to write .mat files (pip install scipy)')
    fout = a.output or base + '_slices.mat'
    savemat(fout, out, do_compression=True)
    print('wrote', fout)


if __name__ == '__main__':
    main()
