#!/usr/bin/env python3
"""
Least-stable Orr-Sommerfeld mode of plane Poiseuille flow U = 1 - y^2 (Chebyshev collocation).

Writes the eigenfunction (u, v, w) on the HiCHAN wall-normal grid, in the format read by
init_type='mode' (ny lines: Re(u) Im(u) Re(v) Im(v) Re(w) Im(w)), and prints the
temporal growth rate omega_i (the energy of the mode evolves as exp(2 omega_i t)).

Example (TS-wave test of Lenaers et al. 2014, Sec. 5.1):
    python3 os_mode.py --re 5000 --alpha 1.12 --ny 129 --grid tanh --galpha 2.0 -o mode.in
"""
import argparse
import numpy as np
from scipy.linalg import eig


def cheb(n):
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.hstack([2, np.ones(n - 1), 2]) * (-1) ** np.arange(n + 1)
    X = np.tile(x, (n + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(n + 1))
    D -= np.diag(D.sum(axis=1))
    return D, x


def hichan_grid(ny, kind, alpha, fname=None):
    if kind == 'tanh':
        xi = np.arange(ny) / (ny - 1)
        y = np.tanh(alpha * (xi - 0.5)) / np.tanh(0.5 * alpha)
    elif kind == 'uniform':
        y = np.linspace(-1, 1, ny)
    else:
        y = np.loadtxt(fname)[:ny]
    y[0], y[-1] = -1.0, 1.0
    return y


def cheb_interp(x, f, xn):
    """barycentric interpolation from Chebyshev-Gauss-Lobatto points x to xn"""
    n = len(x) - 1
    w = (-1.0) ** np.arange(n + 1)
    w[0] *= 0.5
    w[-1] *= 0.5
    out = np.empty(len(xn), dtype=complex)
    for k, xv in enumerate(xn):
        d = xv - x
        j = np.argmin(np.abs(d))
        if abs(d[j]) < 1e-14:
            out[k] = f[j]
        else:
            t = w / d
            out[k] = np.dot(t, f) / t.sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--re', type=float, default=5000.0)
    ap.add_argument('--alpha', type=float, default=1.12)
    ap.add_argument('--n', type=int, default=150, help='Chebyshev resolution')
    ap.add_argument('--ny', type=int, required=True)
    ap.add_argument('--grid', default='tanh')
    ap.add_argument('--galpha', type=float, default=2.0)
    ap.add_argument('--gridfile', default=None)
    ap.add_argument('-o', '--out', default='mode.in')
    a = ap.parse_args()

    n = a.n
    D, x = cheb(n)
    D2 = D @ D
    D4 = D2 @ D2
    I = np.eye(n + 1)
    U = np.diag(1 - x ** 2)
    Upp = -2 * I
    al, re = a.alpha, a.re
    L = D2 - al ** 2 * I
    A = U @ L - Upp - (L @ L) / (1j * al * re)
    B = L.astype(complex)
    for r, row in ((0, I[0]), (1, D[0]), (n, I[n]), (n - 1, D[n])):
        A[r] = row
        B[r] = 0
    c, V = eig(A, B)
    ok = np.isfinite(c) & (np.abs(c) < 10)
    c, V = c[ok], V[:, ok]
    k = np.argmax(c.imag)
    cc, v = c[k], V[:, k]
    omega = al * cc
    v = v / v[np.argmax(np.abs(v))]
    u = 1j * (D @ v) / al          # continuity: i alpha u + Dv = 0
    y = hichan_grid(a.ny, a.grid, a.galpha, a.gridfile)
    ug = cheb_interp(x, u, y)
    vg = cheb_interp(x, v, y)
    ug[[0, -1]] = 0
    vg[[0, -1]] = 0
    with open(a.out, 'w') as f:
        for uu, vv in zip(ug, vg):
            f.write('%.16e %.16e %.16e %.16e 0.0 0.0\n' % (uu.real, uu.imag, vv.real, vv.imag))
    print('c = %.12e %+.12ei' % (cc.real, cc.imag))
    print('omega_r = %.12e  omega_i = %.12e' % (omega.real, omega.imag))


if __name__ == '__main__':
    main()
