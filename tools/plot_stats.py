#!/usr/bin/env python3
"""
Convert data/stats.dat to wall units (averaging the two channel halves) and optionally plot.

    python3 plot_stats.py data/stats.dat --re 2800 [--plot]

Writes stats_plus.dat: y+, U+, u_rms+, v_rms+, w_rms+, -uv+, and for each scalar S, s_rms, -vs
(bottom half, not scaled).
u_tau is taken from the mean wall gradient, computed with the 4th-order one-sided difference
of the averaged profile (the code itself uses the compact scheme; see tau_w in log.dat).
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file')
    ap.add_argument('--re', type=float, required=True, help='Reynolds number re of input.nml')
    ap.add_argument('--plot', action='store_true')
    a = ap.parse_args()
    with open(a.file) as f:
        head = [l for l in f if l.startswith('#')]
    names = head[-1].split()[1:]
    d = np.loadtxt(a.file)
    col = {n: i for i, n in enumerate(names)}
    y = d[:, 0]
    n = len(y)
    h = n // 2 + 1
    yb = 1 + y[:h]                                   # distance from the wall
    sym = lambda q, s=1: 0.5 * (d[:h, col[q]] + s * d[::-1, col[q]][:h])
    U = sym('U')
    # 4th-order one-sided derivative at the wall on a non-uniform grid (Lagrange)
    x = yb[:5]
    dudy = sum(U[j] * np.prod([1 / (x[j] - x[m]) for m in range(5) if m != j]) *
               sum(np.prod([-x[m] for m in range(5) if m not in (j, l)]) for l in range(5) if l != j)
               for j in range(5))
    nu = 1 / a.re
    utau = np.sqrt(nu * abs(dudy))
    print('u_tau = %.5f   Re_tau = %.2f' % (utau, utau / nu))
    out = [yb * utau / nu, U / utau,
           np.sqrt(np.abs(sym('uu'))) / utau, np.sqrt(np.abs(sym('vv'))) / utau,
           np.sqrt(np.abs(sym('ww'))) / utau, -sym('uv', -1) / utau ** 2]
    hdr = 'y+ U+ urms+ vrms+ wrms+ -uv+'
    k = 1
    while 'S%d' % k in col:
        bot = lambda q: d[:h, col[q]]               # scalars: bottom half only (may be antisymmetric)
        out += [bot('S%d' % k), np.sqrt(np.abs(bot('s%ds%d' % (k, k)))), -bot('vs%d' % k)]
        hdr += ' S%d srms%d vs%d' % (k, k, k)
        k += 1
    np.savetxt('stats_plus.dat', np.array(out).T, header=hdr)
    print('wrote stats_plus.dat (velocity: average of both halves; scalars: bottom half, outer units)')
    if a.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].semilogx(out[0][1:], out[1][1:], label='U+')
        ax[0].set_xlabel('y+')
        for i, lab in zip(range(2, 6), ['urms', 'vrms', 'wrms', '-uv']):
            ax[1].plot(out[0], out[i], label=lab)
        ax[1].set_xlabel('y+')
        for x_ in ax:
            x_.legend()
        plt.tight_layout()
        plt.savefig('stats_plus.png', dpi=120)
        print('wrote stats_plus.png')


if __name__ == '__main__':
    main()
