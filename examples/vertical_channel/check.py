#!/usr/bin/env python3
# compare the steady natural-convection profile with u = ri re (y^3 - y)/6 (projected on gravity)
import re as regex
import numpy as np
txt = '\n'.join(l.split('!')[0] for l in open('input.nml'))   # without comments
num = lambda key: float(regex.search(r'\b' + key + r'\s*=\s*([-0-9.eE+]+)', txt).group(1))
ri, rey = num(r'ri\(1\)'), num(r're')
g = np.array([float(x) for x in regex.search(r'gravity\s*=\s*([-0-9.eE+, ]+)', txt).group(1).split(',')[:3]])
g /= np.linalg.norm(g)
d = np.loadtxt('data/stats.dat')
y, U, V, W = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
ex = ri * rey * (y**3 - y) / 6
print('max |U - u_ex*(-g_x)| = %.2e,  max |W - u_ex*(-g_z)| = %.2e,  max |V| = %.1e   (max |u_ex| = %.3f)'
      % (abs(U + g[0] * ex).max(), abs(W + g[2] * ex).max(), abs(V).max(), abs(ex).max()))
