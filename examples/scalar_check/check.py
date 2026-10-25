#!/usr/bin/env python3
# compare the wall gradients of the five scalars (last 10 columns of data/log.dat) with the exact solutions
import numpy as np
d = np.loadtxt('data/log.dat')
g = d[-1, -10:]
ex = np.array([5, -5, 5, -5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2])
for n in range(5):
    print('S%d  dS/dy bottom %12.8f  top %12.8f   exact %5.2f %5.2f' % (n+1, g[2*n], g[2*n+1], ex[2*n], ex[2*n+1]))
print('t = %.1f   max rel. error = %.2e' % (d[-1, 1], np.max(np.abs(g-ex)/np.abs(ex))))
