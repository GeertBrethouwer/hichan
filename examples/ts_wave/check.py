#!/usr/bin/env python3
# fit the decay rate of the TS wave from data/log.dat
import numpy as np
d = np.loadtxt('data/log.dat')
t, e = d[:, 1], d[:, 9]
m = t > 1
p = np.polyfit(t[m], np.log(e[m]), 1)
wi = -2.7829009e-3
print('omega_i code = %.8e   Orr-Sommerfeld = %.8e   rel. error = %.2e' % (p[0]/2, wi, abs(p[0]/2-wi)/abs(wi)))
