#!/usr/bin/env python3
# run the Rayleigh-Benard onset case at three Richardson numbers and estimate Ra_c
#   python3 check.py [path/to/hichan] [mpirun command]
import os, re, subprocess, sys
import numpy as np
exe = sys.argv[1] if len(sys.argv) > 1 else '../../build/hichan'
launcher = sys.argv[2].split() if len(sys.argv) > 2 else ['mpirun', '-np', '1']
ric = 1707.76 / (16 * 100.**2 * 1.0)
base = open('input.nml').read()
facs, sig = [0.9, 1.0, 1.1], []
for f in facs:
    d = 'run_%.1f' % f
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, 'input.nml'), 'w').write(re.sub(r'ri\(1\)\s*=\s*[0-9.eE+-]+', 'ri(1) = %.10e' % (f * ric), base))
    subprocess.run(launcher + [os.path.abspath(exe)], cwd=d, stdout=subprocess.DEVNULL, check=True)
    a = np.loadtxt(os.path.join(d, 'data', 'log.dat'))
    m = a[:, 1] > 800
    sig.append(np.polyfit(a[m, 1], np.log(a[m, 9]), 1)[0] / 2)
    print('ri/ri_c = %.1f   growth rate = %+.4e' % (f, sig[-1]))
p = np.polyfit(facs, sig, 2)
r = [x.real for x in np.roots(p) if 0.5 < x.real < 1.5][0]
print('Ra_c = %.2f   (linear theory 1707.76)' % (1707.76 * r))
