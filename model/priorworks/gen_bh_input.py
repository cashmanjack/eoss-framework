"""Gaussian disc input for RT-BarnesHut 'treelogy' mode (same recipe as ~/generate_synthetic1M.py, seed 42).
usage: python gen_bh_input.py N OUT   (refuses to overwrite). Header: 5 floats, one per line (the program reads and discards them)."""
import sys, numpy as np
n, out = int(sys.argv[1]), sys.argv[2]
np.random.seed(42)
x = np.random.normal(0, 100, n).astype(np.float32); y = np.random.normal(0, 100, n).astype(np.float32); z = np.random.normal(0, 20, n).astype(np.float32)
m = np.random.uniform(1.0, 100.0, n).astype(np.float32)
with open(out, 'x') as f:
    f.write('%d\n1\n0.5\n0.01\n1\n' % n)
    for i in range(n): f.write('%f %f %f %f 0.000000 0.000000 0.000000\n' % (m[i], x[i], y[i], z[i]))
