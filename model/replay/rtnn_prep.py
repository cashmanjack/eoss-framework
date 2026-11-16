"""RTNN replay inputs from a patched RTNN run's dump (RTNN_DUMP dir: gas0.bin, points.bin, aabb_radius0.txt, rays0.bin).
    rtnn_prep.py DUMP_DIR OUT_PREFIX      writes OUT_PREFIX.tris (pseudo triangles: lo, hi, lo of each primitive's box) and OUT_PREFIX.table (make_table.py)
The table's leaf slots are filled by box matching (as for triangles); the replay (replay_aabb.cpp) tests the query point against each primitive's box."""
import os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
d, out = sys.argv[1:3]
r = float(open(os.path.join(d, 'aabb_radius0.txt')).read().split()[0]); P = np.fromfile(os.path.join(d, 'points.bin'), np.float32).reshape(-1, 3)
lo, hi = P - np.float32(r), P + np.float32(r)
np.stack([lo, hi, lo], 1).astype(np.float32).tofile(out + '.tris')
subprocess.run([sys.executable, os.path.join(HERE, 'make_table.py'), os.path.join(d, 'gas0.bin'), out + '.tris', out + '.table'], env=dict(os.environ, ASSIGN='/dev/shm/rp/assign'), check=True)
