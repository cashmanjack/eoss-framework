"""Memory_1 TTU half on the RTX 5090 (pc14): second trace latency for R distinct random rays (soup:100000), L1 about 90 KB against --smem 98304 reserved (L1 about 20 KB).
    PC14_TTU_APPROVED=cesar_jack_2026-11-19 TTU_GPU=0 python3 mem1_pc14.py NEWWORKDIR      (the plain chase part was done last session: model/results_5090_plain.md)
Predictions: model/predictions_5090_batch4.md item 3. Same steps as core_set.t_mem1 without the chase. Stops at the first fault, wrong exit or missing block."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'io_core')); sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
import numpy as np
import core_set as C  # noqa: E402
from prep import make_rays  # noqa: E402
work = sys.argv[1]; os.makedirs(work, exist_ok=False)
b0, b1 = C.build('m1', 0, ['--clock', '--twice', '--compact-out']), C.build('m1', 1, ['--clock', '--twice', '--compact-out'])
C.go(b0, 'hit', work + '/m1_s0.json', ['--warps', '1', '--repeat', '2'])
C.go(b0, 'hit', work + '/m1_s0_smem.json', ['--warps', '1', '--repeat', '2', '--smem', '98304'])
C.go(b1, 'hit', work + '/m1_first.json', ['--warps', '1', '--repeat', '1'])
out = []
for R in (1, 4, 8, 16, 32, 64, 128, 256):
    W = max(1, R // 32); f = '%s/m1_rays_%d.bin' % (work, R); make_rays(512, 2026)[np.arange(32 * W) % R].tofile(f); row = {'R': R}
    for name, ex in (('l1big', []), ('l1small', ['--smem', '98304'])):
        j = C.go(b1, 'file:' + f, '%s/m1_%d_%s.json' % (work, R, name), ['--scene', 'soup:100000', '--warps', str(W), '--repeat', '11'] + ex)
        row[name] = C.med([e - s for l in j['launches'][1:] for s, e in l['warp_clk']])
    row['ratio'] = row['l1small'] / row['l1big']; out.append(row); print('   ', row, flush=True)
json.dump(out, open(work + '/mem1.json', 'x'))
