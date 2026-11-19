"""Memory_1 vs Memory_2 reconciliation at one ray (pc14). Predictions: model/predictions_5090_batch5.md part A (committed first).
    PC14_TTU_APPROVED=cesar_jack_2026-11-19 TTU_GPU=0 python3 mem1b_pc14.py NEWWORKDIR
T1: 1 warp, rays 0..15 of pool 2026, one ray per launch (32 lanes). T2: 16 warps, rays 0..7 (512 lanes, one ray). Both carve outs, 11 launches (first dropped).
Same steps and gates as mem1_pc14.py: stage 0 first, one stage 1 launch (1 warp, ray 0) before the matrix; stops at the first fault, wrong exit or missing block."""
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
pool = make_rays(512, 2026); out = []
for W, rays in ((1, range(16)), (16, range(8))):
    for i in rays:
        f = '%s/rays_%d_%d.bin' % (work, W, i); pool[np.full(32 * W, i)].tofile(f); row = {'W': W, 'ray': i}
        for name, ex in (('l1big', []), ('l1small', ['--smem', '98304'])):
            j = C.go(b1, 'file:' + f, '%s/r_%d_%d_%s.json' % (work, W, i, name), ['--scene', 'soup:100000', '--warps', str(W), '--repeat', '11'] + ex)
            row[name] = C.med([e - s for l in j['launches'][1:] for s, e in l['warp_clk']])
        row['ratio'] = row['l1small'] / row['l1big']; out.append(row); print('   ', row, flush=True)
json.dump(out, open(work + '/mem1b.json', 'x'))
