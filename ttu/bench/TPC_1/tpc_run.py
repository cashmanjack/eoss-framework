"""TPC_1: RT work and compute work on different SMs of a TPC (OptiX only, no TTU code).   /usr/bin/python3 tpc_run.py WORK LOG2_RAYS
RT: OptiX closest hit launches of 2^LOG2_RAYS random rays (random order, and Morton sorted by origin 5 bits per axis), soup 1e5 and 3e6, 40 launches, median launch time.
Compute: persistent plain CUDA kernel, 384 thread blocks. place all: 2 blocks per SM on all 142 SMs (RT blocks fit beside them). place odd: 4 blocks per SM on the 71 odd SMs only (all resident threads used,
so OptiX cannot land there; it runs on the 71 even SMs, one per TPC). Types: alu (FMA chain), l1 (re-reads 32 KB per block), mem (streams 1 GB).
Rows: RT alone; compute alone (all, odd); both together (all, odd). Compute rate = block iterations per second sampled over the window."""
import json, os, subprocess, sys, statistics as st
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'Morton')); sys.path.insert(0, os.path.join(HERE, '..', 'Memory_4'))
import morton_run as mr  # noqa: E402
B = '/dev/shm/tpc'; ENV = mr.ENV


def run(scene, rays, launches, place, typ):
    mr.gpu_free()
    r = subprocess.run(['timeout', '-k', '2', '120', B + '/tpc', B + '/tpc_dev.ptx', str(scene), rays, str(launches), place, typ, '150'], env=ENV, capture_output=True, text=True)
    if r.returncode: sys.exit('failed %d\n%s\n%s' % (r.returncode, r.stdout[-800:], r.stderr[-800:]))
    return json.loads([l for l in r.stdout.splitlines() if l.startswith('{')][0])


def main(work, lg):
    os.makedirs(work, exist_ok=True); n = 1 << int(lg); out = []
    R = mr.prep.make_rays(n)
    o5 = mr.code_o(R, 5); sets = {'random': R, 'morton': R[mr.order_by(o5)]}
    for scene in (100000, 3000000):
        for sname, RR in sets.items():
            f = '/dev/shm/tpc/rays.bin'; RR.tofile(f)
            base = run(scene, f, 40, 'none', 'alu'); rt0 = n / st.median(base['ms']) / 1000
            print('%d %s: RT alone %.0f rays/us' % (scene, sname, rt0), flush=True); out.append(dict(scene=scene, order=sname, place='none', type='-', rt=rt0, ms=base['ms']))
            for typ in ('alu', 'l1', 'mem'):
                for place in ('all', 'odd'):
                    c = run(scene, f, 0, place, typ) if sname == 'random' and scene == 100000 else None
                    both = run(scene, f, 40, place, typ); rt = n / st.median(both['ms']) / 1000
                    ci = both['compute_iters'] / both['window_s']; c0 = c['compute_iters'] / c['window_s'] if c else None
                    print('  %-3s %-4s RT %6.0f (%.2fx alone) | compute %.3g it/s%s | blocks even %d odd %d, SMs %d' % (place, typ, rt, rt / rt0, ci, '' if c0 is None else ' (alone %.3g, %.2fx)' % (c0, ci / c0), both['blocks_even'], both['blocks_odd'], both['sms_with_blocks']), flush=True)
                    out.append(dict(scene=scene, order=sname, place=place, type=typ, rt=rt, compute_rate=ci, compute_alone=c0, blocks_even=both['blocks_even'], blocks_odd=both['blocks_odd'], sms=both['sms_with_blocks'], ms=both['ms']))
    # compute alone on both placements for every type (scene independent): measured above only on the first set; add all here
    for typ in ('alu', 'l1', 'mem'):
        for place in ('all', 'odd'):
            c = run(100000, '/dev/null', 0, place, typ); print('compute alone %-3s %-4s %.3g it/s (blocks even %d odd %d, SMs %d)' % (place, typ, c['compute_iters'] / c['window_s'], c['blocks_even'], c['blocks_odd'], c['sms_with_blocks']), flush=True)
            out.append(dict(place=place, type=typ, compute_alone_only=c['compute_iters'] / c['window_s'], blocks_even=c['blocks_even'], blocks_odd=c['blocks_odd'], sms=c['sms_with_blocks']))
    json.dump(out, open(os.path.join(work, 'tpc1_%s.json' % lg), 'w'), indent=1)


if __name__ == '__main__': main(*sys.argv[1:3])
