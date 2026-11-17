"""Unit_1 runner (predictions: model/predictions_unit1.md). One SM, checked remap build of Throughput_1, scenes par:N:s|o and tri1.

    run.py WORK [first|stage0|runs]      (run with /usr/bin/python3; builds are made beforehand in /dev/shm/t9c: t1r_s0, t1r_s1, both check_seq ALL PASS)
      first   : the one smallest TTU launch (tri1, 1 warp, 1 launch)
      stage0  : no TTU instruction, 32 warps, the two largest scenes
      runs    : per scene 1 warp x 30 launches (latency) and 32 warps x 30 launches (rate); stops at the first failure, nothing retried
Raw JSON per run: WORK/u1_<scene>_w<warps>_<timestamp>.json (refuses to overwrite). Needs TTU_GPU set (gpucfg), run.py of setup3 does the idle check and the 30 s timeout."""
import json, os, statistics as st, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
S3 = os.path.join(HERE, '..', '..', 'harness', 'setup3')
B0, B1 = '/dev/shm/t9c/t1r_s0', '/dev/shm/t9c/t1r_s1'
TS = time.strftime('%Y%m%dT%H%M%S')
SCENES = ['par:%d:s' % n for n in (1, 4, 16, 64, 256, 1024, 4096)] + ['par:%d:o' % n for n in (16, 64, 256, 1024)]


def go(binp, ray, out, extra, blocks=1):
    if os.path.exists(out): sys.exit('exists ' + out)
    p = subprocess.run([sys.executable, os.path.join(S3, 'run.py'), binp, ray, 'zero', out] + extra, capture_output=True, text=True)
    ok = 'exit 0' in p.stdout and 'ttu_one: no error' in p.stdout and ('%d block(s) ran' % blocks) in p.stdout and 'GPU' in p.stdout
    print('   ', os.path.basename(out), 'OK' if ok else 'FAILED', flush=True)
    if not ok: sys.exit('STOP (fault, hang, wrong exit or missing block):\n' + p.stdout[-2500:] + p.stderr[-1500:])
    return json.load(open(out))


def stats(j):
    L = j['launches'][1:] if len(j['launches']) > 1 else j['launches']; w = j['warps']
    span = [max(e for _, e in l['warp_clk']) - min(s for s, _ in l['warp_clk']) for l in L]
    gaps = []
    for l in L:
        ends = sorted(e for _, e in l['warp_clk']); h = ends[len(ends) // 2:]; gaps += [b - a for a, b in zip(h, h[1:])]
    return dict(w=w, rpc=st.median([32 * w / s for s in span]), lat=st.median([e - s for l in L for s, e in l['warp_clk']]), gap=st.median(gaps) if gaps else None,
                span=st.median(span), launches=len(L))


if __name__ == '__main__':
    work, step = sys.argv[1], sys.argv[2]; os.makedirs(work, exist_ok=True)
    if step == 'first':
        go(B1, 'hit', '%s/u1_first_tri1_w1_%s.json' % (work, TS), ['--warps', '1', '--repeat', '1'])
    elif step == 'stage0':
        for sc in ('par:4096:s', 'par:1024:o'):
            go(B0, 't2', '%s/u1_s0_%s_w32_%s.json' % (work, sc.replace(':', ''), TS), ['--scene', sc, '--warps', '32', '--repeat', '2'])
    elif step == 'runs':
        res = []
        for sc in SCENES:
            for w in (1, 32):
                j = go(B1, 't2', '%s/u1_%s_w%d_%s.json' % (work, sc.replace(':', ''), w, TS), ['--scene', sc, '--warps', str(w), '--repeat', '30'])
                s = stats(j); s['scene'] = sc; res.append(s); print('   ', s, flush=True)
        json.dump(res, open('%s/u1_summary_%s.json' % (work, TS), 'x'))
