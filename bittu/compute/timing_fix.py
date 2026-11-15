"""BITTU timing fix check: for each scene, (a) stage 0 twin and stage 1 under the harness clock only, (b) stage 0 twin with the empty probe pair (BITTU_MODE=empty), (c) stage 1 under BITTU.
corrected = window(stage 1 under tool) - window(empty pair under tool); compared with the harness clock of the same group without the tool. 1 warp, 5 launches (first dropped).
    timing_fix.py WORK       run with ~/eoss/.venv/bin/python (run_bittu.py needs the eoss package). Only the checked group; every run under run.py's 30 s timeout; any failure stops."""
import json, os, re, statistics as st, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); S3 = os.path.join(HERE, '..', '..', 'ttu', 'harness', 'setup3', 'run.py')
B0, B1 = '/dev/shm/t9r/g_s0c', '/dev/shm/t9r/g_s1c'
SCENES = [('d1', 'tri1', 'tri:0'), ('d2', 'grid:144', 'tri:72'), ('d3', 'grid:2500', 'tri:1250')]


def sh(cmd, env=None):
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode or 'ttu_one: no error' not in p.stdout: sys.exit('STOP\n' + p.stdout[-2000:] + p.stderr[-1500:])
    return p.stdout


def clk(j): return st.median(l['warp_clk'][0][1] - l['warp_clk'][0][0] for l in json.load(open(j))['launches'][1:])


def win(txt):
    m = [int(x) for x in re.findall(r'median (\d+), probe overhead', open(txt).read())][1:]
    o = [int(x) for x in re.findall(r'overhead o \(t0b - t0a\): median (\d+)', open(txt).read())][1:]
    return st.median(m), st.median(o)


def main(work):
    os.makedirs(work, exist_ok=True); res = {}
    for tag, sc, ray in SCENES:
        a = [ray, 'zero']; ex = ['--scene', sc, '--warps', '1', '--repeat', '5']; o = lambda n: os.path.join(work, '%s_%s' % (tag, n))
        sh(['/usr/bin/python3', S3, B0, a[0], a[1], o('b0.json')] + ex)
        sh(['/usr/bin/python3', S3, B1, a[0], a[1], o('b1.json')] + ex)
        env = dict(os.environ, BITTU_PRINT='0')
        sh([sys.executable, os.path.join(HERE, 'run_bittu.py'), B0, a[0], a[1], o('t0.json'), o('t0.txt')] + ex, env=dict(env, BITTU_LIKE=B1))
        sh([sys.executable, os.path.join(HERE, 'run_bittu.py'), B1, a[0], a[1], o('t1.json'), o('t1.txt')] + ex, env=env)
        w0, o0 = win(o('t0.txt')); w1, o1 = win(o('t1.txt')); h1, h0 = clk(o('b1.json')), clk(o('b0.json'))
        res[tag] = dict(scene=sc, ray=ray, hit=json.load(open(o('b1.json')))['ref'][0]['hit'], harness_s1=h1, harness_s0=h0, tool_s1_window=w1, tool_empty_window=w0, probe_cost_s1=o1, probe_cost_empty=o0, corrected=w1 - w0, ratio=(w1 - w0) / h1)
        print(tag, json.dumps(res[tag]), flush=True)
    json.dump(res, open(os.path.join(work, 'timing_fix.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
