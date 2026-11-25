"""Baseline 1: RTNN (Zhu, PPoPP 2022), the build as it stands in ~/thirdparty/div1/rtnn (our copy, instrumentation switches off).
usage: python base_rtnn.py [REPS]   -> notes/base1/rtnn_<ts>.json. Change no code."""
import re, sys, base_common as B
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
D = B.HOME + '/thirdparty/div1'; BIN = D + '/rtnn/src/build/bin/optixNSearch'; DATA = D + '/data/u100000.csv'
def parse(t):
    p = {}
    for m in re.finditer(r'^time ([a-zA-Z0-9 /_,]+?): ([0-9.eE+-]+) ms', t, re.M): p[m.group(1).strip().replace(' ', '_')] = float(m.group(2))
    m = re.search(r'KERNEL queries (\d+) ms ([0-9.]+)', t)
    if m: p['queries'] = int(m.group(1)); p['kernel_ms'] = float(m.group(2))
    return p
doc = B.header('RTNN', BIN, [DATA], 'u100000.csv = 100,000 uniform random 3D points, queries = points; -k 50 -p 0 -s 0 -m 0 -d 0 (the flags the earlier 4090 runs used). RTNN_DRAM_GB=11.5958 (the memory size the 4070 Ti based partition rule assumed; kept so the batching matches earlier runs). Build contains lane histogram and callback clock code; both off here (no LHOUT, no RTNN_REC).')
doc['configs'] = []
for r in ('0.05', '0.025'):
    cmd = [BIN, '-f', DATA, '-r', r, '-k', '50', '-p', '0', '-s', '0', '-m', '0', '-d', '0']
    runs = []
    for i in range(REPS):
        x = B.run(cmd, D + '/rtnn/src/build/bin', B.env(extra={'RTNN_DRAM_GB': '11.5958'}), 120); x['parsed'] = parse(x['stdout']); runs.append(x)
    keys = sorted({k for x in runs for k in x['parsed']})
    doc['configs'].append(dict(name='radius_' + r, cmd=' '.join(cmd).replace(B.HOME, '~'), runs=runs, summary=B.summarize(runs, keys + [])
                               | {'wall_s': B.stats([x['wall_s'] for x in runs if x['rc'] == 0])}))
print(B.write('rtnn', doc))
