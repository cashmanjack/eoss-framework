"""Finding 82 check on the CPU, from saved data (no GPU): the first query of 32 random rays (notes/setup3/runs/traversal2b2/s1clk.json, soup:100000)
returned pending node pointers (words R56..R61, R70, R71). The CPU replay (nearest child first, depth first) is stopped at each ray's first return
and its pending entries are compared with the node pointers the RT core returned.   verify_f82.py RUNDIR SOUP_TRIS WORK
"""
import json, os, struct, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', '..', 'ttu', 'bench', 'BVH_5'))
import gas  # noqa: E402
run, tris, work = sys.argv[1:4]; T = '/dev/shm/cashman3-tmp'
j = json.load(open(os.path.join(run, 's1clk.json'))); gp = os.path.join(run, 'soup.gas'); G = open(gp, 'rb').read(); W = gas.walk(G)
base = int(j['gas'], 16); addr = {base + o: o for o in W}
root = struct.unpack_from('<Q', G, 0x20)[0]; idx = {root: 0}; order = [root]; i = 0
while i < len(order):
    for s, k, ch, box in gas.slots(G, order[i]):
        if k == 'node': idx[ch] = len(order); order.append(ch)
    i += 1
os.makedirs(work, exist_ok=True)
subprocess.run(['/usr/bin/python3', os.path.join(HERE, 'make_table.py'), gp, tris, work + '/table.bin'], env=dict(os.environ, ASSIGN=T + '/assign'), check=True)
rays = np.zeros((32, 8), np.float32)
for l, r in enumerate(j['lane_rays']): rays[l, 0:3] = r[0:3]; rays[l, 4:7] = r[3:6]; rays[l, 7] = 1e16
rays.tofile(work + '/rays.bin')
subprocess.run([T + '/pend', work + '/table.bin', work + '/rays.bin', '32', '0', work + '/pend.txt'], check=True)
cpu = {}
for ln in open(work + '/pend.txt'):
    h, rest = ln.split(':', 1); l = int(h.split()[0]); cpu[l] = [(e[0], int(e[1:].split('@')[0]), float(e.split('@')[1])) for e in rest.split()]
inv = {v: k for k, v in idx.items()}  # table node idx -> gas offset
def ptrs(o):
    out = []
    for a in range(56, 72):
        if a + 1 < len(o) and (o[a] | o[a + 1]):
            v = ((o[a + 1] & 0xff) << 32) | o[a]
            if (v << 7) in addr: out.append(addr[v << 7])
    return sorted(set(out))
tot = dict(near_t=0, in_unculled=0, lanes=0, gpu_ptrs=0, in_cpu=0, two_depth_gpu=0, two_depth_cpu=0, same_set_top=0)
for l in range(32):
    o = j['out'][l]; g = ptrs(o); c = cpu.get(l, [])
    cn = [(inv[n], t) for k, n, t in c if k == 'n']                 # stack bottom -> top
    gd = sorted({W[x]['depth'] for x in g}); cd = sorted({W[x]['depth'] for x, _ in cn})
    inset = [x in {a for a, _ in cn} for x in g]
    top = [a for a, _ in cn[::-1][:len(g)]]
    same = set(g) == set(top)
    best = float(open(work + '/pend.txt').read().split('\n')[0].split()[2].rstrip(':')) if False else None
    nt = [a for a, _ in sorted(cn, key=lambda x: x[1])[:len(g)]]
    tot['near_t'] += set(g) == set(nt)
    tot['lanes'] += 1; tot['gpu_ptrs'] += len(g); tot['in_cpu'] += sum(inset); tot['two_depth_gpu'] += len(gd) > 1; tot['two_depth_cpu'] += len(cd) > 1; tot['same_set_top'] += same
    print('lane %2d queries %d | gpu ptrs %d depths %s | cpu pending nodes %d depths %s | gpu in cpu %d/%d | gpu == top %d of cpu stack: %s' % (l, j['lane_queries'][l], len(g), gd, len(cn), cd, sum(inset), len(g), len(g), same))
print(tot)
