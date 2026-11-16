"""Parsers for the queue 14 probe logs (notes/q14/logs). CPU only."""
import re, glob, os, statistics as st
LOG = os.path.expanduser('~/eoss/notes/q14/logs')
CBK = re.compile(r'CBK (?:round (\d+) )?warps (\d+) total_warp_cycles (\d+) trace_warp_cycles (\d+) f0_cycles (\d+) f0_calls (\d+) f1_cycles (\d+) f1_calls (\d+) f2_cycles (\d+) f2_calls (\d+) f3_cycles (\d+) f3_calls (\d+)')
def cbk(m):
    k = ['warps', 'total', 'trace', 'f0c', 'f0n', 'f1c', 'f1n', 'f2c', 'f2n', 'f3c', 'f3n']
    d = dict(zip(k, [int(x) for x in m.groups()[1:]])); d['round'] = int(m.group(1)) if m.group(1) else 0; return d
def read(path):
    return open(path).read() if os.path.exists(path) else ''
def runs(prefix, mode, reps=(1, 2, 3)):
    """list of parsed dicts per rep for file prefix_mode_rep.log; RTNN, Arkade: one launch; BH: PHASE; kNN: one dict per round (key 'rounds')"""
    out = []
    for r in reps:
        t = read('%s/%s_%s_%d.log' % (LOG, prefix, mode, r)); d = {}
        m = re.search(r'KERNEL queries (\d+) ms ([\d.]+)', t)
        if m: d['ms'] = float(m.group(2)); d['q'] = int(m.group(1))
        m = re.search(r'LAUNCH_MS ([\d.]+)', t)
        if m: d['ms'] = float(m.group(1))
        m = re.search(r'PHASE prologue_cycles (\d+) loop_cycles (\d+) epilogue_cycles (\d+)', t)
        if m: d['phase'] = tuple(int(x) for x in m.groups())
        ms = [cbk(x) for x in CBK.finditer(t)]
        if prefix == 'knn':
            rk = {}
            for x in re.finditer(r'ROUND (\d+) active (\d+) radius \S+ KERNEL_MS ([\d.]+)', t): rk[int(x.group(1))] = dict(active=int(x.group(2)), ms=float(x.group(3)))
            for c in ms: rk.setdefault(c['round'], {}).update(c)
            d['rounds'] = rk
        elif ms: d.update(ms[0])
        out.append(d)
    return out
def med(xs): xs = [x for x in xs if x is not None]; return st.median(xs) if xs else float('nan')
if __name__ == '__main__':
    for mode in ('loaded', 'small', 'off'):
        print('==', mode)
        for p in sorted(set(re.sub(r'_%s_\d\.log' % mode, '', os.path.basename(f)) for f in glob.glob('%s/*_%s_*.log' % (LOG, mode)))):
            rs = runs(p, mode)
            if p == 'knn':
                for rd in range(1, 6): print(p, rd, [round(r['rounds'].get(rd, {}).get('ms', float('nan')), 3) for r in rs], 'f0 cyc/call', [round(r['rounds'][rd]['f0c'] / r['rounds'][rd]['f0n']) if r['rounds'].get(rd, {}).get('f0n') else None for r in rs][:1])
            else: print(p, [r.get('ms') for r in rs], 'f0 cyc/call', [round(r['f0c'] / r['f0n']) if r.get('f0n') else None for r in rs][:1], 'f1', [round(r['f1c'] / r['f1n']) if r.get('f1n') else None for r in rs][:1])
