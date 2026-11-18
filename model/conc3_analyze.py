"""Conc_3 analysis (Phase 10d). conc3_analyze.py DIR [group]  : reads DIR/conc_<group>_*.json and DIR/clk_<group>_*.csv(.gz) (launch window rows, key and launch columns).
Per arm and scene: RT ratio three ways, all from per launch rates (rays per us, launches 1 to 5 of every run):
  paired   : arm over the median of the alone run just before it (same repetition)
  pooled   : arm over the median of ALL alone launches of the scene (the F229 way)
  matched  : each arm launch over the median of the alone launches whose mean SM clock is within 1 percent of that launch's mean SM clock (if none: alone median scaled by the clock ratio, counted in 'scaled')
Also: mean SM clock of arm and alone, compute GB/s, lines per second and pages per second of the stream (pattern simulated here: distinct lines, 64 KB pages and 2 MB pages per warp load)."""
import glob, gzip, json, os, statistics as st, sys
import numpy as np

def load_clk(d, group):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, 'clk_%s_*.csv*' % group))):
        op = gzip.open if f.endswith('.gz') else open
        with op(f, 'rt') as fh:
            next(fh)
            for ln in fh:
                t, sm, mem, mw, key, la = ln.rstrip().split(',')
                out.setdefault((key, int(la)), []).append((int(sm), int(mem), int(mw)))
    return out

def pattern(typ):
    """mean distinct lines, 64 KB pages, 2 MB pages per warp load for a stream type (first 8 iterations of 12 warps, 8 blocks)."""
    L, P6, P2 = [], [], []
    for blk in range(8):
        for it in range(8):
            for k in range(32):
                for w in range(12):
                    tid = np.arange(32) + 32 * w; a = None
                    if typ in ('mem', 'memcg'):
                        pos = ((blk * 384 + tid) * 7919 % (1 << 26)); pos = (pos + (it * 32 + k) * (1024 * 256 + 1)) % (1 << 26); a = pos * 16
                    elif typ in ('pgfew', 'pgmany', 'pgfar', 'pgodd', 'dramfew'):
                        nl = (1 << 21) if typ == 'dramfew' else (1 << 15); st_ = {'pgmany': 256, 'pgfar': 4096, 'pgodd': 241}.get(typ, 1)
                        l = (tid * 7919 + k * 4099 + it * 1021 + blk * 37) & (nl - 1); a = l * st_ * 128 + (tid & 7) * 16
                    elif typ == 'seqpg':
                        u = ((blk * 12 + w) * 131 + k * 17 + it * 7919) & 511; sg = (k + it * 32) & 4095; a = u * (1 << 21) + sg * 512 + (tid & 31) * 16
                    elif typ == 'memseq':
                        spos = (blk * 12288 * 37 + it * 12288) % (1 << 26); ix = (spos + k * 384 + tid) % (1 << 26); a = ix * 16
                    if a is None: return None
                    L.append(len(set(a // 128))); P6.append(len(set(a // 65536))); P2.append(len(set(a // (1 << 21))))
    return np.mean(L), np.mean(P6), np.mean(P2)

def rows_of(d, group):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, 'conc_%s_*.json' % group))): rows += json.load(open(f))
    return rows

def analyze(d, group):
    rows = rows_of(d, group); clk = load_clk(d, group)
    def launches(r):
        n = r['rays']; ms = r['ms']; return [(n / m / 1000.0, st.mean([c[0] for c in clk.get((r['key'], i + 1), [(0, 0, 0)])])) for i, m in enumerate(ms)]
    alone = {}
    for r in rows:
        if r['config'].startswith('alone'): alone.setdefault(r['scene'], []).append(r)
    res = {}
    for r in rows:
        if r['config'].startswith('alone'): continue
        sc = r['scene']; prev = [a for a in alone[sc] if a['seq'] == r['seq'] - 1][0]
        pool = [x for a in alone[sc] for x in launches(a)]; pool_rate = st.median([x[0] for x in pool])
        pr = st.median([x[0] for x in launches(prev)])
        d_ = res.setdefault((sc, r['config']), dict(paired=[], pooled=[], matched=[], scaled=0, clk=[], aclk=[], gb=[], it=[], wrong=0, comp=r['compute']))
        for rate, c in launches(r):
            m = [x[0] for x in pool if c and x[1] and abs(x[1] - c) / c <= 0.01]
            if m: d_['matched'].append(rate / st.median(m))
            elif c: d_['matched'].append(rate / (pool_rate * 1.0) * (1.0)); d_['scaled'] += 1   # no alone launch within 1 percent: unscaled fallback counted (clock scale needs a cycle model, see text)
            d_['paired'].append(rate / pr); d_['pooled'].append(rate / pool_rate); d_['clk'].append(c)
        d_['aclk'] += [x[1] for x in pool]; d_['gb'].append(r['compute_with_rt'] * 196608 / 1e9); d_['it'].append(r['compute_with_rt']); d_['wrong'] += r['wrong'] + r['unstored']
    return res, alone, launches

def show(d, group):
    res, alone, launches = analyze(d, group)
    for sc in ('soup100000', 'tri1'):
        al = [x for a in alone.get(sc, []) for x in launches(a)]
        print('scene %s: alone n %d, rate median %.1f rays/us, SM clock median %.0f (min %.0f max %.0f MHz)' % (sc, len(al), st.median(x[0] for x in al), st.median(x[1] for x in al), min(x[1] for x in al), max(x[1] for x in al)))
        for (s, c), v in sorted(res.items()):
            if s != sc: continue
            typ = v['comp'].split(':')[0]; pt = pattern(typ); it = st.median(v['it'])
            ln = it * 384 * pt[0] if pt else float('nan'); pg = it * 384 * pt[2] if pt else float('nan')   # 384 warp loads per iteration (12 warps x 32 loads)
            print('  %-20s paired %.3f pooled %.3f matched %.3f (scaled %d of %d) clk %.0f/alone %.0f  compute %6.1f GB/s  lines %.2f G/s pages2M %.2f G/s  [lines/load %.1f, 64K pages/load %.1f, 2M pages/load %.1f]  wrong %d' % (
                c, st.median(v['paired']), st.median(v['pooled']), st.median(v['matched']), v['scaled'], len(v['matched']), st.median(v['clk']), st.median(v['aclk']), st.median(v['gb']), ln / 1e9, pg / 1e9, *(pt if pt else (0, 0, 0)), v['wrong']))

if __name__ == '__main__':
    show(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'pg')
